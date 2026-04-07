import argparse
import os
import torch
import numpy as np
import time
import glob
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision.utils import save_image
from model_baryce import BaryCE
from tqdm import tqdm

parser = argparse.ArgumentParser(description="BaryCE-IR Testing")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--degset", default="./data/test/input/", type=str, help="degraded data directory")
parser.add_argument("--tarset", default="./data/test/target/", type=str, help="target data directory")
parser.add_argument("--savedeg", default="./results/test/DEG/", type=str, help="save path for degraded images")
parser.add_argument("--save", default="./results/test/OUT/", type=str, help="save path for output images")
parser.add_argument("--savetar", default="./results/test/TAR/", type=str, help="save path for target images")
parser.add_argument("--model", default="./checkpoint/model_BaryCE_ep001.pth", type=str, help="model path")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

# BaryCE model parameters
parser.add_argument("--num_experts", type=int, default=4, help="number of experts in MoCE")
parser.add_argument("--expert_rank", type=int, default=2, help="rank of LoRA adaptation")
parser.add_argument("--top_k", type=int, default=2, help="top-k experts to activate")
parser.add_argument("--num_classes", type=int, default=5, help="number of degradation classes")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# 创建保存目录
os.makedirs(opt.save, exist_ok=True)
os.makedirs(opt.savedeg, exist_ok=True)
os.makedirs(opt.savetar, exist_ok=True)

# 退化类型映射
DEG_TYPE_MAP = {
    'denoise': 0,
    'derain': 1,
    'dehaze': 2,
    'deblur': 3,
    'lowlight': 4
}
DEG_NAMES = ['denoise', 'derain', 'dehaze', 'deblur', 'lowlight']

def extract_degradation_type(filename):
    """从文件名中提取退化类型"""
    filename_lower = filename.lower()
    for deg_type in DEG_TYPE_MAP.keys():
        if deg_type in filename_lower:
            return deg_type
    return None

def calculate_metrics(output_img, target_img):
    """计算PSNR和SSIM"""
    # 转换为numpy数组 [0, 1]
    output_np = output_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    target_np = target_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # 裁剪到[0, 1]
    output_np = np.clip(output_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # 计算PSNR
    psnr_val = psnr_metric(target_np, output_np, data_range=1.0)
    
    # 计算SSIM
    ssim_val = ssim_metric(target_np, output_np, data_range=1.0, channel_axis=2)
    
    return psnr_val, ssim_val

# Load checkpoint and create model
print("="*70)
print(f"Loading model from: {opt.model}")
checkpoint = torch.load(opt.model, weights_only=False)

# Create model
model = BaryCE(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_enc_blocks=[4, 6, 6, 8],
    num_bary_blocks=4,
    num_moce_shared_blocks=1,
    num_tail_blocks=4,
    num_refinement_blocks=4,
    num_experts=opt.num_experts,
    expert_rank=opt.expert_rank,
    top_k=opt.top_k,
    num_classes=opt.num_classes,
    with_complexity=True
)

# Load weights
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"].state_dict(), strict=False)
    print("Model loaded successfully")
else:
    raise ValueError("Unknown checkpoint format")

model.eval()
if cuda:
    model = model.cuda()

print("="*70)

# 获取测试图像列表
deg_list = sorted(glob.glob(os.path.join(opt.degset, '*')))
tar_list = sorted(glob.glob(os.path.join(opt.tarset, '*')))

if len(deg_list) == 0:
    print(f"No images found in {opt.degset}")
    exit(1)

if len(tar_list) == 0:
    print(f"No images found in {opt.tarset}")
    exit(1)

print(f"Found {len(deg_list)} degraded images")
print(f"Found {len(tar_list)} target images")

# 按退化类型分组统计
results_by_type = {}
for deg_type in DEG_NAMES:
    results_by_type[deg_type] = {
        'psnr_list': [],
        'ssim_list': [],
        'correct_predictions': 0,
        'total_predictions': 0
    }

# 总体统计
all_psnr = []
all_ssim = []
total_correct = 0
total_predictions = 0
processed_count = 0
skipped_count = 0

# 开始测试
print("="*70)
print("Starting Testing...")
print("="*70 + "\n")

with torch.no_grad():
    for idx, (deg_path, tar_path) in tqdm(enumerate(zip(deg_list, tar_list)), total=len(deg_list), desc="Processing"):
        filename = os.path.basename(tar_path)
        
        # 提取退化类型
        deg_type = extract_degradation_type(filename)
        
        try:
            # 读取图像
            input_img = Image.open(deg_path).convert('RGB')
            target_img = Image.open(tar_path).convert('RGB')
            input_np = np.array(input_img)
            target_np = np.array(target_img)
            
            # 检查尺寸
            if input_np.shape != target_np.shape:
                print(f"[Skip] {filename}: Shape mismatch")
                skipped_count += 1
                continue
            
            # 调整尺寸为8的倍数
            h, w = input_np.shape[0], input_np.shape[1]
            h = h - (h % 8)
            w = w - (w % 8)
            
            if h == 0 or w == 0:
                print(f"[Skip] {filename}: Image too small")
                skipped_count += 1
                continue
            
            input_np = input_np[:h, :w]
            target_np = target_np[:h, :w]
            
            # 转换为tensor
            input_tensor = torch.from_numpy(input_np.transpose(2, 0, 1)).float() / 255.0
            target_tensor = torch.from_numpy(target_np.transpose(2, 0, 1)).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)
            
            if cuda:
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
            
            # 推理
            output, aux_data = model(input_tensor)
            
            # 计算指标
            psnr_val, ssim_val = calculate_metrics(output, target_tensor)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            
            # 按类型统计
            if deg_type and deg_type in results_by_type:
                results_by_type[deg_type]['psnr_list'].append(psnr_val)
                results_by_type[deg_type]['ssim_list'].append(ssim_val)
                
                # 分类准确度
                if 'degradation_probs' in aux_data:
                    pred_label = torch.argmax(aux_data['degradation_probs'], dim=1).item()
                    true_label = DEG_TYPE_MAP[deg_type]
                    
                    if pred_label == true_label:
                        results_by_type[deg_type]['correct_predictions'] += 1
                        total_correct += 1
                    
                    results_by_type[deg_type]['total_predictions'] += 1
                    total_predictions += 1
            
            # 保存结果
            save_image(input_tensor.data, os.path.join(opt.savedeg, filename))
            save_image(output.data, os.path.join(opt.save, filename))
            save_image(target_tensor.data, os.path.join(opt.savetar, filename))
            
            processed_count += 1
        
        except Exception as e:
            skipped_count += 1
            continue

print()  # 进度条后换行

# 打印详细结果
print("="*70)
print("TESTING RESULTS")
print("="*70 + "\n")

# 每种退化的详细结果
for deg_type in DEG_NAMES:
    data = results_by_type[deg_type]
    
    if len(data['psnr_list']) == 0:
        continue
    
    psnr_arr = np.array(data['psnr_list'])
    ssim_arr = np.array(data['ssim_list'])
    
    print(f"【{deg_type.upper()}】")
    print("-"*70)
    print(f"  Images Tested: {len(data['psnr_list'])}")
    print(f"  PSNR: Average={psnr_arr.mean():.4f}dB, Best={psnr_arr.max():.4f}dB, Worst={psnr_arr.min():.4f}dB")
    print(f"  SSIM: Average={ssim_arr.mean():.4f}, Best={ssim_arr.max():.4f}, Worst={ssim_arr.min():.4f}")
    
    if data['total_predictions'] > 0:
        accuracy = data['correct_predictions'] / data['total_predictions'] * 100
        print(f"  Classification Accuracy: {accuracy:.2f}% ({data['correct_predictions']}/{data['total_predictions']})")
    
    print()

# 总体统计
print("="*70)
print("OVERALL STATISTICS")
print("="*70)

if len(all_psnr) > 0:
    all_psnr = np.array(all_psnr)
    all_ssim = np.array(all_ssim)
    
    print(f"Total Images Processed: {processed_count}")
    print(f"Total Images Skipped: {skipped_count}")
    print(f"PSNR: Average={all_psnr.mean():.4f}dB, Best={all_psnr.max():.4f}dB, Worst={all_psnr.min():.4f}dB")
    print(f"SSIM: Average={all_ssim.mean():.4f}, Best={all_ssim.max():.4f}, Worst={all_ssim.min():.4f}")
    
    if total_predictions > 0:
        overall_accuracy = total_correct / total_predictions * 100
        print(f"Overall Classification Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_predictions})")
else:
    print("No images were successfully processed!")

print("="*70)
