import argparse, os, glob
import torch, pdb
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math, random, time
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_baryce import BaryCE, BaryNet, Potentials
from util.universal_dataset import TrainDataset
from torchvision.utils import save_image
from utils import unfreeze, freeze
from scipy import io as scio
import torch.nn.functional as F
import random
import cv2


class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for handling class imbalance
    Focuses on hard-to-classify samples while reducing overfitting
    """
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] - logits
            targets: [B] - class labels
        """
        # Use label smoothing to reduce overfitting
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Training settings
parser = argparse.ArgumentParser(description="PyTorch BaryCE-IR Training")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=2e-4")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=20")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default=None, type=str,
                    help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=16, help="Number of threads for data loader to use, (default: 16)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pairnum", default=10000000, type=int, help="num of paired samples")
parser.add_argument('--num_sources', type=int, default=5, help='number of source domains.')

parser.add_argument('--de_type', nargs='+', default=['denoise', 'derain', 'dehaze', 'deblur', 'lowlight'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_dir', type=str, default='data/Train/Deblur/',
                    help='where training images of dehazing saves.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/lowlight/',
                    help='where training images of deraining saves.')

parser.add_argument("--degset", default="./data/val/Derain/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/val/Derain/target/", type=str, help="target data")
parser.add_argument("--Sigma", default=1, type=float, help="weight for L1 loss")
parser.add_argument("--sigma", default=1, type=float, help="noise standard deviation")
parser.add_argument("--optimizer", default="Adam", type=str, help="optimizer type")
parser.add_argument("--type", default="Deraining", type=str, help="to distinguish the ckpt name")
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--backbone', type=str, default='BaryCE', help='backbone model name (BaryCE or BaryNet)')

# BaryCE specific parameters
# 简化的损失权重配置：
# - L1: 1.0 (重建质量，基准)
# - Classification: 2.0 (分类准确性，需要强调)
# - MoCE: 1.0 (专家平衡)
# - Orthogonality: 0.5 (特征解耦，辅助约束)
# - Label smoothing: 0.1 (减少过拟合)
parser.add_argument('--lambda_moce', type=float, default=1.0, help='MoCE auxiliary loss weight')
parser.add_argument('--lambda_cls', type=float, default=2.0, help='Classification loss weight')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing for classification')
parser.add_argument('--num_experts', type=int, default=4, help='number of experts in MoCE')
parser.add_argument('--expert_rank', type=int, default=2, help='rank of LoRA adaptation in experts')
parser.add_argument('--top_k', type=int, default=1, help='top-k experts to activate')
parser.add_argument('--num_classes', type=int, default=5, help='number of degradation classes')


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def main():
    global opt, model, Lambda, K

    opt = parser.parse_args()
    print(opt)

    K = opt.num_sources
    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("------Creating BaryCE-IR Model------")
    model = BaryCE(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_enc_blocks=[4, 6, 6, 8],
        num_bary_blocks=8,
        num_moce_shared_blocks=1,
        num_tail_blocks=4,
        num_refinement_blocks=4,
        num_experts=opt.num_experts,
        expert_rank=opt.expert_rank,
        top_k=opt.top_k,
        num_classes=opt.num_classes,
        with_complexity=True
    )

    total_params, trainable_params = get_parameter_number(model)
    print(f"Model parameters: Total={total_params/1e6:.2f}M, Trainable={trainable_params/1e6:.2f}M")

    print("------Network constructed------")
    if cuda:
        model = model.cuda()

    # Potentials module (for multi-domain learning)
    channels_latent = 384
    Pots = Potentials(num_potentials=opt.num_sources, channels=channels_latent, size=opt.patch_size)
    print("------Potentials constructed------")
    if cuda:
        Pots = Pots.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, weights_only=False)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict(), strict=False)
            if "Pots" in checkpoint:
                try:
                    Pots.load_state_dict(checkpoint["Pots"].state_dict(), strict=False)
                except:
                    print("=> Potentials loading failed, using fresh initialization")
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained, weights_only=False)
            try:
                model.load_state_dict(weights['model'].state_dict(), strict=False)
                print("=> Model loaded with strict=False")
            except Exception as e:
                print(f"=> Model loading failed: {e}")
            
            if 'Pots' in weights:
                try:
                    Pots.load_state_dict(weights['Pots'].state_dict(), strict=False)
                    print("=> Potentials loaded with strict=False")
                except:
                    print("=> Potentials loading failed, using fresh initialization")
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("------Using Optimizer: '{}' ------".format(opt.optimizer))

    if opt.optimizer == 'Adam':
        model_optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
        Pots_optimizer = torch.optim.Adam(Pots.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    elif opt.optimizer == 'AdamW':
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
        Pots_optimizer = torch.optim.AdamW(Pots.parameters(), lr=opt.lr, weight_decay=1e-4)
    elif opt.optimizer == 'RMSprop':
        model_optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr)
        Pots_optimizer = torch.optim.RMSprop(Pots.parameters(), lr=opt.lr)

    print("------Training------")
    MSE = []
    ModelLOSS = []
    PotLOSS = []
    train_set = TrainDataset(opt)
    domain_sample_counts = train_set.get_num_samples()
    print(domain_sample_counts)
    
    valid_counts = [count for count in domain_sample_counts if count > 0]
    if len(valid_counts) == 0:
        raise ValueError("No training data found. Please check data paths.")
    
    inverse_counts = [1 / count for count in valid_counts]
    total_inverse = sum(inverse_counts)
    Lambda = [inv_count / total_inverse for inv_count in inverse_counts]
    print(Lambda)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads,
                                      batch_size=opt.batchSize, shuffle=True)
    num = 0
    
    # Define validation datasets
    validation_datasets = {}
    
    denoise_target = './data/val/Denoise/target/'
    if os.path.exists(denoise_target):
        validation_datasets['denoise'] = (denoise_target, denoise_target, 'denoise')
    
    other_datasets = {
        'derain': ('./data/val/Derain/input/', './data/val/Derain/target/'),
        'dehaze': ('./data/val/Dehaze/input/', './data/val/Dehaze/target/'),
        'deblur': ('./data/val/Deblur/input/', './data/val/Deblur/target/'),
        'lowlight': ('./data/val/lowlight/input/', './data/val/lowlight/target/')
    }
    
    for deg_type, (deg_path, tar_path) in other_datasets.items():
        if os.path.exists(deg_path) and os.path.exists(tar_path):
            validation_datasets[deg_type] = (deg_path, tar_path, 'other')
    
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        model_loss = 0
        pot_loss = 0
        a, b = train(training_data_loader, model_optimizer, Pots_optimizer, model, Pots, epoch)

        # Validate on all degradation types
        print('----------Multi-domain Validation-----------')
        all_results = {}
        for deg_name, (deg_path, tar_path, deg_type) in validation_datasets.items():
            deg_exists = os.path.exists(deg_path)
            tar_exists = os.path.exists(tar_path)
            
            if deg_exists and tar_exists:
                deg_list = sorted(glob.glob(deg_path + "*"))
                tar_list = sorted(glob.glob(tar_path + "*"))
                
                if deg_list and tar_list:
                    psnr_val, ssim_val, acc_val = evaluate(model, deg_list, tar_list, deg_name, deg_type)
                    all_results[deg_name] = (psnr_val, ssim_val, acc_val)
                    print(f"  {deg_name.upper()}: PSNR={psnr_val:.4f}    SSIM={ssim_val:.4f}    Classification Accuracy={acc_val:.2f}%")
                else:
                    print(f"  {deg_name.upper()}: [SKIP] No files found")
            else:
                print(f"  {deg_name.upper()}: [SKIP] Path not found")
        
        # Save validation results
        os.makedirs("./checksample/all/", exist_ok=True)
        with open("./checksample/all/validation_results.txt", "a") as f:
            f.write(f"Epoch {epoch}: ")
            for deg_name, (psnr_val, ssim_val, acc_val) in all_results.items():
                f.write(f"{deg_name}=(PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, Acc={acc_val:.2f}%) ")
            f.write(f"(BatchSize {opt.batchSize})\n")

        model_loss += a
        pot_loss += b
        num += 1
        model_loss = model_loss / num
        ModelLOSS.append(format(model_loss))
        PotLOSS.append(format(pot_loss))
        scio.savemat('ModelLOSS.mat', {'ModelLOSS': ModelLOSS})
        scio.savemat('PotLOSS.mat', {'PotLOSS': PotLOSS})
        save_checkpoint(model, Pots, epoch)


def evaluate(model, deg_list, tar_list, dataset_name="default", deg_type="other"):
    cuda = True
    psnr_sum = 0
    ssim_sum = 0
    correct = 0
    total = 0
    
    deg_type_map = {
        'denoise': 0,
        'derain': 1,
        'dehaze': 2,
        'deblur': 3,
        'lowlight': 4
    }
    true_label = deg_type_map.get(dataset_name, 0)
    
    print(f'  Validating {dataset_name}...')
    model.eval()
    with torch.no_grad():
        for deg_name, tar_name in zip(deg_list, tar_list):
            if deg_type == 'denoise':
                noisy_dir = os.path.dirname(tar_name).replace('target', 'input')
                noisy_path = os.path.join(noisy_dir, os.path.basename(tar_name))
                
                if os.path.exists(noisy_path):
                    deg_img = Image.open(noisy_path).convert('RGB')
                    deg_img = np.array(deg_img).astype(np.float32) / 255.0
                else:
                    tar_img_temp = Image.open(tar_name).convert('RGB')
                    tar_img_temp = np.array(tar_img_temp).astype(np.float32) / 255.0
                    noise_levels = [15, 25, 50]
                    noise_level = random.choice(noise_levels)
                    noise = np.random.normal(0, noise_level / 255.0, tar_img_temp.shape)
                    deg_img = np.clip(tar_img_temp + noise, 0, 1)
                
                tar_img = Image.open(tar_name).convert('RGB')
                tar_img = np.array(tar_img).astype(np.float32) / 255.0
            else:
                deg_img = Image.open(deg_name).convert('RGB')
                deg_img = np.array(deg_img).astype(np.float32) / 255.0
                tar_img = Image.open(tar_name).convert('RGB')
                tar_img = np.array(tar_img).astype(np.float32) / 255.0
            
            h, w = tar_img.shape[0], tar_img.shape[1]
            shape1 = deg_img.shape
            shape2 = tar_img.shape
            
            if (h % 8) != 0 or (w % 8) != 0:
                continue
            if shape1 != shape2:
                continue
            
            deg_img = np.transpose(deg_img, (2, 0, 1))
            deg_img = torch.from_numpy(deg_img).float()
            deg_img = deg_img.unsqueeze(0)
            data_degraded = deg_img

            tar_img = np.transpose(tar_img, (2, 0, 1))
            tar_img = torch.from_numpy(tar_img).float()
            tar_img = tar_img.unsqueeze(0)
            gt = tar_img
            
            if cuda:
                model = model.cuda()
                gt = gt.cuda()
                data_degraded = data_degraded.cuda()
            else:
                model = model.cpu()

            im_output, aux_data = model(data_degraded)
            
            if 'degradation_probs' in aux_data:
                pred_label = torch.argmax(aux_data['degradation_probs'], dim=1).item()
                if pred_label == true_label:
                    correct += 1
                total += 1
            
            im_output = im_output.squeeze(0).cpu()
            tar_img = tar_img.squeeze(0).cpu()

            im_output = im_output.numpy()
            tar_img = tar_img.numpy()
            im_output = np.transpose(im_output, (1, 2, 0))
            tar_img = np.transpose(tar_img, (1, 2, 0))
            im_output = np.clip(im_output, 0, 1)
            
            psnr_sum += psnr(im_output, tar_img, data_range=1)
            ssim_val = ssim(im_output, tar_img, data_range=1, channel_axis=2)
            ssim_sum += ssim_val
        
        avg_psnr = psnr_sum / len(deg_list) if len(deg_list) > 0 else 0
        avg_ssim = ssim_sum / len(deg_list) if len(deg_list) > 0 else 0
        accuracy = correct / total * 100 if total > 0 else 0
    
    model.train()
    return avg_psnr, avg_ssim, accuracy


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


def train(training_data_loader, model_optimizer, Pots_optimizer, model, Pots, epoch):
    lr = adjust_learning_rate(model_optimizer, epoch - 1)
    
    # Loss tracking (简化版)
    loss_tracker = {
        'total': [],
        'l1': [],
        'cls': [],
        'moce': [],
        'ort': [],
        'pots': []
    }
    
    # Focal Loss with Label Smoothing for classification
    # alpha=1.0: balanced weight for all classes
    # gamma=2.0: focus on hard samples
    # label_smoothing: reduces overfitting and improves generalization
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=opt.label_smoothing)

    for param_group in model_optimizer.param_groups:
        param_group["lr"] = lr
    for param_group in Pots_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, model_optimizer.param_groups[0]["lr"]))

    for iteration, batch in enumerate(training_data_loader):
        ([clean_name, de_id], degraded, target) = batch

        if opt.cuda:
            target = target.cuda()
            degraded = degraded.cuda()
        
        # Remap de_id to match classifier labels
        # de_id: 0=denoise, 1=derain, 2=dehaze, 3=deblur, 4=lowlight, 5=single
        de_id_remapped = de_id.clone() if isinstance(de_id, torch.Tensor) else torch.tensor(de_id, dtype=torch.long)
        
        if opt.cuda:
            de_id_remapped = de_id_remapped.cuda()

        # ========== Model optimization ==========
        freeze(Pots)
        unfreeze(model)

        model.zero_grad()
        out_restored, aux_data = model(degraded)

        # ========================================================================
        # 核心损失函数设计（简洁鲁棒版）
        # ========================================================================
        
        # 1. 重建损失 (L1 Loss) - 最重要的损失
        # 目标：确保输出图像质量
        l1_loss = F.l1_loss(out_restored, target)
        
        # 2. 分类损失 (Classification Loss with Focal Loss)
        # 目标：准确识别退化类型
        # 使用 Focal Loss 处理类别不平衡问题
        degradation_logits = aux_data.get('degradation_logits')
        if degradation_logits is not None:
            cls_loss = focal_loss_fn(degradation_logits, de_id_remapped)
        else:
            cls_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # 3. MoCE 辅助损失 (Load Balancing + Importance)
        # 目标：平衡专家负载，防止专家崩溃
        moce_loss = aux_data.get('moce_loss', torch.tensor(0.0, device=l1_loss.device))
        if not isinstance(moce_loss, torch.Tensor):
            moce_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # 4. 特征正交性损失 (Feature Orthogonality)
        # 目标：确保退化特征和内容特征解耦
        # 只在有明确物理意义的地方使用
        degradation_feat = aux_data['degradation_feat']
        bary_latent = aux_data['unified_latent']
        
        B = degradation_feat.size(0)
        deg_flat = degradation_feat.view(B, -1)
        bary_flat = bary_latent.view(B, -1)
        
        # 余弦相似度应该接近0（正交）
        cos_sim = F.cosine_similarity(deg_flat, bary_flat, dim=1).abs().mean()
        orthogonality_loss = cos_sim
        
        # Potential Loss (per-sample)
        pot_loss_total = torch.tensor(0.0, device=l1_loss.device)
        for i in range(out_restored.shape[0]):
            source_id_i = de_id[i].item() if isinstance(de_id[i], torch.Tensor) else de_id[i]
            pot_loss = Pots(out_restored[i:i+1], source_id_i)
            pot_loss_total = pot_loss_total - Lambda[source_id_i] * pot_loss
        
        pot_loss_total = pot_loss_total / out_restored.shape[0]
        
        # ========================================================================
        # 总损失 (Total Loss) - 简洁的权重配置
        # ========================================================================
        # 权重设计原则：
        # - L1 Loss: 1.0 (基准，最重要)
        # - Classification: 2.0 (需要强调，提高分类准确率)
        # - MoCE: 1.0 (专家平衡)
        # - Orthogonality: 0.5 (辅助约束，不要太强)
        # - Potentials: 自适应权重 Lambda
        
        total_loss = (
            1.0 * l1_loss +                      # 重建质量
            opt.lambda_cls * cls_loss +          # 分类准确性 (default: 2.0)
            opt.lambda_moce * moce_loss +        # 专家平衡 (default: 1.0)
            0.5 * orthogonality_loss +           # 特征解耦
            pot_loss_total                       # 多域学习
        )
        
        # Track losses (简化版)
        loss_tracker['total'].append(total_loss.item())
        loss_tracker['l1'].append(l1_loss.item())
        loss_tracker['cls'].append(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else 0.0)
        loss_tracker['moce'].append(moce_loss.item() if isinstance(moce_loss, torch.Tensor) else 0.0)
        loss_tracker['ort'].append(orthogonality_loss.item())
        loss_tracker['pots'].append(pot_loss_total.item())
        
        total_loss.backward()
        
        # 梯度裁剪（简化版）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        model_optimizer.step()

        # ========== Potential optimization ==========
        unfreeze(Pots)
        freeze(model)

        if iteration % 1 == 0:
            Pots.zero_grad()
            potential_train_loss = torch.tensor(0.0, device=l1_loss.device)
            
            with torch.no_grad():
                out_restored_temp, _ = model(degraded)
            
            for i in range(out_restored_temp.shape[0]):
                source_id_i = de_id[i].item() if isinstance(de_id[i], torch.Tensor) else de_id[i]
                potential_loss = Pots(out_restored_temp[i:i+1], source_id_i)
                potential_train_loss = potential_train_loss + Lambda[source_id_i] * potential_loss

            potential = potential_train_loss / out_restored_temp.shape[0]
            potential.backward()
            Pots_optimizer.step()

        # Potential constraint
        Pots.zero_grad()
        potential_constraint = torch.tensor(0.0, device=l1_loss.device)
        with torch.no_grad():
            out_restored, _ = model(degraded)
        
        for j in range(len(Lambda)):
            potential_constraint = potential_constraint + Lambda[j] * Pots(out_restored[0:1], j)

        potential_constraint_loss = 10 * (potential_constraint ** 2)
        potential_constraint_loss.backward()
        Pots_optimizer.step()

        # ========== Logging ==========
        if iteration % 10 == 0:
            # Calculate recent averages
            n_recent = min(10, len(loss_tracker['l1']))
            avg_total = np.mean(loss_tracker['total'][-n_recent:])
            avg_l1 = np.mean(loss_tracker['l1'][-n_recent:])
            avg_cls = np.mean(loss_tracker['cls'][-n_recent:])
            avg_moce = np.mean(loss_tracker['moce'][-n_recent:])
            avg_ort = np.mean(loss_tracker['ort'][-n_recent:])
            avg_pots = np.mean(loss_tracker['pots'][-n_recent:])
            
            print(f"Epoch {epoch}({iteration}/{len(training_data_loader)}): "
                  f"Total={avg_total:.4f} | "
                  f"L1={avg_l1:.4f} | "
                  f"Cls={avg_cls:.4f} | "
                  f"MoCE={avg_moce:.4f} | "
                  f"Ort={avg_ort:.4f} | "
                  f"Pots={avg_pots:.4f}")
            
            try:
                os.makedirs('./checksample/' + opt.type + '/', exist_ok=True)
                save_image(out_restored.data, './checksample/' + opt.type + '/output.png')
                save_image(degraded.data, './checksample/' + opt.type + '/degraded.png')
                save_image(target.data, './checksample/' + opt.type + '/target.png')
            except Exception as e:
                pass
    
    return torch.mean(torch.FloatTensor(loss_tracker['total'])), torch.mean(torch.FloatTensor(loss_tracker['pots']))


def save_checkpoint(model, Pots, epoch):
    """Save checkpoint with epoch number"""
    backbone_name = opt.backbone if hasattr(opt, 'backbone') else 'BaryCE'
    model_out_path = "checkpoint/" + "model_{}_ep{:03d}.pth".format(backbone_name, epoch)
    
    state = {"epoch": epoch, "model": model, "Pots": Pots}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


if __name__ == "__main__":
    main()
