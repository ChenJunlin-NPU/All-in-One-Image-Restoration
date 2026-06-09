"""
BaryCE-OT 测试代码 (传统OT版本)
兼容传统OT版本 Model_BaryCE 的 9 个 forward 返回值。
"""

import argparse
import os
import torch
import numpy as np
import time, math, glob
from PIL import Image
from evaluate import calculate_evaluation_floder
import torchvision
from torchvision.utils import save_image
import pytorch_fid.fid_score as fid_score

parser = argparse.ArgumentParser(description="BaryCE Testing")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--degset", default="./data/test/Deblur/GoPro/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/Deblur/GoPro/target/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/Dehaze/SOTS/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/Dehaze/SOTS/target/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/Denoise/BSD68/noisy15/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/Denoise/BSD68/noisy15/target/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/Denoise/BSD68/noisy25/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/Denoise/BSD68/noisy25/target/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/Denoise/BSD68/noisy50/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/Denoise/BSD68/noisy50/target/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/Derain/Rain100L/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/Derain/Rain100L/target/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/lowlight/LOLv1/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/lowlight/LOLv1/target/", type=str, help="target data")
parser.add_argument("--save", default="./results/deblur/OUT/", type=str, help="savepath, Default: results")
parser.add_argument("--savetar", default="./results/deblur/TAR/", type=str, help="savepath, Default: targets")
parser.add_argument("--model", default="./checkpoint/model_BaryCE_all_128_100.pth", type=str, help="model path")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--compute_fid", action="store_true", default=False, help="compute FID score")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)


def main():
    opt = parser.parse_args()
    print(opt)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
    cuda = opt.cuda

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # Create output directories
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    if not os.path.exists(opt.savetar):
        os.makedirs(opt.savetar)

    # Load model
    print("=> Loading model from '{}'".format(opt.model))
    checkpoint = torch.load(opt.model)
    
    if "Model" in checkpoint:
        Model = checkpoint["Model"]
    else:
        # Handle old checkpoint format
        from model import Model_BaryCE
        Model = Model_BaryCE(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            num_degradations=5
        )
        Model.load_state_dict(checkpoint)
    
    print("=> Model loaded successfully")

    # Get test image lists
    deg_list = sorted(glob.glob(opt.degset + "*"))
    tar_list = sorted(glob.glob(opt.tarset + "*"))
    
    print(f"=> Found {len(deg_list)} degraded images")
    print(f"=> Found {len(tar_list)} target images")
    
    if len(deg_list) == 0:
        raise Exception(f"No images found in {opt.degset}")
    
    num = len(deg_list)

    # Testing
    Model.eval()
    
    with torch.no_grad():
        for idx, (deg_name, tar_name) in enumerate(zip(deg_list, tar_list)):
            name = tar_name.split('/')[-1]
            print(f"[{idx+1}/{num}] Processing {name}")
            
            # Load images
            deg_img = Image.open(deg_name).convert('RGB')
            tar_img = Image.open(tar_name).convert('RGB')
            deg_img = np.array(deg_img)
            tar_img = np.array(tar_img)

            h, w = deg_img.shape[0], deg_img.shape[1]
            shape1 = deg_img.shape
            shape2 = tar_img.shape
            
            # Ensure dimensions are divisible by 8
            while (h % 8) != 0:
                h = h - 1
                deg_img = deg_img[0:h, :]
                tar_img = tar_img[0:h, :]
            while (w % 8) != 0:
                w = w - 1
                deg_img = deg_img[:, 0:w]
                tar_img = tar_img[:, 0:w]
            
            if shape1 != shape2:
                print(f"Warning: Shape mismatch for {name}, skipping...")
                continue
            
            # Convert to tensor
            deg_img = np.transpose(deg_img, (2, 0, 1))
            deg_img = torch.from_numpy(deg_img).float() / 255
            deg_img = deg_img.unsqueeze(0)
            
            tar_img = np.transpose(tar_img, (2, 0, 1))
            tar_img = torch.from_numpy(tar_img).float() / 255
            tar_img = tar_img.unsqueeze(0)
            
            data_degraded = deg_img
            gt = tar_img
            
            if cuda:
                Model = Model.cuda()
                gt = gt.cuda()
                data_degraded = data_degraded.cuda()
            else:
                Model = Model.cpu()

            start_time = time.time()

            # Forward pass
            model_outputs = Model(data_degraded)
            im_output = model_outputs[0] if isinstance(model_outputs, (tuple, list)) else model_outputs
            
            elapsed_time = time.time() - start_time
            
            # Save results
            save_image(im_output.data, opt.save + '/' + name)
            save_image(tar_img.data, opt.savetar + '/' + name)
            
            if idx == 0:
                print(f"=> Processing time: {elapsed_time:.4f}s per image")

    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60)

    # Compute metrics
    print("\n=> Computing metrics...")
    psnr_val, ssim_val, pmax, smax, pmin, smin = calculate_evaluation_floder(opt.savetar, opt.save)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print("PSNR: Average {:.4f} dB,   Best {:.4f} dB,   Worst {:.4f} dB".format(psnr_val, pmax, pmin))
    print("SSIM: Average {:.4f},      Best {:.4f},      Worst {:.4f}".format(ssim_val, smax, smin))
    
    # Compute FID if requested
    if opt.compute_fid:
        print("\n=> Computing FID score...")
        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [opt.savetar, opt.save], 
                batch_size=1,
                device='cuda' if cuda else 'cpu', 
                dims=2048, 
                num_workers=8
            )
            print('FID value: {:.4f}'.format(fid_value))
        except Exception as e:
            print(f"Warning: Failed to compute FID score: {e}")
    
    print("="*60)
    
    # Save results to file
    result_file = opt.save + '/results.txt'
    with open(result_file, 'w') as f:
        f.write("BaryCE Testing Results\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {opt.model}\n")
        f.write(f"Test set: {opt.degset}\n")
        f.write(f"Number of images: {num}\n")
        f.write("="*60 + "\n")
        f.write(f"PSNR: Average {psnr_val:.4f} dB,   Best {pmax:.4f} dB,   Worst {pmin:.4f} dB\n")
        f.write(f"SSIM: Average {ssim_val:.4f},      Best {smax:.4f},      Worst {smin:.4f}\n")
        if opt.compute_fid:
            f.write(f"FID: {fid_value:.4f}\n")
    
    print(f"\n=> Results saved to {result_file}")


if __name__ == "__main__":
    main()
