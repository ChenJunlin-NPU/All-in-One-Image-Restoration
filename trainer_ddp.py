"""
BaryCE-2 训练代码（多卡DDP版本）
基于 BaryIR 的训练框架，增加了：
1. 退化分类器的分类损失
2. MoCE 的 importance loss 和 load balance loss
3. 保留 BaryIR 的 MWB、IRC、BRO 损失
"""

import argparse, os, glob
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import math, random, time
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import Model_BaryCE, Potentials
from util.universal_dataset import TrainDataset
from torchvision.utils import save_image
from utils import unfreeze, freeze
from scipy import io as scio
import torch.nn.functional as F

# Training settings
parser = argparse.ArgumentParser(description="BaryCE-2 Training DDP")
parser.add_argument("--batchSize", type=int, default=6, help="training batch size per GPU")
parser.add_argument("--nEpochs", type=int, default=60, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate")
parser.add_argument("--step", type=int, default=31, help="learning rate decay step")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default=None, type=str, help="Path to resume model")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number")
parser.add_argument("--threads", type=int, default=16, help="Number of threads for data loader")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--pairnum", default=10000000, type=int, help="num of paired samples")
parser.add_argument('--num_sources', type=int, default=5, help='number of source domains')
parser.add_argument('--num_degradations', type=int, default=5, help='number of degradation types for classifier')

# Dataset paths
parser.add_argument('--de_type', nargs='+', 
                    default=['denoise', 'derain', 'dehaze', 'deblur', 'lowlight'],
                    help='degradation types for training')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/')
parser.add_argument('--deblur_dir', type=str, default='data/Train/Deblur/')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/lowlight/')

# Validation path
parser.add_argument("--degset", default="./data/test/derain/Rain100L/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/derain/Rain100L/target/", type=str, help="target data")

# Loss weights
parser.add_argument("--Sigma", default=10000, type=float, help="weight for L1 loss")
parser.add_argument("--lambda_cls", default=1.0, type=float, help="weight for classification loss")
parser.add_argument("--lambda_importance", default=0.01, type=float, help="weight for MoCE importance loss")
parser.add_argument("--lambda_load", default=0.01, type=float, help="weight for MoCE load balance loss")

parser.add_argument("--optimizer", default="RMSprop", type=str, help="optimizer type")
parser.add_argument("--type", default="all", type=str, help="experiment name")
parser.add_argument('--patch_size', type=int, default=128, help='patch size of input')
parser.add_argument('--data_file_dir', type=str, default='data_dir/')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")


def is_main_process():
    return dist.get_rank() == 0


def main():
    global opt, Model, Lambda, K

    opt = parser.parse_args()

    # DDP setup
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0

    if opt.cuda:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    
    if is_main_process():
        print(opt)

    K = opt.num_sources

    # Seed synchronization
    opt.seed = random.randint(1, 10000)
    if is_main_process():
        print("Random Seed: ", opt.seed)
    
    seed_tensor = torch.tensor(opt.seed, device=device)
    dist.broadcast(seed_tensor, 0)
    opt.seed = seed_tensor.item()
    
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    if is_main_process():
        print("------Datasets loaded------")
    
    # Create model
    Model = Model_BaryCE(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        num_degradations=opt.num_degradations
    )
    
    if is_main_process():
        print("------Model_BaryCE constructed------")
    
    if opt.cuda:
        Model = Model.to(device)

    # Resume from checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            if is_main_process():
                print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=device)
            opt.start_epoch = checkpoint["epoch"] + 1
            
            state_dict = checkpoint["Model"].state_dict()
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            Model.load_state_dict(new_state_dict)
        else:
            if is_main_process():
                print("=> no checkpoint found at '{}'".format(opt.resume))

    # Load pretrained model (only Model, Pots will be loaded later)
    pretrained_pots_weights = None
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            if is_main_process():
                print("=> loading pretrained model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained, map_location=device)
            
            state_dict_m = weights['model'].state_dict() if 'model' in weights else weights
            new_state_dict_m = {k.replace("module.", ""): v for k, v in state_dict_m.items()}
            Model.load_state_dict(new_state_dict_m, strict=False)

            # Save Pots weights for later loading
            if 'discr' in weights:
                pretrained_pots_weights = weights['discr'].state_dict()
        else:
            if is_main_process():
                print("=> no pretrained model found at '{}'".format(opt.pretrained))

    # Wrap Model with DDP (Pots will be wrapped later after initialization)
    Model = DDP(Model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Model optimizer (Pots optimizer will be created later)
    if is_main_process():
        print("------Using Optimizer: '{}' ------".format(opt.optimizer))

    if opt.optimizer == 'Adam':
        Model_optimizer = torch.optim.Adam(Model.parameters(), lr=opt.lr/2)
    elif opt.optimizer == 'RMSprop':
        Model_optimizer = torch.optim.RMSprop(Model.parameters(), lr=opt.lr/2)

    if is_main_process():
        print("------Training------")
    
    ModelLOSS = []
    PotLOSS = []
    
    # 先加载数据集，获取实际的退化类型数量
    train_set = TrainDataset(opt, noise_combine=True)
    domain_sample_counts = train_set.get_num_samples()
    
    if is_main_process():
        print("Domain sample counts:", domain_sample_counts)
    
    # 动态设置 K 为实际的退化类型数量
    K = len(domain_sample_counts)
    if is_main_process():
        print(f"Actual number of degradation types (K): {K}")
    
    # 根据实际的 K 初始化 Potentials 网络
    channels_latent = 384
    Pots = Potentials(num_potentials=K, channels=channels_latent, size=opt.patch_size)
    
    if is_main_process():
        print(f"------Potentials network constructed with {K} potentials------")
    
    if opt.cuda:
        Pots = Pots.to(device)
    
    # Load pretrained Pots weights if available
    if pretrained_pots_weights is not None:
        new_state_dict_d = {k.replace("module.", ""): v for k, v in pretrained_pots_weights.items()}
        Pots.load_state_dict(new_state_dict_d, strict=False)
        if is_main_process():
            print("=> Loaded pretrained Pots weights")
    
    # 计算 Lambda 权重
    inverse_counts = [1 / count for count in domain_sample_counts]
    total_inverse = sum(inverse_counts)
    Lambda = [inv_count / total_inverse for inv_count in inverse_counts]
    
    if is_main_process():
        print("Lambda weights:", Lambda)
    
    # Wrap Pots with DDP
    Pots = DDP(Pots, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Create Pots optimizer
    if opt.optimizer == 'Adam':
        Pots_optimizer = torch.optim.Adam(Pots.parameters(), lr=opt.lr)
    elif opt.optimizer == 'RMSprop':
        Pots_optimizer = torch.optim.RMSprop(Pots.parameters(), lr=opt.lr)

    train_sampler = DistributedSampler(train_set, shuffle=True)
    
    training_data_loader = DataLoader(
        dataset=train_set, 
        num_workers=opt.threads, 
        batch_size=opt.batchSize, 
        sampler=train_sampler,
        pin_memory=True
    )

    deg_list = glob.glob(opt.degset + "*")
    deg_list = sorted(deg_list)
    tar_list = sorted(glob.glob(opt.tarset + "*"))
    
    num = 0

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train_sampler.set_epoch(epoch)
        train_set.set_epoch(epoch)
        
        Modelloss = 0
        Ploss = 0
        a, b = train(training_data_loader, Model_optimizer, Pots_optimizer, Model, Pots, epoch, local_rank)

        if is_main_process():
            p = evaluate(Model, deg_list, tar_list, device)
            
            if not os.path.exists("./checksample/all/"):
                os.makedirs("./checksample/all/")
                
            with open("./checksample/all/validation_results.txt", "a") as f:
                f.write(
                    f"BaryCE-2 Patchsize {opt.patch_size} Epoch {epoch}, psnr {p:.4f}, Batchsize {opt.batchSize}\n"
                )

            Modelloss += a
            Ploss += b
            num += 1
            Modelloss = Modelloss / num
            ModelLOSS.append(format(Modelloss))
            PotLOSS.append(format(Ploss))
            
            scio.savemat('ModelLOSS.mat', {'ModelLOSS': ModelLOSS})
            scio.savemat('PotLOSS.mat', {'PotLOSS': PotLOSS})
            
            save_checkpoint(Model, Pots, epoch)
        
        dist.barrier()


def evaluate(Model, deg_list, tar_list, device):
    pp = 0
    print('----------validating-----------')
    Model.eval()
    
    with torch.no_grad():
        for deg_name, tar_name in zip(deg_list, tar_list):
            deg_img = Image.open(deg_name).convert('RGB')
            tar_img = Image.open(tar_name).convert('RGB')
            deg_img = np.array(deg_img)
            tar_img = np.array(tar_img)
            h, w = deg_img.shape[0], deg_img.shape[1]
            shape1 = deg_img.shape
            shape2 = tar_img.shape
            if (h % 4) or (w % 4) != 0:
                continue
            if shape1 != shape2:
                continue
            deg_img = np.transpose(deg_img, (2, 0, 1))
            deg_img = torch.from_numpy(deg_img).float() / 255
            deg_img = deg_img.unsqueeze(0)
            data_degraded = deg_img

            tar_img = np.transpose(tar_img, (2, 0, 1))
            tar_img = torch.from_numpy(tar_img).float() / 255
            tar_img = tar_img.unsqueeze(0)
            gt = tar_img
            
            gt = gt.to(device)
            data_degraded = data_degraded.to(device)

            im_output, _, _, _, _, _, _, _, _ = Model(data_degraded)
            im_output = im_output.squeeze(0).cpu()
            tar_img = tar_img.squeeze(0).cpu()

            im_output = im_output.numpy()
            tar_img = tar_img.numpy()
            im_output = np.transpose(im_output, (1, 2, 0))
            tar_img = np.transpose(tar_img, (1, 2, 0))
            pp += psnr(im_output, tar_img, data_range=1)
        p = pp / len(deg_list)
        return p


def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def train(training_data_loader, Model_optimizer, Pots_optimizer, Model, Pots, epoch, local_rank):
    lr = adjust_learning_rate(Pots_optimizer, epoch - 1)
    
    Model.train()
    Pots.train()

    for param_group in Model_optimizer.param_groups:
        param_group["lr"] = lr / 2
    for param_group in Pots_optimizer.param_groups:
        param_group["lr"] = lr / 2

    if is_main_process():
        print("Epoch={}, lr={}".format(epoch, Pots_optimizer.param_groups[0]["lr"]))

    epoch_model_loss = 0.0
    epoch_pot_loss = 0.0
    count = 0

    for iteration, batch in enumerate(training_data_loader):
        ([clean_name, de_id], degraded, target) = batch

        if opt.cuda:
            target = target.cuda(local_rank, non_blocking=True)
            degraded = degraded.cuda(local_rank, non_blocking=True)
            de_id = de_id.cuda(local_rank, non_blocking=True)

        # ========== Model optimization ==========
        freeze(Pots)
        unfreeze(Model)

        Model_optimizer.zero_grad()
        
        # Forward pass (9 outputs for model.py)
        (out_restored, f_deg, b, z, deg_logits,
         routing_weights, complexity_importance, moe_loss_imp, moe_loss_load) = Model(degraded)

        # 1. L1 reconstruction loss
        diff = out_restored - target
        l1_loss = torch.mean(abs(diff))

        # 2. BaryIR losses (MWB, IRC, BRO)
        bary_loss = 0
        mse_loss = 0
        ort_loss = 0
        contra_loss = 0

        # Use no_sync for Pots during Model loss calculation
        with Pots.no_sync():
            for i in range(out_restored.shape[0]):
                source_id_i = de_id[i]
                
                f_deg_slice_i = f_deg[i, :]
                b_slice_i = b[i, :]
                z_slice_i = z[i, :]
                res_bary_slice_i = f_deg_slice_i

                # MWB loss
                mse_loss_i = torch.mean((abs(res_bary_slice_i)) ** 2) ** 0.5

                # IRC loss (orthogonality)
                zc = F.normalize(b_slice_i.reshape(-1), dim=0)
                orth = 0
                for j in range(out_restored.shape[0]):
                    z_slice_j = z[j, :]
                    b_slice_j = b[j, :]
                    res_bary_slice_j = f_deg[j, :]
                    zs = F.normalize(res_bary_slice_j.reshape(-1), dim=0)
                    inner_product = torch.sum(zc * zs)
                    orth += inner_product ** 2
                ort_loss_i = orth

                # BRO loss (contrastive)
                zi = F.normalize(res_bary_slice_i.reshape(-1), dim=0)
                pos = neg = 0
                for j in range(out_restored.shape[0]):
                    source_id_j = de_id[j]
                    z_slice_j = z[j, :]
                    b_slice_j = b[j, :]
                    res_bary_slice_j = f_deg[j, :]
                    zj = F.normalize(res_bary_slice_j.reshape(-1), dim=0)
                    
                    if source_id_i == source_id_j:
                        pos = pos + torch.mean(torch.exp(torch.sum(zi * zj) / 0.07))
                    else:
                        neg = neg + torch.mean(torch.exp(torch.sum(zi * zj) / 0.07))
                contra_loss_i = -torch.log((pos + 1e-6) / (pos + neg + 1e-6))

                # Potential loss (simplified direct mapping)
                potential_loss_i = Pots(b_slice_i, source_id_i).squeeze()

                # Weighted sum (simplified direct mapping)
                bary_loss += Lambda[source_id_i] * (mse_loss_i + 0.05 * (ort_loss_i + contra_loss_i) - potential_loss_i)

                mse_loss += mse_loss_i
                ort_loss += ort_loss_i
                contra_loss += contra_loss_i

        bary_loss = bary_loss / out_restored.shape[0]
        mse_loss = mse_loss / out_restored.shape[0]

        # 3. Classification loss
        cls_loss = F.cross_entropy(deg_logits, de_id)

        # 4-5. MoCE routing losses (directly from model.py)
        importance_loss = 0.5 * moe_loss_imp
        load_loss = 0.5 * moe_loss_load

        # Total Model loss
        Model_train_loss = (bary_loss + opt.Sigma * l1_loss + 
                           opt.lambda_cls * cls_loss +
                           opt.lambda_importance * importance_loss +
                           opt.lambda_load * load_loss)

        epoch_model_loss += Model_train_loss.item()
        
        Model_train_loss.backward()
        Model_optimizer.step()

        # ========== Potential optimization ==========
        unfreeze(Pots)
        freeze(Model)
        
        Pots_optimizer.zero_grad()
        potential_train_loss_acc = 0.0
        
        with Pots.no_sync():
            if iteration % 1 == 0:
                with torch.no_grad():
                    _, _, b_detached, _, _, _, _, _, _ = Model(degraded)
                
                batch_size = out_restored.shape[0]
                for i in range(batch_size):
                    source_id_i = de_id[i]
                    b_slice_i = b_detached[i, :]
                    
                    # Simplified direct mapping
                    potential_loss = Pots(b_slice_i, source_id_i).squeeze()
                    weighted_loss = Lambda[source_id_i] * potential_loss
                    
                    potential_train_loss_acc += weighted_loss.item()
                    
                    loss_i = weighted_loss / batch_size
                    loss_i.backward()

            # Constraint
            potential_constraint = 0
            for j in range(K):
                potential_constraint += Lambda[j] * Pots(b_slice_i, j).squeeze()

            potential_constraint_loss = 10 * (potential_constraint ** 2)
            potential_constraint_loss.backward()
        
        # Manual gradient synchronization
        for param in Pots.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad /= dist.get_world_size()

        Pots_optimizer.step()
        
        epoch_pot_loss += potential_train_loss_acc
        count += 1

        if iteration % 10 == 0 and is_main_process():
            print("Epoch {}({}/{}): Loss_Model: {:.5f}, Loss_Pots: {:.5f}, "
                  "L1: {:.5f}, Cls: {:.5f}, Imp: {:.5f}, Load: {:.5f}".format(
                      epoch, iteration, len(training_data_loader),
                      Model_train_loss.item(), 
                      potential_train_loss_acc / (batch_size if batch_size > 0 else 1),
                      l1_loss.item(), cls_loss.item(), 
                      importance_loss.item(), load_loss.item()))
            
            if not os.path.exists('./checksample/' + opt.type):
                os.makedirs('./checksample/' + opt.type)
                
            save_image(out_restored.data, './checksample/' + opt.type + '/output.png')
            save_image(degraded.data, './checksample/' + opt.type + '/degraded.png')
            save_image(target.data, './checksample/' + opt.type + '/target.png')

    return epoch_model_loss / count, epoch_pot_loss / count


def save_checkpoint(Model, Pots, epoch):
    model_out_path = "checkpoint/" + "model_BaryCE_" + str(opt.type) + "_" + str(opt.patch_size) + "_" + str(
        opt.nEpochs) + ".pth"
    
    state = {
        "epoch": epoch, 
        "Model": Model.module if hasattr(Model, "module") else Model, 
        "Pots": Pots.module if hasattr(Pots, "module") else Pots
    }
    
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
