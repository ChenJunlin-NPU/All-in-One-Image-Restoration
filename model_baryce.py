import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from typing import Optional, List, Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal


##########################################################################
## 基础模块
##########################################################################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, 
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, 
                                     padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
                                   nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
                                   nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## 退化分类器
##########################################################################

class DegradationClassifier(nn.Module):
    """
    优化的退化类型分类器 - 简化结构 + 更强的特征提取
    输入: degradation_feat [B, 384, H/8, W/8] - 退化特征 (source_latent - bary_latent)
    输出: degradation_logits [B, num_classes], degradation_probs [B, num_classes]
    
    设计理念（v2 - 简化但更有效）: 
    1. 减少层数，避免过拟合
    2. 保留空间信息的同时增强判别性
    3. 使用更强的正则化（Dropout 0.5）
    4. 添加残差连接增强梯度流
    
    分类类别:
    0: denoise (包含denoise_15, denoise_25, denoise_50)
    1: derain
    2: dehaze
    3: deblur
    4: lowlight
    """
    def __init__(self, dim=384, num_classes=5):
        super(DegradationClassifier, self).__init__()
        
        # 特征提取 - 简化但更有效
        self.conv1 = nn.Conv2d(dim, 256, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 残差连接的通道调整
        self.shortcut = nn.Sequential(
            nn.Conv2d(dim, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # 通道注意力 (Squeeze-and-Excitation)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局特征
        self.local_pool = nn.AdaptiveAvgPool2d(4)   # 局部特征 4x4
        
        # 分类头 - 简化但更鲁棒
        # 全局特征128 + 局部特征128*16 = 2176
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128*16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 更强的dropout
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, dim, H, W] - degradation_feat (退化特征)
        Returns:
            logits: [B, num_classes]
            probs: [B, num_classes]
        """
        # x: [B, 384, H/8, W/8]
        identity = x
        
        # 特征提取 with residual
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # [B, 256, H/8, W/8]
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)  # [B, 256, H/8, W/8]
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)  # [B, 128, H/8, W/8]
        
        # 残差连接
        x = x + self.shortcut(identity)  # [B, 128, H/8, W/8]
        
        # 通道注意力
        se_weight = self.se_pool(x).view(x.size(0), -1)  # [B, 128]
        se_weight = self.se_fc(se_weight).view(x.size(0), x.size(1), 1, 1)  # [B, 128, 1, 1]
        x = x * se_weight  # 通道加权
        
        # 多尺度特征提取
        global_feat = self.global_pool(x).view(x.size(0), -1)  # [B, 128]
        local_feat = self.local_pool(x).view(x.size(0), -1)    # [B, 128*16]
        
        # 特征融合
        feat = torch.cat([global_feat, local_feat], dim=1)  # [B, 2176]
        
        # 分类
        logits = self.classifier(feat)  # [B, num_classes]
        probs = F.softmax(logits, dim=1)  # [B, num_classes]
        
        return logits, probs


##########################################################################
## MoCE 相关模块
##########################################################################

class FFTAttention(nn.Module):
    """FFT注意力机制"""
    def __init__(self, dim: int, **kwargs):
        super(FFTAttention, self).__init__()
        
        self.patch_size = kwargs.get("patch_size", 8)
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2)
        self.norm = LayerNorm(dim, "WithBias")
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)        
        
    def pad_and_rearrange(self, x):
        b, c, h, w = x.shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        x = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        return x
    
    def rearrange_to_original(self, x, x_shape):
        h, w = x_shape
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        x = x[:, :, :h, :w]
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
            
        q = self.pad_and_rearrange(q)
        k = self.pad_and_rearrange(k)
            
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        
        out = self.rearrange_to_original(out, (h, w))
        out = self.norm(out)
        out = out * v
        out = self.proj_out(out)
        return out


class MySequential(nn.Sequential):
    def forward(self, x1, x2):
        for layer in self:
            if isinstance(layer, ModExpert):
                x1 = layer(x1, x2)
            else:
                x1 = layer(x1)
        return x1


class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size: int):
        super(ModExpert, self).__init__()
        
        self.depth = depth
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)
            
    def process(self, x, shared):
        shortcut = x
        x = self.proj[0](x)
        x = self.body(x) * F.silu(self.proj[1](shared))
        x = self.proj[2](x)
        return x + shortcut

    def feat_extract(self, feats, shared):
        for _ in range(self.depth):
            feats = self.process(feats, shared)
        return feats
    
    def forward(self, x, shared):
        if x.shape[0] == 0:
            return x
        return self.feat_extract(x, shared)


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), 
                           expert_out[-1].size(3), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


class RoutingFunction(nn.Module):
    """
    修改后的路由函数
    输入: (bary_latent, degradation_probs)
    输出: gates, top_k_indices, top_k_values, aux_loss
    """
    def __init__(self, dim, num_classes, num_experts, k, complexity, use_complexity_bias=True, complexity_scale="max"):
        super(RoutingFunction, self).__init__()
        
        # Bary_latent gate
        self.bary_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )
        
        # Degradation probability gate
        self.degradation_gate = nn.Linear(num_classes, num_experts, bias=False)
        
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)
        
        self.k = k
        self.tau = 1
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def forward(self, bary_latent, degradation_probs):
        """
        Args:
            bary_latent: [B, dim, H, W]
            degradation_probs: [B, num_classes]
        Returns:
            gates: [B, num_experts]
            top_k_indices: [B, k]
            top_k_values: [B, k]
            aux_loss: scalar
        """
        # Compute logits from both sources
        bary_logits = self.bary_gate(bary_latent)  # [B, num_experts]
        deg_logits = self.degradation_gate(degradation_probs)  # [B, num_experts]
        
        # Fuse logits
        logits = bary_logits + deg_logits  # [B, num_experts]
        
        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))
        
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = torch.tensor(0.0, device=bary_latent.device)
        
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        importance = importance * (self.complexity * self.tau) if self.use_complexity_bias else importance
        imp_mean = importance.mean()
        imp_std = importance.std()
        return (imp_std / (imp_mean + 1e-8)) ** 2

    def load_loss(self, logits, logits_noisy, noise_std):
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        threshold_per_item = torch.sum(F.one_hot(thresholds, self.num_experts) * logits_noisy, dim=-1)
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)
        p_mean = p.mean(dim=0)
        return (p_mean.std() / (p_mean.mean() + 1e-8)) ** 2


class AdapterLayer(nn.Module):
    """
    修改后的MoCE模块
    输入: (degradation_feat, bary_latent, degradation_probs, source_latent)
    """
    def __init__(self, dim, rank, num_experts=4, top_k=2, expert_layer=FFTAttention, stage_depth=1,
                 depth_type="constant", rank_type="constant", num_classes=5, 
                 with_complexity=False, complexity_scale="max"):
        super().__init__()            
        
        self.loss = None
        self.top_k = top_k
        self.num_experts = num_experts

        patch_sizes = [2**(i+2) for i in range(num_experts)]
        kernel_sizes = [3+(2*i) for i in range(num_experts)]
        
        if depth_type == "constant":
            depths = [stage_depth for _ in range(num_experts)]
        else:
            depths = [stage_depth for _ in range(num_experts)]
        
        ranks = [rank for _ in range(num_experts)]
        
        self.experts = nn.ModuleList([
            MySequential(*[ModExpert(dim, rank=r, func=expert_layer, depth=d, patch_size=p, kernel_size=k)])
            for d, r, p, k in zip(depths, ranks, patch_sizes, kernel_sizes)
        ])
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(dim, num_classes, num_experts=num_experts, k=top_k,
                                       complexity=expert_complexity, use_complexity_bias=with_complexity, 
                                       complexity_scale=complexity_scale)
        
    def forward(self, degradation_feat, bary_latent, degradation_probs, source_latent):
        """
        Args:
            degradation_feat: [B, 384, H/8, W/8] - 退化特征 (专家处理的输入)
            bary_latent: [B, 384, H/8, W/8] - barycenter特征 (路由输入1)
            degradation_probs: [B, num_classes] - 分类器输出概率 (路由输入2)
            source_latent: [B, 384, H/8, W/8] - 原始特征 (shared输入)
        """
        gates, top_k_indices, top_k_values, aux_loss = self.routing(bary_latent, degradation_probs)
        self.loss = aux_loss
                
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(degradation_feat)
            expert_shared_inputs = dispatcher.dispatch(source_latent)
            expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_inputs[exp]) 
                            for exp in range(len(self.experts))]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            top_k_indices_list = top_k_indices[0].tolist()
            selected_experts = [self.experts[int(i)] for i in top_k_indices_list]
            expert_outputs = torch.stack([expert(degradation_feat, source_latent) for expert in selected_experts], dim=1)
            gates = gates.gather(1, top_k_indices)  
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs 
            out = weighted_outputs.sum(dim=1)
            
        return self.proj_out(out)



##########################################################################
## BaryCE 主模型 - 与BaryNet通道设计完全一致
##########################################################################

class BaryCE(nn.Module):
    """
    BaryCE-IR: Barycenter-guided Complexity Experts for Image Restoration
    
    新设计（Degradation-Centric）：
    - degradation_extractor: 直接提取退化特征
    - bary_latent = source_latent - degradation_feat (通过相减得到统一内容特征)
    
    通道设计与原始BaryNet完全一致：
    - dim=48
    - Level 1: 48 channels
    - Level 2: 96 channels  
    - Level 3: 192 channels
    - Bottleneck: 384 channels
    
    MoCE模块插入在Barycenter之后，处理退化特征
    """
    
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_enc_blocks=[4, 6, 6, 8],
                 num_bary_blocks=8,
                 num_moce_shared_blocks=1,
                 num_tail_blocks=4,
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 num_experts=4,
                 expert_rank=2,
                 top_k=2,
                 num_classes=5,
                 with_complexity=True,
                 complexity_scale="max"):
        
        super(BaryCE, self).__init__()
        
        self.dim = dim
        self.decoder = True  # 与BaryNet一致
        
        ##########################################################################
        ## 1. 输入投影
        ##########################################################################
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)
        
        ##########################################################################
        ## 2. Encoder - 与BaryNet完全一致
        ##########################################################################
        
        # Level 1: dim=48
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_enc_blocks[0])
        ])
        self.down1_2 = Downsample(dim)  # 48 -> 96
        
        # Level 2: dim*2=96
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_enc_blocks[1])
        ])
        self.down2_3 = Downsample(int(dim * 2))  # 96 -> 192
        
        # Level 3: dim*4=192
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_enc_blocks[2])
        ])
        self.down3_4 = Downsample(int(dim * 4))  # 192 -> 384
        
        # Bottleneck: dim*8=384
        self.latent = nn.Sequential(*[
            TransformerBlock(int(dim * 8), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_enc_blocks[3])
        ])
        
        ##########################################################################
        ## 3. Barycenter Map - 提取退化特征
        ##########################################################################
        # 新设计：直接提取退化特征（degradation_feat）
        # 然后通过相减得到统一内容特征（bary_latent）
        self.degradation_extractor = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_bary_blocks)])
        self.LN = LayerNorm(int(dim * 2 ** 3), LayerNorm_type)
        
        ##########################################################################
        ## 4. 退化分类器
        ##########################################################################
        self.degradation_classifier = DegradationClassifier(dim=int(dim * 8), num_classes=num_classes)
        
        ##########################################################################
        ## 5. MoCE 模块 - 处理退化特征
        ##########################################################################
        self.moce = AdapterLayer(
            dim=int(dim * 8),
            rank=expert_rank,
            num_experts=num_experts,
            top_k=top_k,
            expert_layer=FFTAttention,
            stage_depth=num_moce_shared_blocks,
            num_classes=num_classes,
            with_complexity=with_complexity,
            complexity_scale=complexity_scale
        )
        
        ##########################################################################
        ## 6. 解码器 - 直接从BaryNet复制，确保完全一致
        ##########################################################################
        
        # Level 4->3 清洁路径
        self.up4_3 = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        # + 512 → 192*3 = 576, 所以是 192 + 576 = 768
        self.noise_level3 = TransformerBlock(dim=int(dim * 2 ** 2) + 192 * 3, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 192 * 3, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[2])])

        # Level 4->3 残差路径
        self.resup4_3 = Upsample(int(dim * 2 ** 2)*2)
        self.resreduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) * 2, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.resreduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 192, int(dim * 2 ** 2) *2 , kernel_size=1, bias=bias)
        self.resdecoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[2])])

        # Level 3->2 清洁路径
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # + 224 → *2
        self.noise_level2 = TransformerBlock(dim=int(dim * 2 ** 1) * 2, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) * 2, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[1])])

        # Level 3->2 残差路径
        self.resup3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.resreduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.resreduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) * 2, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.resdecoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[1])])

        # Level 2->1 清洁路径
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        # + 64 → none
        self.noise_level1 = TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_enc_blocks[0])])

        # Level 2->1 残差路径
        self.resup2_1 = Upsample(int(dim * 2 ** 1))
        self.resreduce_noise_level1 = nn.Conv2d(int(dim*2**1) ,int(dim*2**1),kernel_size=1,bias=bias)

        # Refinement and Output
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2), int(dim * 2), kernel_size=1, bias=bias)
        self.resreduce_noise_level1 = nn.Conv2d(int(dim * 2), int(dim * 2), kernel_size=1, bias=bias)
        
        self.up2_1 = Upsample(int(dim * 2))  # 96 -> 48
        self.resup2_1 = Upsample(int(dim * 2))  # 96 -> 48
        
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_enc_blocks[0])
        ])
        
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)
        ])
        
        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.moce_loss = torch.tensor(0.0)

    def forward(self, inp_img):
        """
        前向传播 - 新设计：直接提取退化特征
        """
        
        ##########################################################################
        ## Encoder - 与BaryNet完全一致
        ##########################################################################
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        
        ##########################################################################
        ## Barycenter Map - 新设计：直接提取退化特征
        ##########################################################################
        source_latent = self.latent(inp_enc_level4)
        source_latent = self.LN(source_latent)

        # 新设计：直接提取退化特征
        degradation_feat = self.degradation_extractor(source_latent)
        degradation_feat = self.LN(degradation_feat)
        
        # 通过相减得到统一内容特征
        bary_latent = source_latent - degradation_feat
        res_bary_task = degradation_feat  # 保持变量名一致，用于后续模块
        
        ##########################################################################
        ## 退化分类器 - 基于退化特征预测退化类别
        ##########################################################################
        degradation_logits, degradation_probs = self.degradation_classifier(degradation_feat)
        
        ##########################################################################
        ## MoCE模块 - 处理退化特征
        ##########################################################################
        moce_output = self.moce(degradation_feat, bary_latent, degradation_probs, source_latent)
        self.moce_loss = self.moce.loss if hasattr(self.moce, 'loss') else torch.tensor(0.0, device=inp_img.device)
        
        ##########################################################################
        ## Decoder - 与BaryNet完全一致的数据流
        ##########################################################################
        
        if self.decoder:
            # 清洁路径：cat([bary_latent, moce_output])
            latent = self.noise_level3(torch.cat([bary_latent, moce_output], 1))
            latent = self.reduce_noise_level3(latent)
            # 残差路径：使用原始退化特征
            res_latent = self.resreduce_noise_level3(res_bary_task)

        inp_dec_level3 = self.up4_3(latent)
        res_inp_dec_level3 = self.resup4_3(res_latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3 + 0.8*res_inp_dec_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        res_inp_dec_level3 = self.resreduce_chan_level3(res_inp_dec_level3)
        res_out_dec_level3 = self.resdecoder_level3(res_inp_dec_level3)

        if self.decoder:
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
            res_out_dec_level3 = self.resreduce_noise_level2(res_out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        res_inp_dec_level2 = self.resup3_2(res_out_dec_level3)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2 + 0.8*res_inp_dec_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        res_inp_dec_level2 = self.resreduce_chan_level2(res_inp_dec_level2)
        res_out_dec_level2 = self.resdecoder_level2(res_inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
            res_out_dec_level2 = self.resreduce_noise_level1(res_out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        res_inp_dec_level1 = self.resup2_1(res_out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1 + 0.8*res_inp_dec_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        ##########################################################################
        ## 返回
        ##########################################################################
        aux_data = {
            'source_latent': source_latent,
            'degradation_feat': degradation_feat,  # 通过degradation_extractor提取的退化特征
            'unified_latent': bary_latent,  # 通过相减得到的统一内容特征
            'moce_output': moce_output,
            'moce_loss': self.moce_loss if isinstance(self.moce_loss, torch.Tensor) else torch.tensor(0.0, device=inp_img.device),
            'degradation_logits': degradation_logits,
            'degradation_probs': degradation_probs
        }
        
        return out_dec_level1, aux_data


##########################################################################
## Potentials 模块
##########################################################################

class Potentials(nn.Module):
    def __init__(self, num_potentials, channels=384, size=128):
        super(Potentials, self).__init__()
        self.num_potentials = num_potentials
        self.channels = channels
        self.size = size
        
        self.potentials = nn.ParameterList([
            nn.Parameter(torch.randn(channels, max(1, size // 16), max(1, size // 16)) * 0.01)
            for _ in range(num_potentials)
        ])
        
        self.channel_adapters = nn.ModuleDict()

    def forward(self, x, potential_id):
        B, C, H, W = x.shape
        device = x.device
        
        pot = self.potentials[potential_id]
        pot_channels = pot.shape[0]
        
        if (H, W) != (pot.shape[1], pot.shape[2]):
            pot_resized = F.interpolate(pot.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        else:
            pot_resized = pot
        
        if C != pot_channels:
            adapter_key = f"a{pot_channels}to{C}"
            if adapter_key not in self.channel_adapters:
                self.channel_adapters[adapter_key] = nn.Conv2d(pot_channels, C, 1, bias=False).to(device)
            pot_resized = self.channel_adapters[adapter_key](pot_resized.unsqueeze(0)).squeeze(0)
        
        pot_resized = pot_resized.to(device)
        return torch.sum(x * pot_resized.unsqueeze(0)) / (B * C * H * W + 1e-8)


class BaryNet(BaryCE):
    """向后兼容"""
    pass


if __name__ == "__main__":
    model = BaryCE()
    x = torch.randn(2, 3, 128, 128)
    y, aux = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Aux keys: {list(aux.keys())}")
