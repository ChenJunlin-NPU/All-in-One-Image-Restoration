## BaryCE: Barycenter-based Image Restoration with Content-Degradation Disentanglement
## 基于BaryIR改进，增加退化提取器、退化分类器和MoCE模块

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torchvision.utils import save_image
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal
import time


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
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


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
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


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
##---------- DegExtractor (传统OT，改造自BaryIR的Barycenter Map) -----------------------
class DegExtractor(nn.Module):
    """
    退化提取器：复用BaryIR的Barycenter Map结构
    输入：f_LQ (原始特征，包含内容I和退化D)
    输出：f_deg (纯退化特征D)，同时返回b和z用于损失计算

    核心逻辑：
    - z = f_LQ (输入特征)
    - b = BarycenterNet(z) (WB特征，对应Q分布)
    - f_deg = z - b (残差特征，对应R分布，即退化D)
    """
    def __init__(self,
                 dim=384,
                 num_blocks=8,
                 heads=8,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(DegExtractor, self).__init__()

        self.barylatent = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.LN = LayerNorm(dim, LayerNorm_type)

    def forward(self, f_LQ):
        z = self.LN(f_LQ)
        b = self.barylatent(z)
        b = self.LN(b)
        f_deg = z - b
        return f_deg, b, z


##########################################################################
##---------- Degradation Classifier -----------------------
class DegradationClassifier(nn.Module):
    """
    退化分类器：对退化特征进行分类
    输入：f_deg [B, C, H, W]
    输出：logits [B, num_degradations]
    """
    def __init__(self, in_channels=384, num_degradations=5):
        super(DegradationClassifier, self).__init__()
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_channels // 4, num_degradations)
        )
    
    def forward(self, f_deg):
        """
        前向传播
        Args:
            f_deg: 退化特征 [B, C, H, W]
        Returns:
            logits: 分类logits [B, num_degradations]
        """
        x = self.gap(f_deg)  # [B, C, 1, 1]
        logits = self.classifier(x)  # [B, num_degradations]
        return logits


##########################################################################
##---------- MoCE (Mixture of Complexity Experts, aligned with moce_ir) --
class MySequential(nn.Sequential):
    def forward(self, x1, x2):
        for layer in self:
            x1 = layer(x1, x2)
        return x1


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
        zeros = torch.zeros(
            self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
            requires_grad=True, device=stitched.device
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


class FFTAttention(nn.Module):
    """Fourier-domain attention expert body (from moce_ir)."""
    def __init__(self, dim: int, **kwargs):
        super(FFTAttention, self).__init__()
        self.patch_size = kwargs["patch_size"]
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, stride=1, padding=7 // 2, groups=dim * 2)
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
        return x[:, :, :h, :w]

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


class ModExpert(nn.Module):
    """Low-rank adapter expert with shared conditioning (from moce_ir)."""
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size: int):
        super(ModExpert, self).__init__()
        self.depth = depth
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False),
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


def _build_moce_expert_configs(dim, num_experts, stage_depth, depth_type, rank_type, base_rank):
    patch_sizes = [2 ** (i + 2) for i in range(num_experts)]
    kernel_sizes = [3 + (2 * i) for i in range(num_experts)]

    if depth_type == "lin":
        depths = [stage_depth + i for i in range(num_experts)]
    elif depth_type == "double":
        depths = [stage_depth + (2 * i) for i in range(num_experts)]
    elif depth_type == "exp":
        depths = [2 ** i for i in range(num_experts)]
    elif depth_type == "constant":
        depths = [stage_depth for _ in range(num_experts)]
    else:
        raise NotImplementedError(f"depth_type={depth_type}")

    if rank_type == "constant":
        ranks = [base_rank for _ in range(num_experts)]
    elif rank_type == "lin":
        ranks = [base_rank + i for i in range(num_experts)]
    elif rank_type == "double":
        ranks = [base_rank + (2 * i) for i in range(num_experts)]
    elif rank_type == "spread":
        ranks = [max(dim // (2 ** i), 16) for i in range(num_experts)][::-1]
    else:
        raise NotImplementedError(f"rank_type={rank_type}")

    return depths, ranks, patch_sizes, kernel_sizes


class DegRoutingFunction(nn.Module):
    """
    Dual-path router: combines feature content and degradation guidance.
    - feat_gate: captures sample-specific complexity/difficulty from f_deg
    - guide_gate: captures degradation-type tendency from deg_logits
    Both signals are added in logit space before softmax (following moce_ir design).
    """
    def __init__(self, dim, deg_dim, num_experts, k, complexity,
                 use_complexity_bias=True, complexity_scale="max",
                 guide_temperature=1.0):
        super(DegRoutingFunction, self).__init__()
        
        # Feature gate: GAP + Linear on input feature (captures sample complexity)
        self.feat_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )
        
        # Degradation guidance gate: Linear on deg_logits (captures degradation tendency)
        self.guide_gate = nn.Linear(deg_dim, num_experts, bias=False)

        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)

        self.k = min(k, num_experts)
        self.tau = 1.0
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias
        self.guide_temperature = guide_temperature

    def _prepare_deg_guidance(self, deg_guidance):
        if deg_guidance.dim() == 2:
            deg_guidance = F.softmax(deg_guidance / self.guide_temperature, dim=1)
        elif deg_guidance.dim() == 4:
            deg_guidance = F.softmax(deg_guidance / self.guide_temperature, dim=1)
            deg_guidance = deg_guidance.mean(dim=(-2, -1))
        else:
            raise ValueError(f"deg_guidance must be [B, K] or [B, K, H, W], got {tuple(deg_guidance.shape)}")
        return deg_guidance

    def _build_topk_gates(self, scores):
        top_k_values, top_k_indices = torch.topk(scores, self.k, dim=-1)
        top_k_values = top_k_values / (top_k_values.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.zeros_like(scores).scatter_(1, top_k_indices, top_k_values), top_k_indices, top_k_values

    def forward(self, x, deg_guidance):
        deg_guidance = self._prepare_deg_guidance(deg_guidance)
        
        # Dual-path routing: feature content + degradation guidance
        feat_logits = self.feat_gate(x)           # [B, num_experts] from feature
        guide_logits = self.guide_gate(deg_guidance)  # [B, num_experts] from degradation
        logits = feat_logits + guide_logits       # Combined routing signal
        
        gating_scores = logits.softmax(dim=-1)

        loss_imp = torch.tensor(0.0, device=x.device)
        loss_load = torch.tensor(0.0, device=x.device)
        if self.training:
            loss_imp = self.importance_loss(gating_scores)
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            gating_scores_for_topk = noisy_logits.softmax(dim=-1)
        else:
            gating_scores_for_topk = gating_scores

        gates, _, _ = self._build_topk_gates(gating_scores_for_topk)

        importance = gating_scores.sum(dim=0)
        if self.use_complexity_bias:
            complexity_importance = importance * (self.complexity * self.tau)
        else:
            complexity_importance = importance

        return gates, gating_scores, complexity_importance, loss_imp, loss_load

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        if self.use_complexity_bias:
            importance = importance * (self.complexity * self.tau)
        imp_mean = importance.mean()
        imp_std = importance.std()
        return (imp_std / (imp_mean + 1e-8)) ** 2

    def load_loss(self, logits, logits_noisy, noise_std):
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy, dim=-1
        )
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        normal_dist = Normal(0, 1)
        p = 1.0 - normal_dist.cdf(noise_required_to_win)
        p_mean = p.mean(dim=0)
        return (p_mean.std() / (p_mean.mean() + 1e-8)) ** 2


class MoCEAdapterLayer(nn.Module):
    """Heterogeneous experts for refining f_deg into a complete degradation feature delta_f."""
    def __init__(self, dim, rank, num_experts=4, top_k=1, expert_layer=FFTAttention,
                 stage_depth=1, depth_type="constant", rank_type="spread", deg_dim=5,
                 with_complexity=True, complexity_scale="max"):
        super(MoCEAdapterLayer, self).__init__()
        self.top_k = min(top_k, num_experts)
        self.num_experts = num_experts

        depths, ranks, patch_sizes, kernel_sizes = _build_moce_expert_configs(
            dim, num_experts, stage_depth, depth_type, rank_type, rank
        )

        self.experts = nn.ModuleList([
            MySequential(*[
                ModExpert(dim, rank=r, func=expert_layer, depth=d, patch_size=p, kernel_size=k)
            ])
            for r, d, p, k in zip(ranks, depths, patch_sizes, kernel_sizes)
        ])

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)

        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = DegRoutingFunction(
            dim, deg_dim, num_experts=num_experts, k=self.top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity,
            complexity_scale=complexity_scale,
        )

    def forward(self, x, deg_guidance, shared):
        gates, gating_scores, complexity_importance, loss_imp, loss_load = self.routing(x, deg_guidance)

        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_inputs = dispatcher.dispatch(shared)
            expert_outputs = [
                self.experts[exp](expert_inputs[exp], expert_shared_inputs[exp])
                for exp in range(len(self.experts))
            ]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            out = self._combine_topk(x, shared, gates)

        delta_f = self.proj_out(out)
        return delta_f, gating_scores, complexity_importance, loss_imp, loss_load

    def _combine_topk(self, x, shared, gates):
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            w = gates[:, e]
            mask = w > 0
            if mask.any():
                w = w[mask].view(-1, 1, 1, 1)
                out[mask] = out[mask] + w * self.experts[e](x[mask], shared[mask])
        return out


class MoCE(nn.Module):
    """
    Bottleneck degradation-mechanism MoCE.
    - Experts use moce_ir's heterogeneous ModExpert + FFTAttention design.
    - Router combines feature content (GAP+Linear) and degradation guidance (Linear)
      in logit space, following moce_ir's dual-path routing pattern.
    - f_deg is dispatched to selected experts for refinement.
    - Output delta_f is the modeled degradation feature for f_final = f_LQ - delta_f.
    """
    def __init__(self,
                 dim=384,
                 num_degradations=5,
                 num_experts=4,
                 top_k=1,
                 rank=48,
                 stage_depth=1,
                 depth_type="constant",
                 rank_type="spread",
                 num_heads=8,
                 bias=False,
                 with_complexity=True,
                 complexity_scale="max",
                 expert_layer=FFTAttention):
        super(MoCE, self).__init__()
        self.dim = dim
        self.shared = Attention(dim, num_heads, bias)
        self.adapter = MoCEAdapterLayer(
            dim=dim,
            rank=rank,
            num_experts=num_experts,
            top_k=top_k,
            expert_layer=expert_layer,
            stage_depth=stage_depth,
            depth_type=depth_type,
            rank_type=rank_type,
            deg_dim=num_degradations,
            with_complexity=with_complexity,
            complexity_scale=complexity_scale,
        )

    def forward(self, f_deg, deg_guidance):
        """
        Args:
            f_deg: [B, dim, H, W]
            deg_guidance: [B, num_degradations] or [B, num_degradations, H, W]
        Returns:
            delta_f: [B, dim, H, W]
            routing_weights: [B, K] dense routing scores before top-k sparsification
            complexity_importance: [K]
            moe_loss_imp, moe_loss_load: routing auxiliary losses (train only)
        """
        shared = self.shared(f_deg)
        delta_f, routing_weights, complexity_importance, moe_loss_imp, moe_loss_load = self.adapter(
            f_deg, deg_guidance, shared
        )
        return delta_f, routing_weights, complexity_importance, moe_loss_imp, moe_loss_load


##########################################################################
##---------- Model_BaryCE (完整网络) -----------------------
class Model_BaryCE(nn.Module):
    """
    BaryCE完整网络架构
    数据流：
    1. Input (LQ) -> Encoder -> f_LQ
    2. f_LQ -> DegExtractor -> f_deg (+ b, z用于损失)
    3. f_deg -> DegradationClassifier -> deg_logits
    4. f_deg + deg_logits -> MoCE -> delta_f
    5. f_final = f_LQ - delta_f (显式去除退化)
    6. f_final -> Decoder -> HQ
    """
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 num_degradations=5):
        super(Model_BaryCE, self).__init__()
        
        self.num_degradations = num_degradations
        
        # ========== Encoder ==========
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, 
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        # ========== DegExtractor (传统OT) ==========
        self.deg_extractor = DegExtractor(
            dim=int(dim * 2 ** 3),
            num_blocks=num_blocks[3],
            heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        
        # ========== Degradation Classifier ==========
        self.deg_classifier = DegradationClassifier(
            in_channels=int(dim * 2 ** 3),
            num_degradations=num_degradations
        )
        
        # ========== MoCE (Mixture of Complexity Experts, moce_ir style) ==========
        self.moe = MoCE(
            dim=int(dim * 2 ** 3),
            num_degradations=num_degradations,
            num_experts=4,
            top_k=1,
            rank=48,
            stage_depth=1,
            depth_type="constant",
            rank_type="spread",
            num_heads=heads[3],
            bias=bias,
            with_complexity=True,
            complexity_scale="max",
        )
        
        # ========== Fusion Layer ==========
        self.fusion = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 3), kernel_size=1, bias=bias)
        
        # ========== Decoder (with skip connections) ==========
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 2) + int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 1) + int(dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    
    def forward(self, inp_img):
        """
        前向传播 (传统OT版本)
        Args:
            inp_img: 输入LQ图像 [B, 3, H, W]
        Returns:
            out_restored: 复原的HQ图像 [B, 3, H, W]
            f_deg: 退化特征 (用于损失计算)
            b: WB特征 (用于MWB损失)
            z: 原始特征 (用于IRC损失)
            deg_logits: 退化分类logits (用于分类损失)
            routing_weights: MoCE路由权重 (用于监控)
            complexity_importance: 复杂度感知的重要性 (用于监控)
            moe_loss_imp: MoCE importance loss
            moe_loss_load: MoCE load balance loss
        """
        # ========== Encoder ==========
        enc_level1 = self.patch_embed(inp_img)
        enc_level1 = self.encoder_level1(enc_level1)
        
        enc_level2 = self.down1_2(enc_level1)
        enc_level2 = self.encoder_level2(enc_level2)
        
        enc_level3 = self.down2_3(enc_level2)
        enc_level3 = self.encoder_level3(enc_level3)
        
        enc_level4 = self.down3_4(enc_level3)
        f_LQ = self.latent(enc_level4)  # 原始特征 (包含内容I和退化D)
        
        # ========== DegExtractor (传统OT) ==========
        f_deg, b, z = self.deg_extractor(f_LQ)
        
        # ========== Degradation Classifier ==========
        deg_logits = self.deg_classifier(f_deg)
        
        # ========== MoCE (Mixture of Complexity Experts) ==========
        delta_f, routing_weights, complexity_importance, moe_loss_imp, moe_loss_load = self.moe(
            f_deg, deg_logits
        )
        
        # ========== Fusion ==========
        f_final = self.fusion(f_LQ - delta_f)
        
        # ========== Decoder (with skip connections) ==========
        inp_dec_level3 = self.up4_3(f_final)
        inp_dec_level3 = torch.cat([inp_dec_level3, enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_restored = self.output(out_dec_level1) + inp_img
        
        return (out_restored, f_deg, b, z, deg_logits,
                routing_weights, complexity_importance, moe_loss_imp, moe_loss_load)


##########################################################################
##---------- Potentials (复用BaryIR的Potential网络) -----------------------
class Potentials(nn.Module):
    """
    复用BaryIR的Potential网络，用于计算MWB、IRC、BRO损失
    """
    def __init__(self, num_potentials, channels=384, size=128): 
        super(Potentials, self).__init__()
        self.num_potentials = num_potentials
        self.input_channels = channels
        
        input_spatial = size // 8 
        conv_out_size = ((input_spatial + 2 * 1 - 4) // 4) + 1  
        self.num_features = channels * conv_out_size * conv_out_size
        
        self.potentials = nn.ModuleList(
            [self._build_potentials(self.input_channels, self.num_features) for _ in range(num_potentials)])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def _build_potentials(self, input_channels, num_fea):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=4, stride=4, groups=4, padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(num_fea, int(num_fea / 4)),
            nn.Linear(int(num_fea / 4), 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, input, idx=None):
        input = input.unsqueeze(0) if input.dim() == 3 else input
        if idx is not None:
            return self.potentials[idx](input)
        else:
            return [potential(input) for potential in self.potentials]
