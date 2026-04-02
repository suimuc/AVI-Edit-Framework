import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
from .wan_video_camera_controller import SimpleAdapter
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False, cross_attention_dim: int = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        if cross_attention_dim is None:
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
        else:
            self.k = nn.Linear(cross_attention_dim, dim)
            self.v = nn.Linear(cross_attention_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        batch = x.shape[0]
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)



class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual
    
def zero_module(module):
    for p in module.parameters():
        torch.nn.init.zeros_(p)

    return module

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x_mask: [B, T]
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x
    
    def forward(self, x, context, t_mod, freqs, latent_frame_mask = None, t0_mod = None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if (latent_frame_mask is not None):
            shift_msa_t0, scale_msa_t0, gate_msa_t0, shift_mlp_t0, scale_mlp_t0, gate_mlp_t0 = (
                self.modulation.to(dtype=t0_mod.dtype, device=t0_mod.device) + t0_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
            if (latent_frame_mask is not None):
                shift_msa_t0, scale_msa_t0, gate_msa_t0, shift_mlp_t0, scale_mlp_t0, gate_mlp_t0 = (
                    shift_msa_t0.squeeze(2), scale_msa_t0.squeeze(2), gate_msa_t0.squeeze(2),
                    shift_mlp_t0.squeeze(2), scale_mlp_t0.squeeze(2), gate_mlp_t0.squeeze(2),
                )
        input_x_norm = self.norm1(x)
        input_x = modulate(input_x_norm, shift_msa, scale_msa)
        if (latent_frame_mask is not None):
            input_x_t0 = modulate(input_x_norm, shift_msa_t0, scale_msa_t0)
            input_x = self.t_mask_select(latent_frame_mask, input_x, input_x_t0, latent_frame_mask.shape[1], input_x.shape[1]//latent_frame_mask.shape[1])
        attn_out = self.self_attn(input_x, freqs)
        attn_out_t = attn_out * gate_msa
        if (latent_frame_mask is not None):
            attn_out_t0 = attn_out * gate_msa_t0
            attn_out = self.t_mask_select(latent_frame_mask, attn_out_t, attn_out_t0, latent_frame_mask.shape[1], attn_out.shape[1]//latent_frame_mask.shape[1])
        else:
            attn_out = attn_out_t 
        x = x + attn_out
        x = x + self.cross_attn(self.norm3(x), context)
        input_x_norm = self.norm2(x)
        input_x = modulate(input_x_norm, shift_mlp, scale_mlp)
        if (latent_frame_mask is not None):
            input_x_t0 = modulate(input_x_norm, shift_mlp_t0, scale_mlp_t0)
            input_x = self.t_mask_select(latent_frame_mask, input_x, input_x_t0, latent_frame_mask.shape[1], input_x.shape[1]//latent_frame_mask.shape[1])
        ffn_out = self.ffn(input_x)
        ffn_out_t = ffn_out * gate_mlp
        if (latent_frame_mask is not None):
            ffn_out_t0 = ffn_out * gate_mlp_t0
            ffn_out = self.t_mask_select(latent_frame_mask, ffn_out_t, ffn_out_t0, latent_frame_mask.shape[1], ffn_out.shape[1]//latent_frame_mask.shape[1])
        else:
            ffn_out = ffn_out_t
        x = x + ffn_out
        return x
    

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x
    
class CausalConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode='replicate', **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

class MotionEncoder_tc(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, need_global=True, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1).to(device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)

        return x, x_local
    
class AdaLayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, elementwise_affine=False)

    def forward(self, x, temb):
        temb = self.linear(F.silu(temb))
        shift, scale = temb.chunk(2, dim=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = self.norm(x) * (1 + scale) + shift
        return x
    
class AudioInjector_WAN(nn.Module):

    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=[0, 27],
        enable_adain=False,
        adain_dim=2048,
    ):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):

                
                for inject_id in inject_layer:
                    
                    if f'transformer_blocks.{inject_id}' in mod_name and inject_id not in self.injected_block_id:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1
                        

        self.injector = nn.ModuleList([CrossAttention(
            dim=dim,
            num_heads=num_heads,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_feat = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_vec = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim) for _ in range(audio_injector_id)])

class CausalAudioEncoder(nn.Module):

    def __init__(self, dim=5120, num_layers=25, out_dim=2048, num_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length
        weights = self.act(self.weights.to(device=features.device, dtype=features.dtype))
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim
        return res  # b f n dim


    

    
class Mask_RefineNet_with_Audio_wo_patchfy(nn.Module):
    def __init__(self, dim: int, num_heads: int, mask_dim: int, ffn_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.mask_dim = mask_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size
        self.self_attn = SelfAttention(dim, num_heads, eps)

        self.maskattn = CrossAttention(dim, num_heads, eps)
        self.audio_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        
        self.norm4 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        
        
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation_t = nn.Parameter(torch.randn(1, 3, dim) / dim**0.5)
        self.gate = GateModule()
        
    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)
    
    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x_mask: [B, T]
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x
    
    def forward(self, mask, x, mask_type, mask_type_1, freqs, audio_embed, latent_frame_mask = None, mask_type_t = None, mask_type_t0 = None):
        has_seq = len(mask_type.shape) == 4
        chunk_dim = 2 if has_seq else 1

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=mask_type_t.dtype, device=mask_type_t.device) + mask_type_t).chunk(6, dim=chunk_dim)
        
        if (latent_frame_mask is not None):
            shift_msa_0, scale_msa_0, gate_msa_0, shift_mlp_0, scale_mlp_0, gate_mlp_0 = (
            self.modulation.to(dtype=mask_type_t0.dtype, device=mask_type_t0.device) + mask_type_t0).chunk(6, dim=chunk_dim)
            
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
            if (latent_frame_mask is not None):
                shift_msa_0, scale_msa_0, gate_msa_0, shift_mlp_0, scale_mlp_0, gate_mlp_0 = (
                    shift_msa_0.squeeze(2), scale_msa_0.squeeze(2), gate_msa_0.squeeze(2),
                    shift_mlp_0.squeeze(2), scale_mlp_0.squeeze(2), gate_mlp_0.squeeze(2),
                )
            

        input_mask_norm = self.norm1(mask)
        
        input_mask = modulate(input_mask_norm, shift_msa, scale_msa)
        
        if (latent_frame_mask is not None):
            input_mask_0 = modulate(input_mask_norm, shift_msa_0, scale_msa_0)
            input_mask = self.t_mask_select(latent_frame_mask, input_mask, input_mask_0, latent_frame_mask.shape[1],
                                     input_mask.shape[1] // latent_frame_mask.shape[1])

        attn_out = self.self_attn(input_mask, freqs)
        
        if (latent_frame_mask is not None):
            attn_out_0 = attn_out * gate_msa_0
            attn_out = attn_out * gate_msa
            attn_out = self.t_mask_select(latent_frame_mask, attn_out, attn_out_0, latent_frame_mask.shape[1],
                                        attn_out_0.shape[1] // latent_frame_mask.shape[1])
        else:
            attn_out = attn_out * gate_msa
            
        mask = mask + attn_out
        
        
        shift_msa1, scale_msa1, gate_msa1 = (
            self.modulation_t.to(dtype=mask_type_t[:, :3].dtype, device=mask_type_t[:, :3].device) + mask_type_t[:, :3]).chunk(3, dim=chunk_dim)
        if (latent_frame_mask is not None):
            shift_msa_t0, scale_msa_t0, gate_msa_t0 = (
                self.modulation_t.to(dtype=mask_type_t0[:, :3].dtype, device=mask_type_t0[:, :3].device) + mask_type_t0[:, :3]).chunk(3, dim=chunk_dim)
        if has_seq:
            shift_msa1, scale_msa1, gate_msa1 = (
                shift_msa1.squeeze(2), scale_msa1.squeeze(2), gate_msa1.squeeze(2),
            )
            if (latent_frame_mask is not None):
                shift_msa_t0, scale_msa_t0, gate_msa_t0 = (
                    shift_msa_t0.squeeze(2), scale_msa_t0.squeeze(2), gate_msa_t0.squeeze(2),
                )
                
        num_frames = audio_embed.shape[1]


        x = rearrange(x, "b (t n) c -> (b t) n c", t=num_frames)
        
        
        input_mask_norm1 = self.norm3(mask)

        input_mask1 = modulate(input_mask_norm1, shift_msa1, scale_msa1)
        if (latent_frame_mask is not None):
            input_mask_t0 = modulate(input_mask_norm1, shift_msa_t0, scale_msa_t0)
            input_mask1 = self.t_mask_select(latent_frame_mask, input_mask1, input_mask_t0, latent_frame_mask.shape[1],
                                     input_mask1.shape[1] // latent_frame_mask.shape[1])
        input_mask1 = rearrange(input_mask1, "b (t n) c -> (b t) n c", t=num_frames)
        attn_out1 = self.maskattn(input_mask1, x)
        residual = attn_out1 * gate_msa1
        residual = rearrange(residual, "(b t) n c -> b (t n) c", t=num_frames)
        if (latent_frame_mask is not None):
            attn_out1_t0 = attn_out1 * gate_msa_t0
            attn_out1_t0 = rearrange(attn_out1_t0, "(b t) n c -> b (t n) c", t=num_frames)
            residual = self.t_mask_select(latent_frame_mask, residual, attn_out1_t0, latent_frame_mask.shape[1],
                                        attn_out1_t0.shape[1] // latent_frame_mask.shape[1])
        
        
        
        

        mask = mask + residual
        
        input_mask_norm = self.norm2(mask)
        input_mask = modulate(input_mask_norm, shift_mlp, scale_mlp)
        
        if (latent_frame_mask is not None):
            input_mask_0 = modulate(input_mask_norm, shift_mlp_0, scale_mlp_0)
            input_mask = self.t_mask_select(latent_frame_mask, input_mask, input_mask_0, latent_frame_mask.shape[1],
                                     input_mask.shape[1] // latent_frame_mask.shape[1])

        ffn_out = self.ffn(input_mask)
        if (latent_frame_mask is not None):
            ffn_out_0 = ffn_out * gate_mlp_0
            ffn_out = ffn_out * gate_mlp
            ffn_out = self.t_mask_select(latent_frame_mask, ffn_out, ffn_out_0, latent_frame_mask.shape[1],
                                        ffn_out.shape[1] // latent_frame_mask.shape[1])
        else:
            ffn_out = ffn_out * gate_mlp
            
        mask = mask + ffn_out
        
        input_mask_norm1 = self.norm4(mask)
        input_mask_norm1 = rearrange(input_mask_norm1, "b (t n) c -> (b t) n c", t=num_frames)
        audio_embed = rearrange(audio_embed, "b t n c -> (b t) n c", t=num_frames)
        audio_residual = self.audio_attn(input_mask_norm1, audio_embed)
        audio_residual = rearrange(audio_residual, "(b t) n c-> b (t n) c", t=num_frames)
        
        mask = mask + audio_residual

        return mask
    
    

class MaskRefine_WAN(nn.Module):

    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=[0, 27],
        patch_size=[1,2,2],
        ffn_dim=14336,
        eps=1e-6,
        mask_dim=1,
    ):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name and inject_id not in self.injected_block_id:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1
        self.patch_size = patch_size

        self.injector = nn.ModuleList([Mask_RefineNet_with_Audio_wo_patchfy(
            dim,
            num_heads,
            1,
            ffn_dim,
            patch_size,
            eps,
        ) for _ in range(audio_injector_id)])
        
        self.head = Head(dim, mask_dim, patch_size, eps)
        self.patch_embedding = nn.Conv3d(mask_dim, dim, kernel_size=patch_size, stride=patch_size)
        
    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)
    
    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )




class WanModel_S2V(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        audio_dim: int,
        num_audio_token: int,
        has_image_input: bool,
        enable_adain: bool = True,
        audio_inject_layers: list = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        masktype_adain: bool = False,
        mask_refine_net_variant_with_audio_inject_multi_layers: bool = False,
        control_in_dim: int = 48,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.masktype_adain = masktype_adain
        self.mask_refine_net_variant_with_audio_inject_multi_layers = mask_refine_net_variant_with_audio_inject_multi_layers
        
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        
        
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        
        
        self.casual_audio_encoder = CausalAudioEncoder(dim=audio_dim, out_dim=dim, num_token=num_audio_token, need_global=enable_adain)
        

        
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        
        self.head = Head(dim, out_dim, patch_size, eps)

        
        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")


        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=dim,
            num_heads=num_heads,
            inject_layer=audio_inject_layers,
            enable_adain=enable_adain,
            adain_dim=dim,
        )
        
        
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)
        
        if (self.masktype_adain):
            self.masktype_embedding = nn.Sequential(
                zero_module(nn.Linear(1, dim)),
                nn.SiLU(),
                zero_module(nn.Linear(dim, dim))
            )
            self.masktype_embedding_projection = nn.Sequential(
                nn.SiLU(), zero_module(nn.Linear(dim, dim * 6)))

            
        if (self.mask_refine_net_variant_with_audio_inject_multi_layers):
            self.mask_refine_wan = MaskRefine_WAN(
                all_modules,
                all_modules_names,
                dim=dim,
                num_heads=num_heads,
                inject_layer=audio_inject_layers,
                patch_size=[1,2,2],
                ffn_dim=14336,
                eps=1e-6,
            )
            self.masktype_embedding_refine = nn.Sequential(
                zero_module(nn.Linear(1, dim)),
                nn.SiLU(),
                zero_module(nn.Linear(dim, dim))
            )
            self.masktype_embedding_projection_refine = nn.Sequential(
                nn.SiLU(), zero_module(nn.Linear(dim, dim * 6)))
            self.freqs_mask = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
    def after_transformer_block(self, block_idx, hidden_states, audio_emb_global, audio_emb, use_unified_sequence_parallel=False):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            num_frames = audio_emb.shape[1]
            if use_unified_sequence_parallel:
                from xfuser.core.distributed import get_sp_group
                hidden_states = get_sp_group().all_gather(hidden_states, dim=1)
            input_hidden_states = hidden_states.clone()  # b (f h w) c
            input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
            adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](input_hidden_states, temb=audio_emb_global[:, 0])
            attn_hidden_states = adain_hidden_states

            audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            residual_out = self.audio_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            hidden_states = hidden_states + residual_out
            if use_unified_sequence_parallel:
                from xfuser.core.distributed import get_sequence_parallel_world_size, get_sequence_parallel_rank
                hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            
        return hidden_states
    
    def after_transformer_block_mask_refine(self, block_idx, instance_mask, x, mask_type_adain_reine, mask_type_adain_reine_1, freqs_mask, merged_audio_emb, latent_frame_mask, mask_type_adain_reine_t, mask_type_adain_reine_t0, use_unified_sequence_parallel=False):
        if block_idx in self.mask_refine_wan.injected_block_id.keys():
            audio_attn_id = self.mask_refine_wan.injected_block_id[block_idx]
            instance_mask = self.mask_refine_wan.injector[audio_attn_id](instance_mask, x, mask_type_adain_reine, mask_type_adain_reine_1, freqs_mask, merged_audio_emb, latent_frame_mask, mask_type_adain_reine_t, mask_type_adain_reine_t0)

        return instance_mask

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelS2VStateDictConverter()
    
    
class WanModelS2VStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param

        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "966cffdcc52f9c46c391768b27637614":
            config = {
                
                
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "audio_dim": 1024,
                "num_audio_token": 4,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
                
            }
            
        elif hash_state_dict_keys(state_dict) == "ced44d91d37eef41ed7d1077ef50679d":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48 + 1,
                "dim": 3072,
                "audio_dim": 1024,
                "num_audio_token": 4,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
                "mask_refine_net_variant_with_audio_inject_multi_layers": True,
            }
            
        elif hash_state_dict_keys(state_dict) == "9e2d020458d137a0866de1d618343080" or hash_state_dict_keys(state_dict) == "3dfd35b9b99039e8dc0cccea3f98176a":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48 + 1,
                "dim": 3072,
                "audio_dim": 1024,
                "num_audio_token": 4,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": True,
                "masktype_adain": True,
                "mask_refine_net_variant_with_audio_inject_multi_layers": True,
            }
        else:
            
            config = {}
        return state_dict, config
