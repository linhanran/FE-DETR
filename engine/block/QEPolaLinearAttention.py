import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

class SpatialFilterEnhancer(nn.Module):
    def __init__(self, channels, mode='learnable', kernel_size=3, learnable=True, use_bn=True, activation='relu'):
        super().__init__()
        assert kernel_size in [3, 5], "Only 3x3 or 5x5 kernels are supported."

        self.channels = channels
        self.learnable = learnable
        self.mode = mode
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.weight = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,
            bias=False
        )

        kernel = self._get_initial_kernel(mode, kernel_size)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.weight.weight.data.copy_(kernel)
        self.weight.weight.requires_grad = learnable

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(channels)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    def _get_initial_kernel(self, mode, k):
        if mode == 'edge':
            return torch.tensor([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]], dtype=torch.float32) if k == 3 else torch.ones((5, 5)) * -1 + torch.eye(5)

        elif mode == 'laplacian':
            return torch.tensor([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]], dtype=torch.float32)

        elif mode == 'sharpen':
            return torch.tensor([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]], dtype=torch.float32)

        elif mode == 'identity':
            kernel = torch.zeros((k, k), dtype=torch.float32)
            kernel[k // 2, k // 2] = 1.0
            return kernel

        elif mode == 'learnable':
            return torch.randn((k, k), dtype=torch.float32) * 0.1

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        if H is None or W is None:
            H = W = int(N ** 0.5)
            assert H * W == N, "Cannot infer H and W from N. Please provide explicitly."

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.weight(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x

class QEPolaLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.sr_ratio = sr_ratio
        self.alpha = alpha

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim,
                             kernel_size=kernel_size, groups=head_dim,
                             padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, num_heads, 1, head_dim)))
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.kernel_function = nn.ReLU()

        self.pos_encoding_cache = {}

        # === Signal Processing Enhancement ===
        self.filter_enhancer_q = SpatialFilterEnhancer(dim, mode='learnable', learnable=True)
        self.enhance_weight = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Expected square input, got N={N}"

        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        # === Signal Filtering on q ===
        q_enhance = self.filter_enhancer_q(q, H, W)
        q = q + self.enhance_weight * q_enhance

        # Downsample k,v
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)

        k, v = kv[0], kv[1]
        n = k.shape[1]

        if (n, C) not in self.pos_encoding_cache:
            self.pos_encoding_cache[(n, C)] = nn.Parameter(torch.zeros(1, n, C).to(x.device))
        k = k + self.pos_encoding_cache[(n, C)]

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * torch.sigmoid(self.power)

        q = q / scale
        k = k / scale

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)

        q_pos = self.kernel_function(q) ** power
        q_neg = self.kernel_function(-q) ** power
        k_pos = self.kernel_function(k) ** power
        k_neg = self.kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z

        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n),
                                          size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        spatial_tokens = v.shape[2]
        H = W = int(spatial_tokens ** 0.5)
        assert H * W == spatial_tokens, f"Cannot reshape {spatial_tokens} to square"

        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x