import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')   
from calflops import calculate_flops

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
from engine.block.conv import Conv, autopad
import torch.fft
class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FGPC(nn.Module):   

    def __init__(self, c1, c2, k, s):
        super().__init__()

        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]     
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]

        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

        self.freq_proj = nn.Conv2d(2 * c1, 2 * c2, kernel_size=1, bias=False)
        self.freq_bn = nn.BatchNorm2d(2 * c2)
        self.freq_relu = nn.ReLU(inplace=True)

        self.attn_conv = nn.Conv2d(c2, c2, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))    
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))  
        spatial_feat = self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))  # [B, c2, H, W]

        x_f = torch.fft.fft2(x, norm='ortho')
        x_f_real = x_f.real
        x_f_imag = x_f.imag
        x_f_combined = torch.cat([x_f_real, x_f_imag], dim=1)

        freq_feat = self.freq_proj(x_f_combined)
        freq_feat = self.freq_bn(freq_feat)
        freq_feat = self.freq_relu(freq_feat)

        c = freq_feat.shape[1] // 2  
        real_part = freq_feat[:, :c, :, :]
        imag_part = freq_feat[:, c:, :, :]
        freq_complex = torch.complex(real_part, imag_part)

        spatial_from_freq = torch.fft.ifft2(freq_complex, norm='ortho').real  # [B, c2, H, W]

        attn_map = self.attn_conv(spatial_from_freq)  # [B, c2, H, W]
        attn_map = self.sigmoid(attn_map)  

        out = spatial_feat * attn_map

        return out

