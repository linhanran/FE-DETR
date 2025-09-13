import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .hybrid_encoder import HybridEncoder
from engine.block.QEPolaLinearAttention import QEPolaLinearAttention

__all__ = ['HybridEncoder_QEPOLA']


@register()
class HybridEncoder_QEPOLA(HybridEncoder):
    def __init__(self,
                 in_channels,
                 feat_strides,
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0,
                 enc_act='gelu',
                 use_encoder_idx=[2, 3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1,
                 depth_mult=1,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine'):

        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout,
                         enc_act, use_encoder_idx, num_encoder_layers, pe_temperature, expansion,
                         depth_mult, act, eval_spatial_size, version)

        self.encoder = nn.ModuleList([
            QEPolaLinearAttention(hidden_dim, num_heads=nhead)
            for _ in range(len(use_encoder_idx))
        ])

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

                memory = self.encoder[i](src_flatten)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # FPN top-down
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # PAN bottom-up
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]

            downsample_feat = self.downsample_convs[idx](feat_low)

            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            outs.append(out)
        return outs
