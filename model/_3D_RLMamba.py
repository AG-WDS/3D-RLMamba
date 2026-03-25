import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class SpeMamba3D(nn.Module):
    def __init__(self, channels, token_num=4, use_residual=True, group_num=4):
        super().__init__()
        self.use_residual = use_residual
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.group_channel_num * token_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            x = torch.cat([x, torch.zeros((B, pad_c, D, H, W), device=x.device)], dim=1)

        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(B*D*H*W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, D, H, W, self.channel_num).permute(0, 4, 1, 2, 3).contiguous()
        
        if self.use_residual:
            return x + self.proj(x_recon)
        else:
            return self.proj(x_recon)

class SpaMamba3D(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4):
        super().__init__()
        self.use_residual = use_residual
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(1, -1, C)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        if self.use_residual:
            return x + self.proj(x_recon)
        else:
            return self.proj(x_recon)

class Branch3D(nn.Module):
    def __init__(self, in_ch, hidden_dim=32, token_num=4, group_num=4, use_residual=True, mode='spa'):
        super().__init__()
        self.use_residual = use_residual
        self.mode = mode

        self.patch_embedding = nn.Sequential(
            nn.Conv3d(in_ch, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        if mode == 'spa':
            self.mamba = SpaMamba3D(hidden_dim, use_residual, group_num)
        else:
            self.mamba = SpeMamba3D(hidden_dim, token_num, use_residual, group_num)

        if in_ch != hidden_dim:
            self.shortcut = nn.Conv3d(in_ch, hidden_dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.patch_embedding(x)
        x = self.mamba(x)
        
        if self.use_residual:
            return x + identity
        else:
            return x

class _3D_RLMamba(nn.Module):
    def __init__(self, num_classes=9, hidden_dim=32, group_num=4, token_num=4, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.branch1 = Branch3D(15, hidden_dim, token_num, group_num, use_residual, mode='spa')
        self.branch2 = Branch3D(4, hidden_dim, token_num, group_num, use_residual, mode='spe')
        self.branch3 = Branch3D(8, hidden_dim, token_num, group_num, use_residual, mode='spe')
        self.branch4 = Branch3D(3, hidden_dim, token_num, group_num, use_residual, mode='spe')

        self.shortcut = nn.Conv3d(15, 4 * hidden_dim, kernel_size=1)

        self.cls_head = nn.Sequential(
            nn.Conv3d(4 * hidden_dim, 128, kernel_size=1),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv3d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x[:, 0:15, :, :, :])
        out2 = self.branch2(x[:, 0:4, :, :, :])
        out3 = self.branch3(x[:, 4:12, :, :, :])
        out4 = self.branch4(x[:, 12:15, :, :, :])

        out = torch.cat([out1, out2, out3, out4], dim=1)
        identity = self.shortcut(x[:, 0:15, :, :, :])
        
        if self.use_residual:
            out = out + identity

        logits = self.cls_head(out)
        logits = logits.mean(dim=[2, 3, 4])
        return logits