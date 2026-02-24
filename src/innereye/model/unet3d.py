import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.theta = nn.Conv3d(x_ch, inter_ch, 2, stride=2, bias=False)
        self.phi = nn.Conv3d(g_ch, inter_ch, 1, bias=True)
        self.psi = nn.Conv3d(inter_ch, 1, 1, bias=True)

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:], mode="trilinear", align_corners=False)
        act = F.relu(theta_x + phi_g)
        psi = torch.sigmoid(self.psi(act))
        psi_up = F.interpolate(psi, size=x.shape[2:], mode="trilinear", align_corners=False)
        return x * psi_up


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.gate3 = AttentionGate(base * 8, base * 4, base * 4)
        self.gate2 = AttentionGate(base * 4, base * 2, base * 2)
        self.gate1 = AttentionGate(base * 2, base, base)
        self.up4 = nn.ConvTranspose3d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv3d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, self.gate3(e4, d4)], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.gate2(e3, d3)], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.gate1(e2, d2)], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


def dice_loss(pred, target, eps=1e-5):
    num_classes = pred.shape[1]
    pred_soft = torch.softmax(pred, dim=1)
    target_1hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = torch.sum(pred_soft * target_1hot, dims)
    cardinality = torch.sum(pred_soft + target_1hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()
