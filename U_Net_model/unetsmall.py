import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_to_match(src, target):
    """
    裁剪 src 特征图，使其尺寸与 target 一致（中心裁剪）。
    用于跳跃连接时对齐尺寸，防止尺寸不一致导致的 RuntimeError。
    """
    _, _, h, w = target.shape
    return src[:, :, :h, :w]

class DoubleConv(nn.Module):
    """
    两次卷积 + BN + LeakyReLU + Dropout
    """
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    """
    轻量化 U-Net，适合中小规模气象数据。
    - 通道起步较小（默认 48）
    - BN + Dropout 正则化
    - LeakyReLU 改善梯度流
    - 自动裁剪跳跃连接，避免输入尺寸非 2^n 时出错
    """
    def __init__(self, n_channels=10, n_classes=1, base_features=48, dropout=0.1):
        super().__init__()
        bf = base_features
        self.inc = DoubleConv(n_channels, bf, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf*2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf*2, bf*4, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf*4, bf*8, dropout))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf*8, min(bf*16, 512), dropout))

        self.up1 = DoubleConv(bf*16, bf*8, dropout)
        self.up2 = DoubleConv(bf*8, bf*4, dropout)
        self.up3 = DoubleConv(bf*4, bf*2, dropout)
        self.up4 = DoubleConv(bf*2, bf, dropout)

        self.outc = nn.Conv2d(bf, n_classes, 1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器 + 跳跃连接（裁剪保证尺寸一致）
        x = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x4_c = crop_to_match(x4, x)
        x = torch.cat([x, x4_c], dim=1)
        x = self.up1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x3_c = crop_to_match(x3, x)
        x = torch.cat([x, x3_c], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x2_c = crop_to_match(x2, x)
        x = torch.cat([x, x2_c], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x1_c = crop_to_match(x1, x)
        x = torch.cat([x, x1_c], dim=1)
        x = self.up4(x)

        return self.outc(x)

if __name__ == "__main__":
    # 测试一下尺寸是否匹配
    model = UNetSmall(n_channels=10, n_classes=1, base_features=48, dropout=0.1)
    dummy_input = torch.randn(2, 10, 120, 140)  # 高度和宽度不是 2^n
    out = model(dummy_input)
    print("输出尺寸:", out.shape)  # 期望 (2, 1, 120, 140)
