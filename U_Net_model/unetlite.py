import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers each followed by BatchNorm and ReLU.
    Optionally apply Dropout after the second conv.
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_ch, out_ch, dropout=0.0, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final convolution to output single-channel prediction
    """
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetLite(nn.Module):
    """
    A simplified U-Net for precipitation prediction on 10-channel climate inputs.
    """
    def __init__(self, n_channels=10, n_classes=1, base_filters=32, dropout=0.2, bilinear=True):
        super(UNetLite, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling path)
        self.inc = DoubleConv(n_channels, base_filters, dropout=dropout)
        self.down1 = Down(base_filters, base_filters*2, dropout=dropout)
        self.down2 = Down(base_filters*2, base_filters*4, dropout=dropout)
        self.down3 = Down(base_filters*4, base_filters*8, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters*8, base_filters*16 // factor, dropout=dropout)

        # Decoder (Upsampling path)
        self.up1 = Up(base_filters*16, base_filters*8 // factor, dropout=dropout, bilinear=bilinear)
        self.up2 = Up(base_filters*8, base_filters*4 // factor, dropout=dropout, bilinear=bilinear)
        self.up3 = Up(base_filters*4, base_filters*2 // factor, dropout=dropout, bilinear=bilinear)
        self.up4 = Up(base_filters*2, base_filters, dropout=dropout, bilinear=bilinear)
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    # sanity check
    model = UNetLite(n_channels=10, n_classes=1)
    input_tensor = torch.randn(2, 10, 128, 128)
    output = model(input_tensor)
    print("Output shape:", output.shape)  # expect [2,1,128,128]
