import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Learnable embedding for lead time (month index).
    """
    def __init__(self, max_t=6, emb_dim=16):
        super().__init__()
        self.embed = nn.Embedding(max_t, emb_dim)

    def forward(self, t):
        # t: [B*t]
        return self.embed(t)


class DoubleConv(nn.Module):
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
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ----------- New Modules -----------

class IndexEncoder(nn.Module):
    def __init__(self, index_dim, time_dim, hidden_dim, feature_dims):
        super(IndexEncoder, self).__init__()
        self.feature_dims = feature_dims
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(index_dim, hidden_dim, batch_first=True)
        self.gammas = nn.ModuleList([nn.Linear(hidden_dim, c) for c in feature_dims])
        self.betas  = nn.ModuleList([nn.Linear(hidden_dim, c) for c in feature_dims])

    def forward(self, indices):
        """
        indices: [B*t, K]
        time_emb: [B*t, D]
        """
        # h = torch.cat([indices, time_emb], dim=1)
        _, (h_n, _) = self.rnn(indices.permute(0, 2, 1))  # h_n: [1, 16, hidden_dim]   indices:[16,10,6]
        h = h_n.squeeze(0)  # [16, hidden_dim]
        h = self.mlp(h)
        # h = h.unsqueeze(1).repeat(1, 6, 1).view(-1, self.mlp[0].in_features)  # 假设 mlp 输入是 hidden_dim

        gammas, betas = [], []
        for g, b in zip(self.gammas, self.betas):
            gammas.append(g(h))
            betas.append(b(h))
        return gammas, betas



def apply_film(x, gamma, beta):
    """
    x: [B,C,H,W]
    gamma, beta: [B,C]
    """
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]/16,32,1,1
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    return x * (1 + gamma) + beta


class UNetLiteFiLM(nn.Module):
    """
    U-Net with FiLM conditioning from global indices.
    """
    def __init__(self, n_channels=10, n_classes=1, index_dim=3, max_t=6,
                 base_filters=32, dropout=0.2, bilinear=True, index_hidden=64):
        super(UNetLiteFiLM, self).__init__()
        self.time_embed = TimeEmbedding(max_t=max_t, emb_dim=16)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, base_filters, dropout=dropout)
        self.down1 = Down(base_filters, base_filters*2, dropout=dropout)
        self.down2 = Down(base_filters*2, base_filters*4, dropout=dropout)
        self.down3 = Down(base_filters*4, base_filters*8, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters*8, base_filters*16 // factor, dropout=dropout)

        # Decoder
        self.up1 = Up(base_filters*16, base_filters*8 // factor, dropout=dropout, bilinear=bilinear)
        self.up2 = Up(base_filters*8, base_filters*4 // factor, dropout=dropout, bilinear=bilinear)
        self.up3 = Up(base_filters*4, base_filters*2 // factor, dropout=dropout, bilinear=bilinear)
        self.up4 = Up(base_filters*2, base_filters, dropout=dropout, bilinear=bilinear)
        self.outc = OutConv(base_filters, n_classes)

        # Index encoder
        feature_dims = [base_filters, base_filters*2, base_filters*4,
                        base_filters*8, base_filters*16 // factor]
        self.index_encoder = IndexEncoder(index_dim, 16, index_hidden, feature_dims)

    def forward(self, x, indices):
        """
        x      : [B*t, C, H, W]
        indices: [B*t, K]
        lead   : [B*t]  (0~5)
        """
        # time_emb = self.time_embed(lead)
        gammas, betas = self.index_encoder(indices)

        x1 = self.inc(x)
        x1 = apply_film(x1, gammas[0], betas[0])

        x2 = self.down1(x1)
        x2 = apply_film(x2, gammas[1], betas[1])

        x3 = self.down2(x2)
        x3 = apply_film(x3, gammas[2], betas[2])

        x4 = self.down3(x3)
        x4 = apply_film(x4, gammas[3], betas[3])

        x5 = self.down4(x4)
        x5 = apply_film(x5, gammas[4], betas[4])

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)



if __name__ == "__main__":
    # sanity check
    model = UNetLiteFiLM(n_channels=10, n_classes=1, index_dim=3)
    x = torch.randn(2, 10, 128, 128)
    indices = torch.randn(2, 3)  # 3 global indices (e.g., 3 EOF PCs)
    out = model(x, indices)
    print("Output shape:", out.shape)  # expect [2,1,128,128]
