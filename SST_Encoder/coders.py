import torch
import UnetModels
import os
import xarray
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, ch2, ch3, dropout, atten=False) -> None:
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(1, ch2, kernel_size=3, stride=1, padding=1)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(ch2),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(ch2, ch3, kernel_size=3, stride=1, padding=1)
        )
        
        self.shortcut = torch.nn.Identity()

        if atten:
            self.attention = UnetModels.AttenBlock(3)
        else:
            self.attention = torch.nn.Identity()
        
        self.pool1 = torch.nn.AdaptiveMaxPool2d((64, 64))

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attention(h)
        h = self.pool1(h)
        return h



class Decoder(torch.nn.Module):
    def __init__(self, origin_size, ch2, ch3, dropout, atten=False) -> None:
        super().__init__()
        self.block2 = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(ch3),
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(ch3, ch2, kernel_size=3, stride=1, padding=1)
        )
        self.block1 = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(ch2),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.ConvTranspose2d(ch2, 1, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = torch.nn.Identity()
        if atten:
            self.attention = UnetModels.AttenBlock(3)
        else:
            self.attention = torch.nn.Identity()
        
        self.pool1 = torch.nn.AdaptiveMaxPool2d(origin_size)

    def forward(self, x):
        h = self.pool1(x)
        h = self.block2(h)
        h = self.block1(h)
        # h = h + self.shortcut(x)
        # h = self.attention(h)
        return h


class AutoEncoder(torch.nn.Module):
    def __init__(self, origin_size, ch2, ch3, dropout, atten=False) -> None:
        super().__init__()
        self.encoder = Encoder(ch2, ch3, dropout, atten)
        self.decoder = Decoder(origin_size, ch2, ch3, dropout, atten)

    def forward(self, x):
        h = self.encoder(x)
        h = self.decoder(h)
        return h


if __name__ == "__main__":
    model = AutoEncoder(origin_size=(128,256), ch2=2, ch3=3, dropout=0.2, atten=True)
    model.eval()
    a = torch.rand(1,1,128,256)
    b = model(a)
    print(b.shape)
    print(b)


def save_model(model, optimizer, epoch, save_dir="./checkpoints", filename_prefix="model"):
    """
    保存模型的权重和优化器的状态。
    Args:
        model (torch.nn.Module): 训练的模型。
        optimizer (torch.optim.Optimizer): 训练中使用的优化器。
        epoch (int): 当前的训练轮数。
        save_dir (str): 保存模型的目录。
        filename_prefix (str): 文件名前缀。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存文件路径
    save_path = os.path.join(save_dir, f"{filename_prefix}_epoch{epoch}.pth")
    
    # 保存内容
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")



def read_condition_from_file(file, params) -> torch.Tensor:
    con_data = xarray.open_dataset(file)['sst']
    con_data = con_data.sel(lon=slice(params['lon_start'], params['lon_end']), 
                            lat=slice(params['lat_start'], params['lat_end']), 
                            time=slice(params['time_start'], params['time_end']))
    condition = torch.from_numpy(np.array(con_data)).unsqueeze(1)#276,1,20,61
    mask = ~torch.isnan(condition)
    condition = torch.nan_to_num(condition, nan=0)
    condition = condition / 35
    return condition, mask

    parameters = {
        'lon_start': 120,
        'lon_end': 180,
        'lat_start': -10,
        'lat_end': 10,
        'time_start': '1994-01-01',
        'time_end': '2016-12-01'
    }

    cd = read_condition_from_file("E:/D1/f01/data/sst.oisst.mon.mean.1982.nc", parameters)
    print(cd)


def cal_acc(observed:torch.Tensor, predicted:torch.Tensor) -> torch.Tensor:
    assert observed.shape == predicted.shape, "Observed and predicted tensors must have the same shape."
    assert observed.ndim < 4, "Observed tensor must smaller or eq 3 dimensions (time, height, width)."
    if observed.ndim == 2:
        observed = observed.unsqueeze(0)
        predicted = predicted.unsqueeze(0)
    observed_mean = torch.nanmean(observed, dim=(1, 2), keepdim=True)
    predicted_mean = torch.nanmean(predicted, dim=(1, 2), keepdim=True)
    A = observed - observed_mean
    B = predicted - predicted_mean
    C = torch.sqrt(torch.nansum(torch.mul(A, A), dim=(1, 2))) * torch.sqrt(torch.nansum(torch.mul(B, B), dim=(1, 2)))
    ACC = torch.nansum(torch.mul(A, B), dim=(1, 2)) / C
    return ACC
