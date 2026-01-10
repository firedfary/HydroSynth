import torch
import os
import warmup_scheduler
import tqdm
import numpy as np
import fucs
import config
# from torch.utils.tensorboard import SummaryWriter
from unetlite import UNetLite
from unet_model import UNet
from fucs import save_image
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import HydroSynth.process.observe_norm as observe_norm
from unetlitefilm import UNetLiteFiLM
from Unet2 import prepare_data as prepare_data2
from Unet3 import prepare_data as prepare_data3
from torch.utils.data import TensorDataset, DataLoader
train_set, test_set = prepare_data3()

train_loader = DataLoader(
    train_set, batch_size=config.modelconfig["batch_size"],
    shuffle=False, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=config.modelconfig["batch_size"],
    shuffle=False, num_workers=0, pin_memory=True
)

input_channels = train_set[0][1].shape[0]  # condition channels
index_dim = train_set[0][3].shape[0]       # PCs dimension (n_pcs * window)

with torch.no_grad():
    device = torch.device(config.modelconfig['device'])
    model = UNetLiteFiLM(
        n_channels=input_channels,
        n_classes=1,
        index_dim=index_dim,
        base_filters=16,
        dropout=0.2
    ).to(device)
    ckpt = torch.load(
        os.path.join(config.modelconfig['save_weight_path'], "epoch_500.pt"),
        map_location=device
    )
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

output_list = []
x0_list = []
with torch.no_grad():
    for x_0, cond, mask, pcs in tqdm.tqdm(train_loader):
        x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
        output = model(cond, pcs)
        output[mask] = float('nan')
        output_list.append(output)
        x0_list.append(x_0)

with torch.no_grad():
    for x_0, cond, mask, pcs in tqdm.tqdm(test_loader):
        x_0, cond, mask, pcs = x_0.to(device), cond.to(device), mask.to(device), pcs.to(device)
        output = model(cond, pcs)
        output[mask] = float('nan')
        output_list.append(output)
        x0_list.append(x_0)

output_all = torch.cat(output_list, dim=0)
x0_all = torch.cat(x0_list, dim=0)
acc = fucs.cal_acc(torch.tensor(x0_all).squeeze(1), torch.tensor(output_all).squeeze(1))
print(acc[0:345].mean(),acc[345:366].mean())

from torchviz import make_dot
import graphviz
y = model(cond, pcs)
output = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
output.format = "png"
output.directory = "./"
output.render("torchviz", view=True)