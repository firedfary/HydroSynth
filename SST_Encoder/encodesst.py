import torch
import coders
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = coders.AutoEncoder(origin_size=(20,61), ch2=2, ch3=3, dropout=0.2, atten=True).to(device)
checkpoint = torch.load('E:/D1/diffusion/my_models/AEmodels/final_with1982_1993.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

parameters = {
    'lon_start': 120,
    'lon_end': 180,
    'lat_start': -10,
    'lat_end': 10,
    'time_start': '2012-01-01',
    'time_end': '2014-12-01'
}
conditions, mask = coders.read_condition_from_file("E:/D1/f01/data/sst.oisst.mon.mean.1982.nc", parameters)#276,1,20,61
mse_loss = torch.nn.MSELoss()

class condition_dataset(torch.utils.data.Dataset):
    def __init__(self, conditions):
        self.data = conditions
        print("valid file number:", str(len(self.data)))
        print("shape of training:", self.data.shape)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)    # self.data是一个变量
    
test_dataset = condition_dataset(conditions)

test_data = test_dataset[:].to(device)

fiture_map = model.encoder(test_data)
re_img = model.decoder(fiture_map)

print(test_data.shape)
print(mask.shape)

test_data.masked_fill_(~mask.to(device), float('nan'))
re_img.masked_fill_(~mask.to(device), float('nan'))


print(test_data.shape)
print(re_img.shape)
# print(mask)

ACC = coders.cal_acc(test_data.squeeze(), re_img.squeeze())
print(ACC)

time_range = pd.date_range(start=parameters['time_start'], end=parameters['time_end'], freq='MS')
plt.plot(time_range, ACC.cpu().detach().numpy())

# 添加每年1月的竖线
for year in range(time_range[0].year, time_range[-1].year + 1):
    plt.axvline(pd.Timestamp(f'{year}-01-01'), color='r', linestyle='--', linewidth=0.5)

plt.show()

fiture_map_np = fiture_map.cpu().detach().numpy()
np.save('E:\D1\diffusion\my_models\my_model_data/sst_map3.npy', fiture_map_np)
