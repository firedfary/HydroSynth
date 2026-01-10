import torch
import coders
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = coders.AutoEncoder(origin_size=(20,61), ch2=2, ch3=3, dropout=0.2, atten=True).to(device)
checkpoint = torch.load('E:\D1\diffusion\my_models\model_epoch900.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

parameters = {
    'lon_start': 120,
    'lon_end': 180,
    'lat_start': -10,
    'lat_end': 10,
    'time_start': '1993-01-01',
    'time_end': '1993-12-01'
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

n_img = 2
test_data = test_dataset[n_img:n_img+1].to(device)#这batchsize=1
print(test_dataset[0:1].shape)
fiture_map = model.encoder(test_data)
re_img = model.decoder(fiture_map)

print(torch.squeeze(test_data).shape)
print(torch.squeeze(re_img).shape)

fig, axs = plt.subplots(3, 1, figsize=(7, 6))
im1 = axs[0].imshow(torch.squeeze(test_data).detach().cpu().numpy(), vmin=0.75, vmax=1.0)
cbar = fig.colorbar(im1, ax=axs[0])
axs[0].set_title('Original Image')
axs[0].axis('off')

im2 = axs[1].imshow(fiture_map[0][0].detach().cpu().numpy())
axs[1].set_title('Fiture Map')
cbar = fig.colorbar(im2, ax=axs[1])
axs[1].axis('off')

im3 = axs[2].imshow(torch.squeeze(re_img).detach().cpu().numpy(), vmin=0.75, vmax=1.0)
axs[2].set_title('Reconstructed Image')
cbar = fig.colorbar(im3, ax=axs[2])
axs[2].axis('off')

obs = test_data.masked_fill(~mask[0:1].to(device), float('nan'))
re = re_img.masked_fill(~mask[0:1].to(device), float('nan'))
print(obs.shape)
print(re.shape)
print(mask)

ACC = coders.cal_acc(obs[0], re[0]).item()
fig.text(0.15, 0.9, f'ACC: {ACC:.2f}', ha='center', va='center')
print(ACC)
plt.show()





