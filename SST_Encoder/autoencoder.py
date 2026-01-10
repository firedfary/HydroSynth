import torch
import coders as coders
from torch.utils.tensorboard import SummaryWriter

parameters = {
    'lon_start': 120,
    'lon_end': 180,
    'lat_start': -10,
    'lat_end': 10,
    'time_start': '1982-01-01',
    'time_end': '1993-12-01'
}
conditions, mask = coders.read_condition_from_file("E:/D1/f01/data/sst.oisst.mon.mean.1982.nc", parameters)#276,1,20,61
print(torch.max(conditions))

class condition_dataset(torch.utils.data.Dataset):
    def __init__(self, conditions):
        self.data = conditions
        print("valid file number:", str(len(self.data)))
        print("shape of training:", self.data.shape)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)    # self.data是一个变量
    
train_dataset = condition_dataset(conditions)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=3,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = coders.AutoEncoder(origin_size=(20,61), ch2=2, ch3=3, dropout=0.2, atten=True).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
whiter =  SummaryWriter(log_dir="E:\D1\diffusion\my_models")
mse_loss = torch.nn.MSELoss()

for epoch in range(1000):
    train_loss = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = mse_loss(output[mask[0].unsqueeze(0).repeat(3, 1, 1, 1)], data[mask[0].unsqueeze(0).repeat(3, 1, 1, 1)])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print("epoch:", epoch, "loss:", loss)
    whiter.add_scalar('train_loss', train_loss, epoch)
    if epoch % 100 == 0:
        coders.save_model(model, optimizer, epoch, save_dir="E:\D1\diffusion\my_models", filename_prefix="model")








