import torch
import os
import warmup_scheduler
import tqdm
import numpy as np
import fucs
import config
from sklearn.model_selection import train_test_split  # 添加导入
# from torch.utils.tensorboard import SummaryWriter
from unet_model import UNet

class npy_dataset(torch.utils.data.Dataset):
    def __init__(self, data, condition, mask):
        self.data = data
        self.condition = condition
        self.mask = mask
        print("valid file number:", str(len(self.data)))
        print("shape of training:", self.data.shape)

    def __getitem__(self, index):
        return self.data[index], self.condition[index], self.mask[index]

    def __len__(self):
        return len(self.data)    # self.data是一个变量


def train():
    device = torch.device(config.vscodeconfig["device"])
    target_file_dir = config.vscodeconfig["hr_path"]

    target_data = target_file_dir+'/1d.npy'
    target_data = torch.tensor(np.load(target_data))
    target_data = torch.unsqueeze(target_data, dim=1)
    # target_data = target_data.view(-1, 1, 120, 140)#51786,1,32,32
    data_mask = torch.isnan(target_data)

    model_file_dir = config.vscodeconfig["lr_path"]
    model_data = np.load(model_file_dir+'/1d.npy')
    condition = torch.from_numpy(model_data)#189,274,10,32,32

    # ============= 新增代码: 分割训练集和测试集 =============
    # 生成随机索引
    indices = np.arange(len(target_data))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=config.vscodeconfig["test_ratio"], 
        random_state=config.vscodeconfig["seed"]
    )
    
    # 创建训练集和测试集
    train_dataset = npy_dataset(
        target_data[train_idx], 
        condition[train_idx], 
        data_mask[train_idx]
    )
    test_dataset = npy_dataset(
        target_data[test_idx], 
        condition[test_idx], 
        data_mask[test_idx]
    )
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.vscodeconfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.vscodeconfig["batch_size"],
        shuffle=False,  # 测试集不需要shuffle
        num_workers=2,
        pin_memory=True
    )
    # ====================================================

    model = UNet(n_channels=10, n_classes=1)
    if config.vscodeconfig["training_load_weight"] is not None:
        model.load_state_dict(torch.load(os.path.join(
            config.vscodeconfig["save_weight_dir"],
            config.vscodeconfig["training_load_weight"]), map_location=device))

    model = model.to(config.vscodeconfig["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.vscodeconfig["lr"])
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.vscodeconfig["epoch"], eta_min=0, last_epoch=-1
    )
    warmupScheduler = warmup_scheduler.GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=config.vscodeconfig["multiplier"], 
        total_epoch=config.vscodeconfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )
    loss_fn = torch.nn.MSELoss()
    # writer = SummaryWriter(config.vscodeconfig["log_dir"])
    
    best_test_loss = float('inf')  # 记录最佳测试损失
    
    for e in range(config.vscodeconfig["epoch"]):
        # 训练阶段
        model.train()
        train_losses = []
        with tqdm.tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for step, (x_0, condition, mask) in enumerate(tqdmDataLoader):
                optimizer.zero_grad()
                x_0 = x_0.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                condition = condition.to(device, non_blocking=True)

                # 前向传播得到模型输出
                output = model(condition)
                output[mask] = float('nan')
                # 使用 mask 对模型输出和目标数据进行掩码处理
                masked_output = output[~mask]
                masked_x_0 = x_0[~mask]
                # 计算掩码区域的损失
                loss = loss_fn(masked_output, masked_x_0)
                # loss = 1-fucs.cal_acc(output[:,0,:,:], x_0[:,0,:,:]).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.vscodeconfig["grad_clip"])
                optimizer.step()
                
                train_losses.append(loss.item())
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img_shape": x_0.shape,
                    "lr": optimizer.state_dict()['param_groups'][0]['lr']
                })
            warmupScheduler.step()
        
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {e} - Avg Train Loss: {avg_train_loss:.6f}")
        
        # ============= 新增代码: 测试阶段 =============
        model.eval()
        test_losses = []
        with torch.no_grad():
            with tqdm.tqdm(test_dataloader, desc=f"Testing Epoch {e}", dynamic_ncols=True) as test_tqdm:
                for x_0, condition, mask in test_tqdm:
                    x_0 = x_0.to(device)
                    mask = mask.to(device)
                    condition = condition.to(device)
                    
                    output = model(condition)
                    output[mask] = float('nan')
                    
                    masked_output = output[~mask]
                    masked_x_0 = x_0[~mask]
                    
                    loss = loss_fn(masked_output, masked_x_0)
                    test_losses.append(loss.item())
                    
                    test_tqdm.set_postfix(ordered_dict={
                        "test_loss": loss.item()
                    })
        
        avg_test_loss = np.mean(test_losses)
        print(f"Epoch {e} - Test Loss: {avg_test_loss:.6f}")
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(
                model.state_dict(), 
                os.path.join(config.vscodeconfig["save_weight_dir"], 'best_unet.pt')
            )
            print(f"Saved best model with test loss: {best_test_loss:.6f}")
        # ============================================
        
        # 保存常规检查点
        torch.save(
            model.state_dict(), 
            os.path.join(config.vscodeconfig["save_weight_dir"], f'ckptunet_{e}.pt')
        )


if __name__ == '__main__':
    train()
    # os.system("shutdown -s -t 0 ")