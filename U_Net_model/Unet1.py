import sys
import os
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_proj_root = os.path.normpath(_proj_root)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
import torch
import os
import warmup_scheduler
import tqdm
import numpy as np
from HydroSynth.utils import utils
import config
# config.auto_save_config()
from torch.utils.tensorboard import SummaryWriter
from unetlite import UNetLite




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
    device = torch.device(config.modelconfig["device"])
    target_file_dir = config.modelconfig["hr_path"]

    target_data = target_file_dir+'/hr_dataf1.npy'
    target_data = np.load(target_data).astype(np.float32)
    target_data = np.expand_dims(target_data, 1)
    target_data = torch.from_numpy(target_data)  # 使用from_numpy而不是tensor()
    data_mask = torch.isnan(target_data)

    model_file_dir = config.modelconfig["lr_path"]
    model_data = np.load(model_file_dir+'/lr_data_lead0_f1.npy').astype(np.float32)
    condition = torch.from_numpy(model_data)  # 使用from_numpy而不是tensor()
    # condition = condition.view(-1, 10, 120, 140)#51786,10,32,32

    # ============= 修改：直接在展平后的数据上分割测试集 =============
    num_test_samples = 21  
    
    # 计算训练集和测试集的索引范围
    total_samples = len(target_data)
    train_end_idx = total_samples - num_test_samples
    
    # 创建训练集
    train_target_data = target_data[:train_end_idx]
    train_condition = condition[:train_end_idx]
    train_mask = data_mask[:train_end_idx]

    test_target_data = target_data[train_end_idx:]
    test_condition = condition[train_end_idx:]
    test_mask = data_mask[train_end_idx:]
    
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {len(train_target_data)}, Testing samples: {len(test_target_data)}")

    # 创建数据集
    train_dataset = npy_dataset(train_target_data, train_condition, train_mask)
    test_dataset = npy_dataset(test_target_data, test_condition, test_mask)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.modelconfig["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows上应使用0避免多进程开销
        drop_last=True,
        pin_memory=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.modelconfig["batch_size"],
        shuffle=False,  # 测试集不需要shuffle
        num_workers=0,  # Windows上应使用0避免多进程开销
        pin_memory=True
    )
    # ===========================================================

    model = UNetLite(n_channels=7, n_classes=1)
    if config.modelconfig["train_load_weight"] is not None:
        model.load_state_dict(torch.load(os.path.join(
            config.modelconfig["save_weight_path"],
            config.modelconfig["train_load_weight"]), map_location=device))

    model = model.to(config.modelconfig["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.modelconfig["lr"], weight_decay=1e-5)
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.modelconfig["epoch"], eta_min=0, last_epoch=-1
    )
    warmupScheduler = warmup_scheduler.GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=config.modelconfig["multiplier"], 
        total_epoch=config.modelconfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )
    loss_fn = torch.nn.MSELoss()
    
    # 创建TensorBoard SummaryWriter
    writer = SummaryWriter(config.modelconfig["log_path"])  # 确保配置中有log_path路径
    
    best_test_loss = float('inf')  # 记录最佳测试损失
    
    for e in range(config.modelconfig["epoch"]):
        # 训练阶段
        model.train()
        train_losses = []
        train_Acc = []

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
                # diff = (output - x_0) * (~mask).float()
                # loss = (diff**2).sum() / (~mask).sum()

                # loss = 1-fucs.cal_acc(output[:,0,:,:], x_0[:,0,:,:]).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.modelconfig["grad_clip"])
                optimizer.step()
                Acc = utils.cal_acc(output[:,0,:,:]*100, x_0[:,0,:,:]*100).mean()
                train_Acc.append(Acc.item())
                train_losses.append(loss.item())
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img_shape": x_0.shape,
                    "lr": optimizer.state_dict()['param_groups'][0]['lr']
                })
            warmupScheduler.step()
        
        avg_train_loss = np.mean(train_losses)
        avg_train_Acc = np.nanmean(train_Acc)
        print(f"Epoch {e} - Avg Train Loss: {avg_train_loss:.6f}")
        

        # 记录训练损失和学习率
        writer.add_scalar('Loss/train', avg_train_loss, e)
        writer.add_scalar('Acc/train', Acc, e)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], e)
        
        # 测试阶段
        model.eval()
        test_losses = []
        test_Acc = []
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
                    # diff = (output - x_0) * (~mask).float()
                    # loss = (diff**2).sum() / (~mask).sum()

                    test_losses.append(loss.item())
                    Acc = utils.cal_acc(output[:,0,:,:]*100, x_0[:,0,:,:]*100).mean()
                    test_Acc.append(Acc.item())
                    test_tqdm.set_postfix(ordered_dict={
                        "test_loss": loss.item()
                    })
        
        avg_test_loss = np.mean(test_losses)
        avg_test_Acc = np.nanmean(test_Acc)
        print(f"Epoch {e} - Test Loss: {avg_test_loss:.6f}")
        print(f"Epoch {e} - Test Acc: {avg_test_Acc:.6f}")
        
        # 记录测试损失
        writer.add_scalar('Loss/test', avg_test_loss, e)
        writer.add_scalar('Acc/test', avg_test_Acc, e)
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(
                model.state_dict(), 
                os.path.join(config.modelconfig["save_weight_path"], 'best_unet.pt')
            )
            print(f"Saved best model with test loss: {best_test_loss:.6f}")
        
        # 保存常规检查点
        torch.save(
            model.state_dict(), 
            os.path.join(config.modelconfig["save_weight_path"], f'unet_{e}.pt')
        )
    
    # 训练结束后关闭TensorBoard写入器
    writer.close()


if __name__ == '__main__':
    config.auto_save_config()
    train()
    # os.system("shutdown -s -t 0 ")