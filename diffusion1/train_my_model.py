import torch
# import torchvision
# import Unetwork
import os
import warmup_scheduler
import HydroSynth.diffusion.diff_trainer as diff_trainer
import tqdm
import numpy as np
import cProfile
# from fucs import save_image
import fucs
import config
# import unetD4
# import diffusers
from diffusers import UNet2DConditionModel
from torch.utils.tensorboard import SummaryWriter



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




def train(modelconfig:dict):#modelconfig是一个参数。。。
    device = torch.device(modelconfig["device"])
    writer = SummaryWriter(log_dir="./runs")

    target_data_list = []
    target_file_dir = modelconfig["hr_path"]
    target_file_list = os.listdir(target_file_dir)
    for file in target_file_list:
        file_path = os.path.join(target_file_dir, file)
        data = np.load(file_path, allow_pickle=True)#276,64,64
        target_data_list.append(data)

    target_data = torch.from_numpy(np.stack(target_data_list).astype(np.float32))#189,274,32,32
    target_data = torch.unsqueeze(target_data, dim=2)
    target_data = target_data.view(-1, 1, 32, 32)#51786,1,32,32
    data_mask = ~torch.isnan(target_data)

    # for i in range(target_data.shape[0]):
    #     if not torch.isnan(target_data[i]).any():
    #         print("nan in target_data", i)
    #         save_image(target_data[i], path='E:/D1/diffusion/my_models/my_model_data/picture/', step=str(i)+'real')

    model_file_dir = modelconfig["lr_path"]
    model_file_list = os.listdir(model_file_dir)
    assert model_file_list == target_file_list, '顺序完全一致！'
    if model_file_list == target_file_list:
        print("顺序完全一致！")
    model_data_list = []
    for file in model_file_list:
        file_path = os.path.join(model_file_dir, file)
        data = np.load(file_path, allow_pickle=True)#274,10,32,32
        model_data_list.append(data)
    condition = torch.from_numpy(np.stack(model_data_list).astype(np.float32))#189,274,10,32,32
    condition = condition.view(-1, 10, 32, 32)#51786,10,32,32
    zero_count = (condition == 0).sum().item()
    print(f"condition 中 0 的个数为: {zero_count}")
    fucs.plot_anomaly_distribution(condition, 0, 1)

    # condition = fucs.symmetric_max_normalize(condition)
    # model_data = torch.unsqueeze(model_data, dim=2).view(-1, 1, 64, 64)#72080,1,64,64


    # sst = torch.from_numpy(np.load(modelconfig["sst_path"]).astype(np.float32))#212,3,64,64
    # # total_month = sst.shape[0]#212
    # num_patch = len(target_data_list)
    # sst = torch.unsqueeze(sst, dim=0).repeat(num_patch, 1, 1, 1, 1).view(-1, 3, 64, 64)#340,212,3,64,64
    # # selected_files = np.random.choice(file_list, 60, replace=False)
    # condition = torch.cat((sst, condition), dim=1)#72080,4,64,64
    dataset = npy_dataset(target_data, condition, data_mask)#49,276,64,64 注意276不能是channel，而且channel只能是1+16.


    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=modelconfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    # net_model = Unetwork.UNet(
    #     T=modelconfig["T"],
    #     in_ch=11,#4个条件通道+1个目标通道
    #     ch=modelconfig["channel"],
    #     ch_mult=modelconfig["channel_mult"],
    #     atten=modelconfig["atten"],
    #     num_res_block=modelconfig["num_res_block"],
    #     dropout=modelconfig["dropout"],
    #     cond_dim=modelconfig["cond_dim"],
    # )

    net_model = UNet2DConditionModel(
        sample_size=32,
        in_channels=1,            # Stable Diffusion 的 latent 空间通道数
        out_channels=1,
        block_out_channels=(128, 256, 512, 512),
        cross_attention_dim=10,  # 条件嵌入的维度（如文本编码的 hidden_size）
        attention_head_dim=8,     # 注意力头数
    )


    if modelconfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelconfig["save_weight_dir"],
            modelconfig["training_load_weight"]), map_location=device))
        
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelconfig["lr"], weight_decay=1e-4)
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelconfig["epoch"], eta_min=0, last_epoch=-1
    )

    #渐进调度器
    warmupScheduler = warmup_scheduler.GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=modelconfig["multiplier"], 
        total_epoch=modelconfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )
    #扩散训练器
    trainer = diff_trainer.GaussianDiffusionTrainer(
        net_model, modelconfig["bata_1"], modelconfig["bata_T"], modelconfig["T"]
    ).to(device=device)


    # condition_layer_normal = torch.nn.LayerNorm((64,64)).to(device=device)

    # 启动！！！！！
    for e in range(modelconfig["epoch"]):
        # torch.cuda.empty_cache()
        with tqdm.tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for step, (x_0, condition, mask) in enumerate(tqdmDataLoader):
                optimizer.zero_grad()
                x_0 = x_0.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                condition = condition.to(device, non_blocking=True)

                loss = trainer(x_0, condition, mask).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelconfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img_shape": x_0.shape,
                    "lr": optimizer.state_dict()['param_groups'][0]['lr']
                })
                # 记录loss到tensorboard
                global_step = e * len(dataloader) + step
                writer.add_scalar('Loss/train', loss.item(), global_step)
            warmupScheduler.step()

        torch.save(
            net_model.state_dict(), os.path.join(modelconfig["save_weight_dir"], 'ckpt'+str(e)+'_.pt')
        )


if __name__ == '__main__':
    profiler = cProfile.Profile()
    modelconfig = config.vscodeconfig

    profiler.enable()
    train(modelconfig)
    profiler.disable()
    profiler.dump_stats('profile_output2.prof')









