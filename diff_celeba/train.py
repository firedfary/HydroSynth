import torch
import torch.utils
import torch.utils.data
import torchvision
import Unetwork
import os
import warmup_scheduler
import HydroSynth.diffusion.diff_trainer as diff_trainer
import tqdm
import threading

def save_model_async(model, path):
    """异步保存模型权重"""
    state_dict = model.state_dict()
    threading.Thread(target=torch.save, args=(state_dict, path)).start()

def train(modelConfig:dict):#modelconfig是一个参数。。。
    device = torch.device(modelConfig["device"])

    
    dataset = torchvision.datasets.ImageFolder(
        root=r"E:\D1\diffusion\my_models\data128",
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    # 预加载所有数据到内存以避免重复磁盘I/O
    print("预加载数据集到内存...")
    all_images = []
    all_labels = []
    for img, label in tqdm.tqdm(dataset, desc="Loading data"):
        all_images.append(img)
        all_labels.append(label)
    
    dataset = torch.utils.data.TensorDataset(
        torch.stack(all_images), 
        torch.tensor(all_labels, dtype=torch.long)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=0,  # 内存中加载，无需多进程
        drop_last=True,
        pin_memory=True
    )

    net_model = Unetwork.UNet(
        T=modelConfig["T"],
        in_ch=13,
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        atten=modelConfig["atten"],
        num_res_block=modelConfig["num_res_block"],
        dropout=modelConfig["dropout"]
    )


    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(#这是一个继承于torch.nn.module的东西（大概）
            modelConfig["save_weight_dir"],
            modelConfig["training_load_weight"]), map_location=device))
        
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1
    )

    #渐进调度器
    warmupScheduler = warmup_scheduler.GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=modelConfig["multiplier"], 
        total_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )
    #扩散训练器
    trainer = diff_trainer.GaussianDiffusionTrainer(
        net_model, modelConfig["bata_1"], modelConfig["bata_T"], modelConfig["T"]
    ).to(device=device)


    conditions = torch.zeros((8, 10, 128, 128)).to(device)
    mask = torch.ones((8, 3, 128, 128), dtype=bool)
    # 启动！！！！！
    for e in range(modelConfig["epoch"]):
        # torch.cuda.empty_cache()  # 移除每epoch清空缓存，避免等待
        with tqdm.tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0, conditions=conditions, mask=mask).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss": loss.item(),
                    "img_shape": x_0.shape,
                    "lr": optimizer.state_dict()['param_groups'][0]['lr']
                })
        warmupScheduler.step()
        if e % 10 == 0:
            save_path = os.path.join(modelConfig["save_weight_dir"], str(e)+'_.pt')
            save_model_async(net_model, save_path)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelconfig = {
    'device': device,
    'batch_size': 8,
    'T': 100,
    'channel': 16,
    'channel_mult': [2,2,2,2],
    'atten': [2,3],
    'num_res_block': 3,                  #不知道是多少，随便写一个吧
    'dropout': 0.2,                      #dropout置零率，在0到1之间
    'training_load_weight': '40_.pt',        #填从那个文件夹里加载权重
    'eval_load_weight': None,  #填从那个文件夹里加载权重
    'save_weight_dir': 'E:/D1/diffusion/my_models/diff_test/weight/',#把权重保存在那个文件夹里
    'picture_save_path': 'E:/D1/diffusion/my_models/diff_test/picture/',#把生成的图片保存在那个文件夹里
    'lr': 0.00001,
    'epoch': 100,                        #不知道是多少，x修改了这个注释
    'multiplier': 1.0,                   #不知道填啥
    'bata_1':0.0001,
    'bata_T':0.02,
    'grad_clip': 2.0,
    }


    train(modelconfig)









