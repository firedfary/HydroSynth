from diffusers import UNet2DModel
import torch
from diffusers import DDPMScheduler
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from PIL import Image
from tqdm.auto import tqdm
import numpy as np


def create_celeba_unet():
    model = UNet2DModel(
        sample_size=128,  # 目标图像分辨率
        in_channels=3,    # 输入通道 (RGB)
        out_channels=3,   # 输出通道 (RGB)
        layers_per_block=2,  # 每个 ResNet 块的层数
        
        # 通道数配置：从浅层到深层逐渐增加
        # 128 -> 256 -> 512 -> 512 这种配置对于 128px 分辨率通常效果很好
        # block_out_channels=(128, 256, 512, 512), 
        block_out_channels=(64, 128, 256, 512),  # 每个 ResNet 块的输出通道数

        
        # 下采样块类型
        down_block_types=(
            "DownBlock2D",      # level 1: 128x128 -> 64x64 (保留纹理细节)
            "DownBlock2D",      # level 2: 64x64 -> 32x32
            "AttnDownBlock2D",  # level 3: 32x32 -> 16x16 (加入 Self-Attention 捕捉人脸结构)
            "AttnDownBlock2D",  # level 4: 16x16 -> 8x8 (最深层，强语义信息)
        ),
        
        # 上采样块类型 (必须与 DownBlock 对称)
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        
        # 归一化组数 (这是 ResNet 的标准配置)
        norm_num_groups=32,
    )
    return model


# --- 1. 配置参数 ---
class Config:
    dataset_path = r"E:\D1\diffusion\my_models\data128" # 你的数据集路径
    output_dir = 'E:/D1/diffusion/my_models/diff_test/weight/'      # 权重保存路径
    image_size = 128
    train_batch_size = 4  # T4 建议 16，若显存够可加到 32
    eval_batch_size = 4    # 测试时生成的图片数量
    num_epochs = 50        # 根据需要调整
    gradient_accumulation_steps = 1
    learning_rate = 1e-4   # 你询问的推荐学习率
    lr_warmup_steps = 500
    save_image_epochs = 1  # 每隔几个 epoch 生成一次预览图
    save_model_epochs = 5  # 每隔几个 epoch 保存一次模型
    mixed_precision = "no" # 开启半精度，T4 必备

config = Config()

# --- 2. 数据准备 ---
preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]), # 映射到 [-1, 1]
])

def train():
    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=config.output_dir
    )

    # 模型与调度器 (使用你之前定义的配置)
    model = create_celeba_unet() # 假设你已经定义了上一轮对话中的这个函数
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear", # 或者 "squaredcos_cap_v2" 效果可能更好
        clip_sample=True        # 训练时将像素值截断在 [-1, 1] 之间，防止数值爆炸
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 数据集
    dataset = datasets.ImageFolder(root=config.dataset_path, transform=preprocess)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # 学习率调度器
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # 通过 Accelerator 准备所有组件
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # --- 3. 训练循环 ---
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            
            # 为图片添加噪声
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # 预测噪声
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # --- 4. 评估与保存 ---
        if accelerator.is_main_process:
            # 定期生成预览图 (这里使用 DDIM 采样以提速)
            if (epoch + 1) % config.save_image_epochs == 0:
                evaluate(config, epoch, accelerator.unwrap_model(model))

            # 定期保存模型 Checkpoint
            if (epoch + 1) % config.save_model_epochs == 0:
                save_path = os.path.join(config.output_dir, f"checkpoint-{epoch}")
                accelerator.save_state(save_path)

def evaluate(config, epoch, model):
    # 使用 DDIM Scheduler 进行快速采样测试
    scheduler = DDIMScheduler.from_config(DDPMScheduler(num_train_timesteps=1000).config)
    scheduler.set_timesteps(50)  # 减少采样步数以提速
    model.eval()
    
    # 初始噪声
    image = torch.randn((config.eval_batch_size, 3, config.image_size, config.image_size)).to(model.device)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling..."):
        with torch.no_grad():
            noise_pred = model(image, t).sample
        image = scheduler.step(noise_pred, t, image).prev_sample

    # 反归一化并保存
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    
    rows = []
    for i in range(config.eval_batch_size):
        rows.append(image[i])
    
    # 横向拼接
    combined_image = np.concatenate(rows, axis=1) 
    full_image = Image.fromarray(combined_image)
    
    save_path = f"E:/D1/diffusion/my_models/diff_test/picture/epoch_{epoch}.png"
    full_image.save(save_path)

if __name__ == "__main__":
    # 请确保你在同一脚本内或 import 进来了上一轮定义的 create_celeba_unet
    train()





