import torch
import numpy as np
from matplotlib import pyplot as plt
import diffusers
import os
from fucs import save_image
import fucs
from diffusers import UNet2DConditionModel
import config
from sklearn.cluster import KMeans

modelconfig = config.vscodeconfig
print(modelconfig['eval_load_weight'])
with torch.no_grad():
    device = torch.device(modelconfig['device'])
    model = UNet2DConditionModel(
        sample_size=32,
        in_channels=1,
        out_channels=1,
        block_out_channels=(128, 256, 512, 512),
        cross_attention_dim=10,
        attention_head_dim=8,
    )
    ckpt = torch.load(
        os.path.join(modelconfig['save_weight_dir'], modelconfig['eval_load_weight']),
        map_location=device
    )
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

target_data_list = []
target_file_dir = modelconfig["hr_path"]
target_file_list = os.listdir(target_file_dir)
for file in target_file_list:
    file_path = os.path.join(target_file_dir, file)
    data = np.load(file_path, allow_pickle=True)
    target_data_list.append(data)

target_data = torch.from_numpy(np.stack(target_data_list).astype(np.float32))
target_data = torch.unsqueeze(target_data, dim=2)
target_data = target_data.view(-1, 1, 32, 32)

model_file_dir = modelconfig["lr_path"]
model_file_list = os.listdir(model_file_dir)
assert model_file_list == target_file_list, '顺序不一致！'
if model_file_list == target_file_list:
    print("顺序完全一致！")
model_data_list = []
for file in model_file_list:
    file_path = os.path.join(model_file_dir, file)
    data = np.load(file_path, allow_pickle=True)
    model_data_list.append(data)
condition = torch.from_numpy(np.stack(model_data_list).astype(np.float32))
condition = condition.view(-1, 10, 32, 32)

# 可根据显存调整
batch_size = 10
num_samples = 50
n_clusters = 3
H, W = 32, 32

# DDIM调度器定义移到循环外部
ddim_sch = diffusers.schedulers.scheduling_ddim.DDIMScheduler(num_train_timesteps=200, 
                                                              beta_schedule='squaredcos_cap_v2', 
                                                              clip_sample=False)
ddim_sch.set_timesteps(15)

def process_batch(start_idx, batch_size):
    end_idx = min(start_idx + batch_size, len(target_data))
    x0_batch = target_data[start_idx:end_idx].to(device)
    condition_batch = condition[start_idx:end_idx].to(device)
    current_batch_size = x0_batch.size(0)

    # 重复x0_batch和condition_batch以生成num_samples个样本
    x0_batch_repeated = x0_batch.repeat_interleave(num_samples, dim=0)
    condition_batch_repeated = condition_batch.repeat_interleave(num_samples, dim=0)

    # 生成噪声
    noise_to_add = torch.randn_like(x0_batch_repeated)
    inf_image = noise_to_add

    # 准备encoder_hidden_states
    encoder_hidden_states = condition_batch_repeated.flatten(2).permute(0, 2, 1)
    encoder_hidden_states0 = torch.ones_like(encoder_hidden_states)

    # 并行DDIM采样
    with torch.no_grad():
        for t in ddim_sch.timesteps:
            t = t.to(device)
            eps = model(inf_image, t, encoder_hidden_states).sample
            eps1 = model(inf_image, t, encoder_hidden_states0).sample
            total_eps = eps1 + 12 * (eps - eps1)
            inf_image = ddim_sch.step(total_eps, t, inf_image, eta=0.0)[0]

    # 重塑生成的样本为 (batch_size, num_samples, 1, 32, 32)
    all_samples = inf_image.view(current_batch_size, num_samples, 1, 32, 32)
    all_samples_cpu = all_samples.cpu()
    x0_batch_cpu = x0_batch.cpu()

    # 对每个x0进行后续处理
    for i in range(current_batch_size):
        show_which = start_idx + i
        save_dir = os.path.join(modelconfig['picture_save_path'], str(show_which))
        os.makedirs(save_dir, exist_ok=True)

        x0 = x0_batch_cpu[i]
        samples = all_samples_cpu[i]

        # 保存x0
        save_image(x0, names=[f'x0_{show_which}'], 
                   path=save_dir,
                   filename=f'x0_{show_which}.svg',
                   show=False)

        # 计算准确率
        acclist = [fucs.cal_acc(sample, x0) for sample in samples]
        plt.plot(acclist)
        plt.xlabel('Index')
        plt.ylabel('ACC')
        for m, acc in enumerate(acclist):
            plt.text(m, float(acc), f'{float(acc):.2f}', ha='center', va='bottom')
        plt.savefig(os.path.join(save_dir, 'acc.svg'), format='svg')
        plt.close()

        # KMeans聚类
        data = samples.view(num_samples, -1).numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(data)
        counts = np.bincount(labels, minlength=n_clusters)
        for cluster_id, count in enumerate(counts):
            print(f"簇 {cluster_id} 的样本数量: {count}")

        max_cluster = np.argmax(counts)
        print(f"样本最多的簇为: {max_cluster}")
        max_cluster_indices = np.where(labels == max_cluster)[0]
        max_cluster_samples = data[max_cluster_indices]
        mean_sample = np.mean(max_cluster_samples, axis=0)
        mean_sample_reshaped = mean_sample.reshape(1, 1, H, W)
        print("平均样本 shape:", mean_sample_reshaped.shape)
        save_image(x0, torch.from_numpy(mean_sample_reshaped), names=[f'x0{show_which}', 'mean_sample'],
                   path=save_dir,
                   filename='mean_sample.svg',
                   show=False)
        np.save(os.path.join("E:/D1/diffusion/my_models/my_model_data/result3", f'{show_which}.npy'), mean_sample_reshaped)

# 设置batch_size为10
batch_size = 2
for start_idx in range(0, len(target_data), batch_size):
    process_batch(start_idx, batch_size)