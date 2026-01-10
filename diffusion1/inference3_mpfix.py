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
assert model_file_list == target_file_list, '顺序完全一致！'
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
batch_size = 50
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
    
    for i in range(x0_batch.size(0)):
        show_which = start_idx + i
        save_dir = os.path.join(modelconfig['picture_save_path'], str(show_which))
        os.makedirs(save_dir, exist_ok=True)
        
        x0 = x0_batch[i:i+1]
        condition0 = condition_batch[i:i+1]
        encoder_hidden_states = condition0.flatten(2).permute(0, 2, 1)
        encoder_hidden_states0 = torch.ones_like(encoder_hidden_states)
        
        # print(x0.shape, condition0.shape)
        # print(x0.nanmean(), x0.std())
        save_image(x0, names=[f'x0_{show_which}'], 
                   path=save_dir,
                   filename=f'x0_{show_which}.svg',
                   show=False)
        
        x0 = torch.nan_to_num(x0, nan=0.0)
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        all_samples = torch.empty((num_samples, 1, H, W), dtype=torch.float32)
        idx = 0
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            noise_to_add = torch.randn((current_batch_size, *x0.shape[1:]), device=device).float()
            inf_image = noise_to_add
            with torch.no_grad():
                for t in ddim_sch.timesteps:
                    t = t.to(device)
                    eps = model(inf_image, t, encoder_hidden_states.expand(current_batch_size, -1, -1)).sample
                    eps1 = model(inf_image, t, encoder_hidden_states0.expand(current_batch_size, -1, -1)).sample
                    total_eps = eps1 + 12 * (eps - eps1)
                    inf_image = ddim_sch.step(total_eps, t, inf_image, eta=0.0)[0]
            all_samples[idx:idx+current_batch_size] = inf_image.detach().cpu()
            idx += current_batch_size
            # for j in range(current_batch_size):
            #     save_image(x0, inf_image[j:j+1], names=[f'x0{show_which}', f'{batch_idx*batch_size+j}th predict'],
            #                 path=save_dir,
            #                 filename=f'{batch_idx*batch_size+j}th predict.svg',
            #                 show=False)
            #     break
        
        acclist = []
        x0_cpu = x0[0].cpu()
        for k in range(num_samples):
            acc = fucs.cal_acc(all_samples[k], x0_cpu)
            # print(k, 'acc:', acc)
            acclist.append(float(acc))
        plt.plot(acclist)
        plt.xlabel('Index')
        plt.ylabel('ACC')
        for m, acc in enumerate(acclist):
            plt.text(m, acc, f'{acc:.2f}', ha='center', va='bottom')
        plt.savefig(os.path.join(save_dir, 'acc.svg'), format='svg')
        plt.close()
        
        data = all_samples.view(num_samples, -1).numpy()
        print(data.shape)
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
        np.save(os.path.join("E:/D1/diffusion/my_models/my_model_data/result", f'{show_which}.npy'), mean_sample_reshaped)

# 设置batch_size为10
batch_size = 10
for start_idx in range(0, len(target_data), batch_size):
    process_batch(start_idx, batch_size)