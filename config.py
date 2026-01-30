import torch
import os
import json

# print('current platform: ', os.name)
#1是训练集，2是测试集
lr_foldr = 'lr_unet'
hr_foldr = 'hr_unet'
save_weight_foldr = 'weight_t0'
picture_foldr = 'picture'
log_foldr = 'log_ind'

local_data_path = 'e:/D1/diffusion/HydroSynth/datas/UNet_3D_data/'
colab_data_path = '/content/drive/MyDrive/my_models/my_model_data/'

if os.name == 'nt':
    lr_path = local_data_path + lr_foldr
    hr_path = local_data_path + hr_foldr
    sst_file = local_data_path + 'sst_6chan.npy'
    save_weight_path = local_data_path + save_weight_foldr
    picture_save_path = local_data_path + picture_foldr
    log_path = local_data_path + log_foldr 
elif os.name == 'posix':
    lr_path = colab_data_path + lr_foldr
    hr_path = colab_data_path + hr_foldr
    sst_file = colab_data_path + 'sst_6chan.npy'
    save_weight_path = colab_data_path + save_weight_foldr
    picture_save_path = colab_data_path + picture_foldr
    log_path = colab_data_path + log_foldr 
else:
    raise ValueError('Unknown OS')

if not os.path.exists(save_weight_path):
    os.makedirs(save_weight_path)
if not os.path.exists(picture_save_path):
    os.makedirs(picture_save_path)

modelconfig = {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'batch_size': 16,
        'T': 200,
        'channel': 64,
        'channel_mult': [2,4,4,2],
        'atten': [0,1,2,3],
        'num_res_block': 6,
        'dropout': 0.6,
        'save_weight_path': save_weight_path,
        'train_load_weight': None,
        'eval_load_weight': 'ckptunet_1.pt',
        'picture_save_path': picture_save_path,
        'log_path': log_path,
        'lr': 0.0005,
        'epoch': 502,
        'multiplier': 1.0,
        'bata_1':0.0001,
        'bata_T':0.02,
        'grad_clip': 2.0,
        'lr_path': lr_path,
        'hr_path': hr_path,
        'sst_file': sst_file,
        'cond_dim': 10,
        "test_ratio": 0.2,
        "seed": 42,
        "n_pcs": 10,
        'pc_window': 3,
        'pc_step': 1,
        'horizon': 6
    }

def load_config(json_file_path=None, merge_mode='update'):
    """
    读取JSON配置文件并将其合并到modelconfig中
    
    Args:
        json_file_path (str): JSON配置文件的路径
        merge_mode (str): 合并模式，可选值：
            - 'update': 更新现有键值，保留未在JSON中定义的键
            - 'replace': 完全替换modelconfig为JSON内容
            - 'deep_update': 深度更新嵌套字典
    
    Returns:
        dict: 更新后的modelconfig字典
    """
    if json_file_path is None:
        print("警告：未提供JSON文件路径，返回当前modelconfig")
        return modelconfig
    
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"配置文件不存在: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"读取配置文件失败: {e}")
    
    if merge_mode == 'replace':
        # 完全替换modelconfig
        modelconfig.clear()
        modelconfig.update(json_config)
    elif merge_mode == 'deep_update':
        # 深度更新嵌套字典
        def deep_update(source, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                    deep_update(source[key], value)
                else:
                    source[key] = value
        
        deep_update(modelconfig, json_config)
    else:
        # 默认模式：更新现有键值
        modelconfig.update(json_config)
    
    print(f"成功加载配置文件: {json_file_path}")
    print(f"合并模式: {merge_mode}")
    
    return modelconfig