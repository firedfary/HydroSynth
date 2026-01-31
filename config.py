import torch
import os
import json
from datetime import datetime

# print('current platform: ', os.name)
#1是训练集，2是测试集
lr_foldr = 'lr_unet'
hr_foldr = 'hr_unet'
save_weight_foldr = 'weight_t0'
picture_foldr = 'picture'
log_foldr = 'log_ind'

local_data_path = 'e:/D1/diffusion/HydroSynth/datas/UNet_data/'
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
if not os.path.exists(log_path):
    os.makedirs(log_path)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join(log_path, f"run_{run_id}"), exist_ok=True)
os.makedirs(os.path.join(picture_save_path, f"run_{run_id}"), exist_ok=True)
os.makedirs(os.path.join(save_weight_path, f"run_{run_id}"), exist_ok=True)

modelconfig = {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'batch_size': 16,
        'T': 200,
        'channel': 64,
        'channel_mult': [2,4,4,2],
        'atten': [0,1,2,3],
        'num_res_block': 6,
        'dropout': 0.6,
        'save_weight_path': os.path.join(save_weight_path, f"run_{run_id}"),
        'train_load_weight': None,
        'eval_load_weight': 'ckptunet_1.pt',
        'picture_save_path': os.path.join(picture_save_path, f"run_{run_id}"),
        'log_path': os.path.join(log_path, f"run_{run_id}"),
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


def save_config(config_dict=None, file_path=None, indent=4):
    """
    将配置字典保存为JSON格式文件
    
    Args:
        config_dict (dict): 要保存的配置字典，默认为modelconfig
        file_path (str): 保存文件的路径，默认为save_weight_path中的config_{run_id}.json
        indent (int): JSON缩进空格数
    
    Returns:
        str: 保存的配置文件路径
    """
    if config_dict is None:
        config_dict = modelconfig
    
    # 处理无法序列化的对象（如torch.device）
    serializable_config = {}
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            # 将torch.device转换为字符串表示
            serializable_config[key] = str(value)
        elif isinstance(value, (list, dict, int, float, str, bool)) or value is None:
            # 基本类型可以直接序列化
            serializable_config[key] = value
        else:
            # 其他类型转换为字符串
            serializable_config[key] = str(value)
    
    # 默认保存路径
    if file_path is None:
        file_path = os.path.join(modelconfig['save_weight_path'], f'config_{run_id}.json')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=indent, ensure_ascii=False)
        print(f"配置已保存到: {file_path}")
        return file_path
    except Exception as e:
        raise RuntimeError(f"保存配置文件失败: {e}")


# 在模块加载时自动保存配置
def auto_save_config():
    """自动保存配置到save_weight_path"""
    try:
        save_config()
    except Exception as e:
        print(f"自动保存配置失败: {e}")

# 模块导入时自动执行
# _auto_save_config()