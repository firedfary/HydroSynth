import numpy as np
import os
import glob
import config

def restore_to_patches(result_dir, hr_path, output_dir):
    """
    将result文件夹中的结果还原为原始的patch文件格式
    
    参数:
        result_dir: 包含所有小图像块的文件夹路径
        hr_path: 原始HR数据文件夹路径（用于获取文件名列表）
        output_dir: 还原后的patch保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取result文件夹下所有npy文件并排序
    file_paths = glob.glob(os.path.join(result_dir, '*.npy'))
    # 按文件名中的数字排序
    file_paths = sorted(file_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # 2. 将所有文件加载并整理为(N, 1, 32, 32)的形式
    patch_list = []
    for file_path in file_paths:
        data = np.load(file_path)  # 形状应为(1,1,32,32)
        # 确保数据形状正确
        if data.ndim == 4 and data.shape[1] == 1:
            patch_list.append(data)
        else:
            # 处理可能的形状不一致
            data = data.reshape(1, 1, 32, 32)
            patch_list.append(data)
    
    # 将所有patch堆叠为一个数组
    all_patches = np.concatenate(patch_list, axis=0)  # 形状为(N, 1, 32, 32)
    total_patches = all_patches.shape[0]
    print(f"总patch数量: {total_patches}")
    
    # 3. 获取原始文件名列表
    target_file_list = os.listdir(hr_path)
    num_files = len(target_file_list)
    print(f"原始文件数量: {num_files}")
    
    # 4. 计算每个文件包含的patch数量
    patches_per_file = total_patches // num_files
    if total_patches % num_files != 0:
        print(f"警告: 总patch数{total_patches}不能被文件数{num_files}整除")
    
    print(f"每个文件包含的patch数量: {patches_per_file}")
    
    # 5. 按原始文件结构还原patch
    for file_idx, file_name in enumerate(target_file_list):
        # 计算当前文件的patch范围
        start_idx = file_idx * patches_per_file
        end_idx = start_idx + patches_per_file
        
        # 提取当前文件的所有patch
        file_patches = all_patches[start_idx:end_idx]
        
        # 去除多余的维度 (N, 1, 32, 32) -> (N, 32, 32)
        file_patches = np.squeeze(file_patches, axis=1)
        
        # 保存还原后的patch文件
        save_path = os.path.join(output_dir, file_name)
        np.save(save_path, file_patches)
        print(f"已保存: {save_path} ({file_patches.shape})")
    
    print(f"所有patch已还原到: {output_dir}")

# 使用示例
if __name__ == "__main__":
    # 加载配置文件
    modelconfig = config.vscodeconfig
    
    # 设置路径
    result_dir = "E:/D1/diffusion/my_models/my_model_data/result2"  # 替换为实际的result路径
    hr_path = modelconfig["hr_path"]  # 原始HR数据路径
    output_dir = "E:/D1/diffusion/my_models/my_model_data/restored_patches2"  # 还原后的patch保存路径
    
    restore_to_patches(result_dir, hr_path, output_dir)