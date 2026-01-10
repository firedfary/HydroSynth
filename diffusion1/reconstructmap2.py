import numpy as np
import os
import glob

def restore_image(save_dir, original_width=350, original_height=300, size_per_image=32, stride=16):
    # 生成与原分割一致的坐标列表
    x_list = np.arange(0, original_width - size_per_image, stride)
    y_list = np.arange(0, original_height - size_per_image, stride)
    
    # 获取所有.npy文件
    file_paths = glob.glob(os.path.join(save_dir, '*.npy'))
    if not file_paths:
        raise ValueError("未找到.npy文件")
    
    # 确定通道数
    sample = np.load(file_paths[0])
    channels = sample.shape[0]
    
    # 初始化累加数组和计数数组
    sum_array = np.zeros((channels, original_height, original_width))
    count_array = np.zeros((channels, original_height, original_width))
    
    for file_path in file_paths:
        # 解析坐标信息
        filename = os.path.basename(file_path)
        x_id, y_id = map(int, os.path.splitext(filename)[0].split('_'))
        
        # 获取原始坐标
        try:
            x = x_list[x_id]
            y = y_list[y_id]
        except IndexError:
            print(f"跳过无效文件：{filename}")
            continue
        
        # 加载数据并处理NaN
        sub_image = np.load(file_path)
        
        # 计算实际边界
        end_x = min(x + size_per_image, original_width)
        end_y = min(y + size_per_image, original_height)
        
        # 裁剪子图和掩码
        valid_sub = sub_image[:, :end_y-y, :end_x-x]
        mask = ~np.isnan(valid_sub)
        
        # 累加有效值
        sum_array[:, y:end_y, x:end_x] += np.where(mask, valid_sub, 0)
        count_array[:, y:end_y, x:end_x] += mask.astype(int)
    
    # 计算平均值并处理未覆盖区域
    restored = sum_array / count_array
    restored[count_array == 0] = np.nan  # 标记未覆盖区域
    
    return restored

# 使用示例
save_path = "E:/D1/diffusion/my_models/my_model_data/restored_patches2"  # 替换为实际路径
restored_image = restore_image(save_path)

# 保存还原结果
np.save('restored_image2.npy', restored_image)
print("图像还原完成，已保存为restored_image.npy")