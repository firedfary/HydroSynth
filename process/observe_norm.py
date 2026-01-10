import numpy as np

class DataNormalizer:
    def __init__(self, clip_sigma=3.0):
        """
        Z-Score 标准化 + 裁剪到 ±clip_sigma * std -> 映射到 [-1, 1]

        参数:
        clip_sigma: 裁剪阈值，以标准差为单位，默认 3.0
        """
        self.clip_sigma = clip_sigma
        self.mean_ = None
        self.std_ = None

    def fit(self, arr: np.ndarray):
        """
        根据输入数据计算均值和标准差，排除 NaN
        """
        flat = arr[np.isfinite(arr)]
        self.mean_ = np.mean(flat)
        self.std_ = np.std(flat)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        归一化:
        1. Z-Score: (x - mean) / std
        2. Clip 到 [-clip_sigma, +clip_sigma]
        3. 除以 clip_sigma -> 映射到 [-1, 1]

        NaN 保留不变
        """
        x = arr.astype(np.float32).copy()
        mask = np.isfinite(x)

        # Z-Score
        x[mask] = (x[mask] - self.mean_) / (self.std_ + 1e-8)
        # Clip
        lower, upper = -self.clip_sigma, self.clip_sigma
        x[mask] = np.clip(x[mask], lower, upper)
        # 映射到 [-1, 1]
        x[mask] = x[mask] / self.clip_sigma

        return x

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        """
        把归一化后的数据还原到原始分布:
        1. x_norm * clip_sigma -> 裁剪前的 Z 分数
        2. * std + mean -> 原始尺度

        NaN 保留不变
        """
        x = x_norm.astype(np.float32).copy()
        mask = np.isfinite(x)

        # 反向映射
        x[mask] = x[mask] * self.clip_sigma
        x[mask] = x[mask] * (self.std_ + 1e-8) + self.mean_

        return x

# 示例用法
if __name__ == "__main__":
    data = np.random.randn(346, 300, 350)
    data[np.random.rand(*data.shape) < 0.1] = np.nan  # 加入一些 NaN

    normalizer = DataNormalizer(clip_sigma=3.0)
    normalizer.fit(data)
    data_norm = normalizer.transform(data)
    data_restored = normalizer.inverse_transform(data_norm)

    print("归一化后范围:", np.nanmin(data_norm), np.nanmax(data_norm))
    print("还原误差:", np.nanmean(np.abs(data_restored - data)))
