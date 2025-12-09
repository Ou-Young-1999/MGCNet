import numpy as np


class ECGAugment:
    """
    ECG 数据增强与标准化类
    支持：
        - 重采样到固定采样频率（如 200 Hz）
        - 高斯噪声
        - 时间扭曲
        - 时间遮蔽
        - z-score / min-max 归一化
    适用于单导联 ECG，如 MIT-BIH AF 或 LTAF 数据集
    """

    def __init__(self,
                 target_fs=200,  # 固定目标采样率（Hz）
                 original_fs=200,  # 原始采样率（Hz），需根据数据设置
                 prob=0.5,
                 noise_std=0.1,  # 高斯噪声强度（相对 std）
                 time_warp_sigma=10,  # 时间扭曲强度
                 time_mask_ratio=0.1,  # 时间遮蔽最大比例
                 amp_scale_range=0.4,  # 幅度缩放范围：最低和最高缩放因子
                 zoom_in_ratio_range=0.5,  # time_zoom_in 截取比例范围
                 zoom_out_expand_range=0.5,  # time_zoom_out 扩展比例范围
                 norm_type='zscore'):
        """
        初始化

        Args:
            target_fs (int): 目标采样频率（Hz），所有信号将重采样至此
            original_fs (int): 原始信号的采样频率（Hz）
            prob (float): 每种增强的触发概率
            noise_std (float): 高斯噪声强度（相对 std）
            time_warp_sigma (float): 时间扭曲强度
            time_mask_ratio (float): 时间遮蔽最大比例
            norm_type (str): 归一化方式 'zscore'、'minmax' 或 None
        """
        self.target_fs = target_fs
        self.original_fs = original_fs
        self.prob = prob
        self.noise_std = noise_std
        self.time_warp_sigma = time_warp_sigma
        self.time_mask_ratio = time_mask_ratio
        self.amp_scale_range = amp_scale_range
        self.zoom_in_ratio_range = zoom_in_ratio_range
        self.zoom_out_expand_range = zoom_out_expand_range
        self.norm_type = norm_type

        # 计算重采样比例
        self.resample_ratio = target_fs / original_fs

    def __call__(self, x):
        """
        处理 ECG 信号：重采样 → 增强 → 归一化

        Args:
            x (np.ndarray): 输入信号，shape (seq_len,) 或 (seq_len, 1)

        Returns:
            np.ndarray: 处理后的信号，长度可能变化（取决于 target_fs）
        """
        x = x.copy()

        # --- 1. 重采样到固定采样频率 ---
        x = self._resample_to_fixed_fs(x)

        # --- 2. 数据增强（建议顺序）---
        # (1) 几何变换类（时间/幅度缩放、扭曲）
        if np.random.rand() < self.prob:
            x = self._amplitude_scaling(x)

        if np.random.rand() < self.prob:
            x = self._vertical_flip(x)  # <-- 新增垂直翻转

        if np.random.rand() < self.prob:
            if np.random.rand() < 0.5:
                x = self._time_zoom_in(x)
            else:
                x = self._time_zoom_out(x)

        if np.random.rand() < self.prob:
            x = self._random_time_warp(x)

        # (2) 信号退化类（遮蔽、噪声）
        if np.random.rand() < self.prob:
            x = self._random_time_mask(x)

        if np.random.rand() < self.prob:
            x = self._add_gaussian_noise(x)

        # --- 3. 归一化 ---
        x = self._normalize(x)

        return x

    def _vertical_flip(self, x):
        """
        垂直翻转（幅度取负）
        模拟导联极性反转、电极接反等情况
        """
        return -x

    def _resample_to_fixed_fs(self, x):
        """
        将信号重采样到固定采样频率
        使用线性插值实现
        """
        seq_len = x.shape[0]
        # 计算新长度
        new_len = int(seq_len * self.resample_ratio)
        new_len = max(10, new_len)  # 防止太短

        # 原始时间轴和新时间轴
        t_old = np.linspace(0, seq_len - 1, seq_len)
        t_new = np.linspace(0, seq_len - 1, new_len)

        if x.ndim == 1:
            # 单导联
            x_resampled = np.interp(t_new, t_old, x)
        else:
            x_resampled = np.array([np.interp(t_new, t_old, x[:, c]) for c in range(x.shape[1])]).T

        return x_resampled

    def _add_gaussian_noise(self, x):
        """添加高斯噪声"""
        signal_std = np.std(x)
        noise = np.random.normal(0, self.noise_std * signal_std, x.shape)
        return x + noise

    def _random_time_warp(self, x):
        """随机时间扭曲"""
        sigma = self.time_warp_sigma
        seq_len = x.shape[0]
        dx = np.random.randn(seq_len) * sigma

        if seq_len > 5:
            dx = np.convolve(dx, np.hanning(5), 'same')

        t = np.arange(seq_len).astype(float) + dx
        t = np.clip(t, 0, seq_len - 1)

        if x.ndim == 1:
            x = np.interp(t, np.arange(seq_len), x)
        else:
            for c in range(x.shape[1]):
                x[:, c] = np.interp(t, np.arange(seq_len), x[:, c])
        return x

    def _random_time_mask(self, x):
        """随机时间遮蔽"""
        ratio = np.random.uniform(0.01, self.time_mask_ratio)
        duration = int(ratio * x.shape[0])
        duration = max(1, duration)
        start = np.random.randint(0, x.shape[0] - duration + 1)
        x[start:start + duration] = 0
        return x

    def _amplitude_scaling(self, x):
        """
        随机缩放信号幅度
        模拟不同导联增益、电极接触不良、个体差异等
        """
        # 随机生成缩放因子，例如 0.7 ~ 1.3 倍（±30%）
        scale = np.random.uniform(1 - self.amp_scale_range, 1 + self.amp_scale_range)

        return x * scale

    def _time_zoom_in(self, x):
        """
        时间拉伸（Zoom In）：从原信号中随机截取一段，再插值回原长度
        效果：信号被“拉长”，R-R 间隔变大，模拟心率变慢
        """
        seq_len = x.shape[0]

        # 随机选择截取比例：例如 70%~90% 的原始长度
        crop_ratio = np.random.uniform(1 - self.zoom_in_ratio_range, 1)
        crop_len = int(seq_len * crop_ratio)

        # 随机起点（保证能截取 crop_len 长度）
        start = np.random.randint(0, seq_len - crop_len + 1)

        if x.ndim == 1:
            x_cropped = x[start:start + crop_len]
        else:
            x_cropped = x[start:start + crop_len, :]

        # 新时间轴：覆盖裁剪后的时间范围
        t_crop = np.linspace(0, crop_len - 1, crop_len)
        t_old = np.linspace(0, crop_len - 1, seq_len)  # 目标长度为原长度

        # 插值回原始长度
        if x.ndim == 1:
            x_zoomed = np.interp(t_old, t_crop, x_cropped)
        else:
            x_zoomed = np.array([np.interp(t_old, t_crop, x_cropped[:, c]) for c in range(x_cropped.shape[1])]).T

        return x_zoomed

    def _time_zoom_out(self, x):
        """
        时间压缩（Zoom Out）：在首尾填充，再重采样到原长度
        效果：信号被“压缩”，R-R 间隔变小，模拟心率加快
        """
        seq_len = x.shape[0]

        # 随机扩展比例：例如 110%~140%
        expand_ratio = np.random.uniform(1, 1 + self.zoom_out_expand_range)
        expand_len = int(seq_len * expand_ratio)

        # 计算首尾填充量
        pad_len = expand_len - seq_len
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left

        # 使用边界值填充（模拟信号延续）
        if x.ndim == 1:
            x_padded = np.pad(x, (pad_left, pad_right), mode='constant')
        else:
            x_padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode='constant')

        # 原始时间轴（填充后）和目标时间轴（原始长度）
        t_padded = np.linspace(0, expand_len - 1, expand_len)
        t_old = np.linspace(0, expand_len - 1, seq_len)

        # 插值回原始长度
        if x.ndim == 1:
            x_zoomed = np.interp(t_old, t_padded, x_padded)
        else:
            x_zoomed = np.array([np.interp(t_old, t_padded, x_padded[:, c]) for c in range(x_padded.shape[1])]).T

        return x_zoomed

    def _normalize(self, x):
        """归一化"""
        if self.norm_type is None:
            return x

        if self.norm_type == 'zscore':
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            std[std < 1e-6] = 1e-6
            return (x - mean) / std

        elif self.norm_type == 'minmax':
            min_val = np.min(x, axis=0, keepdims=True)
            max_val = np.max(x, axis=0, keepdims=True)
            gap = (max_val - min_val)
            gap[gap < 1e-6] = 1e-6
            return (x - min_val) / gap

        else:
            raise ValueError(f"不支持的归一化方式: {self.norm_type}")