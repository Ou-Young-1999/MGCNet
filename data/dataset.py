import json
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from data.ecg_augment import ECGAugment
from scipy.signal import stft
from skimage.transform import resize
import pywt
import matplotlib
from scipy import signal


class AFDataset(Dataset):
    def __init__(self, path1, path2, split='train', fold=1, transform=None, oversample=False, ga_test=False):
        self.path1 = path1
        self.path2 = path2
        json_path = os.path.join(path1,'fold_'+str(fold)+'.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if ga_test == True:
            self.file_names = data['train'] + data['val']
        else:
            self.file_names = data[split]
        self.transform = transform
        self.oversample = oversample and (split == 'train')  # 只在训练集开启

        # 加载数据
        if 'cpsc2021' in path1:
            self.X, self.y = self._load_all_segments_cpsc2021()
        else:
            self.X, self.y = self._load_all_segments()

        # 转换标签为数字
        self.y = np.array([1 if label == 'AFIB' else 0 for label in self.y])

        # 如果是训练集且启用过采样，则进行少数类过采样
        if self.oversample:
            self.X, self.y = self._oversample_minority_class()

    def _load_all_segments_cpsc2021(self):
        all_X = []
        all_y = []
        json_path = os.path.join(self.path1,'case_to_files.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for fname in self.file_names:
            for fcase in json_data[fname]:
                npz_path = os.path.join(self.path2, fcase)
                data = np.load(npz_path)
                X = data['segments']
                y = data['labels']
                if X.ndim < 3:
                    continue
                all_X.append(X)
                all_y.append(y)
        X_full = np.concatenate(all_X, axis=0)
        y_full = np.concatenate(all_y, axis=0)
        return X_full, y_full

    def _load_all_segments(self):
        all_X = []
        all_y = []
        for fname in self.file_names:
            npz_path = os.path.join(self.path2, fname+'.npz')
            data = np.load(npz_path)
            X = data['segments']
            y = data['labels']
            if X.ndim < 3:
                continue
            all_X.append(X)
            all_y.append(y)
        X_full = np.concatenate(all_X, axis=0)
        y_full = np.concatenate(all_y, axis=0)

        return X_full, y_full

    def _generate_image(self, seg):

        fs = 250
        signal = seg[:, 0].astype(np.float32)
        # 计算 STFT
        win_sec = 1
        nperseg = int(win_sec * fs)
        noverlap = nperseg // 2
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)  # (freq_bins, time_frames)

        # 裁剪频率范围（0–40 Hz）
        f_max_plot = 40.0
        freq_mask = f <= f_max_plot
        mag = mag[freq_mask, :]

        # 转 dB
        mag = 20 * np.log10(mag + 1e-8)
        mag = np.clip(mag, -80, mag.max())

        # 归一化到 [0, 1]
        mag = (mag + 80) / 100.0 

        # 调整尺寸到 (128, 128)
        mag_resized = resize(mag, (128,128), anti_aliasing=True, preserve_range=True)

        # 使用 matplotlib colormap 转换为 RGB
        cmap = matplotlib.colormaps.get_cmap('jet')
        rgb = cmap(mag_resized)[:, :, :3]  # 去掉 alpha 通道

        return rgb

    def _oversample_minority_class(self):
        """
        对少数类（AFIB, label=1）进行随机过采样，使其数量等于多数类
        """
        X = self.X
        y = self.y

        # 分离正负样本
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]

        X_0, y_0 = X[idx_0], y[idx_0]
        X_1, y_1 = X[idx_1], y[idx_1]

        n_majority = len(X_0)
        n_minority = len(X_1)

        if n_minority >= n_majority:
            print(f"[Oversample] No need: minority ({n_minority}) >= majority ({n_majority})")
            return X, y

        # 随机重复采样少数类（允许重复）
        np.random.seed(42)  # 保证可复现
        oversample_idx = np.random.choice(idx_1, size=n_majority - n_minority, replace=True)
        X_1_oversampled = X[oversample_idx]
        y_1_oversampled = y[oversample_idx]

        # 合并
        X_balanced = np.concatenate([X_0, X_1, X_1_oversampled], axis=0)
        y_balanced = np.concatenate([y_0, y_1, y_1_oversampled], axis=0)

        print(f"[Oversample] Done: {len(X_0)} normal + {len(X_1) + len(X_1_oversampled)} AF (original: {len(X_1)})")

        return X_balanced, y_balanced

    # ----------------------------
    # 干扰生成函数
    # ----------------------------

    def _add_baseline_wander(self, x, fs=250, amplitude_ratio=0.7):
        """
        Add baseline wander to multi-lead ECG signal of shape (L, C).
        
        Args:
            x: ECG signal, shape (L,) or (L, C)
            fs: Sampling frequency in Hz
            amplitude_ratio: Ratio of wander amplitude to global peak-to-peak
        
        Returns:
            Noisy signal with same shape as input
        """
        original_shape = x.shape

        # Handle single-lead case: (L,) → (L, 1)
        if x.ndim == 1:
            x = x[:, np.newaxis]  # (L, 1)
            single_lead = True
        else:
            assert x.ndim == 2, "Input must be (L,) or (L, C)"
            single_lead = False

        L, C = x.shape

        # Compute global peak-to-peak across all leads and time
        global_ptp = np.max(x) - np.min(x)
        amp = global_ptp * amplitude_ratio

        # Randomly sample frequency in physiological range
        # freq = np.random.uniform(0.1, 0.5)  # Hz
        freq = 0.1
        phase = np.random.uniform(0, 2 * np.pi)

        # Time vector
        t = np.arange(L) / fs  # shape: (L,)

        # Generate baseline wander (same for all leads)
        wander = amp * np.sin(2 * np.pi * freq * t + phase)  # shape: (L,)

        # Broadcast to all channels: (L,) + (L, C) → (L, C)
        x_noisy = x + wander[:, np.newaxis]  # explicitly make (L, 1) for broadcasting

        # Restore original shape
        if single_lead:
            x_noisy = x_noisy.squeeze(axis=1)  # back to (L,)

        return x_noisy

    def _add_powerline_noise(self, x, fs=250, f0=50, snr_db=20, harmonics=2):
        """
        Add powerline noise to ECG signal of shape (L,) or (L, C).
        All channels share the same interference (physiologically realistic).
        """
        original_shape = x.shape

        # Ensure at least 2D for unified processing: (L, C)
        if x.ndim == 1:
            x = x[:, np.newaxis]  # (L, 1)
            single_lead = True
        else:
            assert x.ndim == 2, "Input must be (L,) or (L, C)"
            single_lead = False

        L, C = x.shape
        t = np.arange(L) / fs  # (L,)

        # Generate noise (same for all leads)
        noise = np.zeros(L)
        for h in range(1, harmonics + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            noise += (1.0 / h) * np.sin(2 * np.pi * h * f0 * t + phase)

        # Compute powers
        signal_power = np.mean(x ** 2)      # scalar, over all elements
        noise_power = np.mean(noise ** 2)   # scalar

        if noise_power == 0:
            return x.reshape(original_shape)

        # Scale to achieve target SNR
        desired_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = np.sqrt(desired_noise_power / noise_power)

        # Add noise (broadcast to all channels)
        x_noisy = x + scale * noise[:, np.newaxis]  # (L, C) + (L, 1)

        # Restore original shape
        if single_lead:
            x_noisy = x_noisy.squeeze(axis=1)
        return x_noisy

    def _add_emg_noise(self, x, snr_db=0.5, bandwidth=(20, 500), fs=250):
        original_shape = x.shape
        if x.ndim == 1:
            x = x[:, np.newaxis]
            single_lead = True
        else:
            assert x.ndim == 2, "Expected (L,) or (L, C)"
            single_lead = False

        L, C = x.shape
        signal_power = np.mean(x ** 2)

        # ✅ 确保 bandwidth 不超过 Nyquist 频率
        nyq = 0.5 * fs
        low, high = bandwidth
        low = max(low, 1)          # 至少 >0
        high = min(high, nyq * 0.999)  # 必须 < nyq（留一点余量）
        
        if low >= high:
            # 如果带宽无效，退化为白噪声或直接返回
            noise = np.random.randn(L, C)
        else:
            noise = np.random.randn(L, C)
            b, a = signal.butter(4, [low / nyq, high / nyq], btype='band')
            for c in range(C):
                noise[:, c] = signal.filtfilt(b, a, noise[:, c])

        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return x.reshape(original_shape)

        scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
        x_noisy = x + scale * noise

        if single_lead:
            x_noisy = x_noisy.squeeze(axis=1)
        return x_noisy

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        # x = self._add_emg_noise(x)
        img = self._generate_image(x)
        img = torch.from_numpy(img).permute(2, 0, 1).float() 
        x = torch.tensor(x, dtype=torch.float32)[:, 0:1]  # 取第一导联
        
        y = torch.tensor(y, dtype=torch.long)
        return x, img, y
