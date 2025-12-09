import json
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from data.ecg_augment import ECGAugment
from scipy.signal import stft
from skimage.transform import resize
import matplotlib

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
        if 'cpsc2021' in self.path1:
            fs = 200
        elif 'bihaf' in self.path1:
            fs = 250
        signal = seg[:, 0].astype(np.float32)
        # 计算 STFT
        win_sec = 1
        nperseg = int(win_sec * fs)
        noverlap = nperseg // 2
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)  # (freq_bins, time_frames)

        # 转 dB
        mag = 20 * np.log10(mag + 1e-8)
        mag = np.clip(mag, -80, mag.max())

        # 归一化到 [0, 1]
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

        # 调整尺寸到 (256, 256)
        mag_resized = resize(mag, (256,256), anti_aliasing=True, preserve_range=True)

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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        img = self._generate_image(x)
        img = torch.from_numpy(img).permute(2, 0, 1).float() 
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x, dtype=torch.float32)[:, 0:1]  # 取第一导联
        y = torch.tensor(y, dtype=torch.long)
        return x, img, y