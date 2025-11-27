# dataset/lhgnn/spoken_dataset.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import torchaudio
import os

class SpokenDataset(Dataset):

    def __init__(self, data_list, audio_conf, mode='train'):
        super().__init__()
        self.data = data_list
        self.audio_conf = audio_conf
        self.mode = mode

        # 获取所有可能的数字标签
        all_labels = set(item['labels'] for item in data_list)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        # 音频配置参数
        self.mixup = audio_conf['mixup']
        self.num_mels = audio_conf['num_mels']
        self.fmin = audio_conf['fmin']
        self.fmax = audio_conf['fmax']
        self.sr = audio_conf['sr']
        self.window_type = audio_conf['window_type']
        self.target_len = audio_conf['target_len']
        self.freqm = audio_conf['freqm']
        self.timem = audio_conf['timem']
        self.norm_mean = audio_conf['norm_mean']
        self.norm_std = audio_conf['norm_std']

        print(f"数据集模式: {mode}, 类别数: {self.num_classes}")

    def __len__(self):
        return len(self.data)

    def wav_to_fbank(self, filename, filename2=None):
        """将音频文件转换为fbank特征"""

        if filename2 is None:
            # 单音频处理
            waveform, sr = torchaudio.load(filename)

            # 重采样
            if sr != self.sr:
                transform = torchaudio.transforms.Resample(sr, self.sr)
                waveform = transform(waveform)

            # 归一化
            waveform = waveform - waveform.mean()

        else:
            # Mixup处理
            waveform1, sr1 = torchaudio.load(filename)
            waveform2, sr2 = torchaudio.load(filename2)

            # 重采样
            if sr1 != self.sr:
                transform1 = torchaudio.transforms.Resample(sr1, self.sr)
                waveform1 = transform1(waveform1)
            if sr2 != self.sr:
                transform2 = torchaudio.transforms.Resample(sr2, self.sr)
                waveform2 = transform2(waveform2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            # 长度对齐
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # 填充
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, :waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # 裁剪
                    waveform2 = waveform2[:, :waveform1.shape[1]]

            # Mixup
            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        # 提取fbank特征
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.num_mels,
            dither=0.0,
            frame_shift=10
        )

        # 调整长度
        n_frames = fbank.shape[0]
        if n_frames < self.target_len:
            # 填充
            p = self.target_len - n_frames
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif n_frames > self.target_len:
            # 裁剪
            fbank = fbank[:self.target_len, :]

        if filename2 is None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, idx):
        if self.mode == 'train' and random.random() < self.mixup:
            # 训练模式且使用mixup
            mixup_idx = random.randint(0, len(self.data) - 1)

            wav_file1 = self.data[idx]['wav']
            wav_file2 = self.data[mixup_idx]['wav']

            fbank, mix_lambda = self.wav_to_fbank(wav_file1, wav_file2)

            # 处理标签
            label1 = self.label_to_idx[self.data[idx]['labels']]
            label2 = self.label_to_idx[self.data[mixup_idx]['labels']]

            # 创建one-hot标签
            label_onehot = torch.zeros(self.num_classes)
            label_onehot[label1] = mix_lambda
            label_onehot[label2] += (1 - mix_lambda)

            # 数据增强：频谱掩蔽
            if self.freqm > 0:
                freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
                fbank = torch.transpose(fbank, 0, 1)
                fbank = fbank.unsqueeze(0)
                fbank = freqm(fbank)
                fbank = fbank.squeeze(0)
                fbank = torch.transpose(fbank, 0, 1)

            if self.timem > 0:
                timem = torchaudio.transforms.TimeMasking(self.timem)
                fbank = torch.transpose(fbank, 0, 1)
                fbank = fbank.unsqueeze(0)
                fbank = timem(fbank)
                fbank = fbank.squeeze(0)
                fbank = torch.transpose(fbank, 0, 1)

        else:
            # 正常模式
            wav_file = self.data[idx]['wav']
            fbank, _ = self.wav_to_fbank(wav_file)

            label = self.label_to_idx[self.data[idx]['labels']]
            label_onehot = torch.zeros(self.num_classes)
            label_onehot[label] = 1.0

            # 训练模式下的数据增强
            if self.mode == 'train':
                if self.freqm > 0:
                    freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
                    fbank = torch.transpose(fbank, 0, 1)
                    fbank = fbank.unsqueeze(0)
                    fbank = freqm(fbank)
                    fbank = fbank.squeeze(0)
                    fbank = torch.transpose(fbank, 0, 1)

                if self.timem > 0:
                    timem = torchaudio.transforms.TimeMasking(self.timem)
                    fbank = torch.transpose(fbank, 0, 1)
                    fbank = fbank.unsqueeze(0)
                    fbank = timem(fbank)
                    fbank = fbank.squeeze(0)
                    fbank = torch.transpose(fbank, 0, 1)

        # 归一化
        fbank = (fbank - self.norm_mean) / self.norm_std

        return fbank, label_onehot

    def get_class_names(self):
        """获取类别名称"""
        return [self.idx_to_label[i] for i in range(self.num_classes)]


