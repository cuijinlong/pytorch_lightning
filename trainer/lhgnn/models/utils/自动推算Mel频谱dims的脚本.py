import glob
import numpy as np
import librosa

DATA_DIR = "/Users/cuijinlong/Documents/workspace_py/pytorch_lightning/tmp"

def infer_sample_rate(data_dir):
    wavs = glob.glob(f"{data_dir}/**/*.wav", recursive=True)
    wavs = wavs[:10]
    sr_list = [librosa.load(w, sr=None)[1] for w in wavs]
    return max(set(sr_list), key=sr_list.count)

def infer_target_len(data_dir, sample_rate):
    wavs = glob.glob(f"{data_dir}/**/*.wav", recursive=True)
    wavs = wavs[:10]
    durations = [len(librosa.load(w, sr=None)[0]) / sample_rate for w in wavs]
    avg_sec = np.mean(durations)

    if avg_sec <= 1.0:
        target_len = 256
    elif avg_sec <= 2.0:
        target_len = 512
    elif avg_sec <= 4.0:
        target_len = 1024
    else:
        target_len = 2048
    return target_len, avg_sec

def infer_num_mels(sample_rate):
    if sample_rate <= 16000:
        return 64
    elif sample_rate <= 32000:
        return 80
    else:
        return 128

def infer_fft_params(sample_rate):
    n_fft = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    n_fft = 2 ** int(np.ceil(np.log2(n_fft)))
    return n_fft, hop_length

def compute_norm_mean_std(data_dir, sample_rate, n_mels, target_len):
    wavs = glob.glob(f"{data_dir}/**/*.wav", recursive=True)
    wavs = wavs[:10]
    all_fbanks = []

    for w in wavs:
        y, sr = librosa.load(w, sr=sample_rate)
        # 计算 Mel 频谱（librosa >=0.10 版本需要用关键字参数）
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=int(0.025*sr),
            hop_length=int(0.010*sr),
            window='hann',
            fmin=20,
            fmax=sr//2,
            power=2.0
        )
        # 调整时间长度
        if mel_spec.shape[1] < target_len:
            pad_width = target_len - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :target_len]

        all_fbanks.append(mel_spec)

    all_fbanks = np.concatenate(all_fbanks, axis=1)
    mean = np.mean(all_fbanks)
    std = np.std(all_fbanks)
    return mean, std

# ---------------------- 推算 ----------------------
sr = infer_sample_rate(DATA_DIR)
target_len, avg_sec = infer_target_len(DATA_DIR, sr)
num_mels = infer_num_mels(sr)
n_fft, hop_length = infer_fft_params(sr)
norm_mean, norm_std = compute_norm_mean_std(DATA_DIR, sr, num_mels, target_len)

# freqm / timem 可按经验值设置
freqm = max(8, num_mels // 2)  # 经验值
timem = max(40, target_len // 1)  # 经验值

print("推算结果:")
print(f"sr: {sr}")
print(f"target_len: {target_len} (平均时长: {avg_sec:.2f}s)")
print(f"num_mels: {num_mels}")
print(f"n_fft: {n_fft}, hop_length: {hop_length}")
print(f"freqm: {freqm}, timem: {timem}")
print(f"norm_mean: {norm_mean:.3f}, norm_std: {norm_std:.3f}")
