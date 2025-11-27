import scipy.io.wavfile
import numpy as np


def split_multiple_recordings(audio, min_silence_duration=0.25, noise_threshold=150, sample_rate_hz=8e3):
    """
    将音频数据按静音段分割成多个音频数组

    :param audio: 音频数据的numpy数组
    :param min_silence_duration: 分割所需的最小静音时长（秒）
    :param noise_threshold: 被认为是静音的噪声阈值
    :param sample_rate_hz: 音频采样率
    :return: 分割后的音频数组列表
    """
    # 计算最小静音帧数
    min_silence_frames = int(sample_rate_hz * min_silence_duration)
    silence_zones = []  # 存储静音区域的列表，每个元素为(start, end)元组

    zone_start = None  # 静音区域开始索引

    # 改进的静音检测逻辑
    for idx, point in enumerate(audio):
        # 检测静音开始
        if abs(point) < noise_threshold and zone_start is None:
            zone_start = idx

        # 检测静音结束：当遇到声音且之前已经开始静音检测
        elif abs(point) >= noise_threshold and zone_start is not None:
            zone_end = idx

            # 检查静音段是否足够长
            if (zone_end - zone_start) >= min_silence_frames:
                silence_zones.append((zone_start, zone_end))

            zone_start = None

    # 处理音频末尾的静音
    if zone_start is not None and (len(audio) - zone_start) >= min_silence_frames:
        silence_zones.append((zone_start, len(audio)))

    print(f"检测到 {len(silence_zones)} 个静音区域")

    # 根据静音区域分割录音
    split_recordings = []

    if not silence_zones:
        # 没有检测到静音区域，返回整个音频
        return [audio]

    # 第一个片段：从开始到第一个静音区域
    if silence_zones[0][0] > 0:
        split_recordings.append(audio[:silence_zones[0][0]])
        print(f"第一个片段: 0 到 {silence_zones[0][0]}")

    # 中间片段：静音区域之间的部分
    for i in range(len(silence_zones) - 1):
        start = silence_zones[i][1]  # 当前静音结束
        end = silence_zones[i + 1][0]  # 下一个静音开始

        if end > start:  # 确保有内容
            split_recordings.append(audio[start:end])
            print(f"片段 {i + 1}: {start} 到 {end}")

    # 最后一个片段：从最后一个静音区域结束到音频末尾
    if silence_zones[-1][1] < len(audio):
        split_recordings.append(audio[silence_zones[-1][1]:])
        print(f"最后一个片段: {silence_zones[-1][1]} 到 {len(audio)}")

    print(f"总共分割成 {len(split_recordings)} 个录音片段")
    return split_recordings


def trim_silence(audio, noise_threshold=150):
    """
    移除音频数据开头和结尾的静音部分

    :param audio: 音频数据的numpy数组
    :param noise_threshold: 被认为是静音的噪声阈值
    :return: 修剪后的numpy数组
    """
    start = None  # 音频实际开始位置
    end = None  # 音频实际结束位置

    # 正向查找音频开始位置
    for idx, point in enumerate(audio):
        if abs(point) > noise_threshold:
            start = idx
            break

    # 如果全是静音，返回空数组
    if start is None:
        return audio[0:0]

    # 反向查找音频结束位置
    for idx, point in enumerate(audio[::-1]):
        if abs(point) > noise_threshold:
            end = len(audio) - idx
            break

    return audio[start:end]


def trim_silence_file(file_path, noise_threshold=150):
    """
    修剪音频文件的静音部分并覆盖原文件

    :param file_path: 要修剪的音频文件路径
    :param noise_threshold: 被认为是静音的噪声阈值
    :return: None
    """
    rate, audio = scipy.io.wavfile.read(file_path)
    trimmed_audio = trim_silence(audio, noise_threshold=noise_threshold)
    scipy.io.wavfile.write(file_path, rate, trimmed_audio)


def split_multiple_recordings_file(file_path, min_silence_duration=0.25, noise_threshold=150):
    """
    将音频文件按静音段分割成多个文件

    适用于一次录制多个发音的情况，可自动分割成单独的文件

    :param file_path: 要分割的wav文件路径
    :param min_silence_duration: 分割所需的最小静音时长（秒）
    :param noise_threshold: 被认为是静音的噪声阈值
    :return: None
    """
    rate, audio = scipy.io.wavfile.read(file_path)
    split_recordings = split_multiple_recordings(audio, min_silence_duration=min_silence_duration,
                                                 noise_threshold=noise_threshold, sample_rate_hz=rate)

    # 验证文件路径格式
    if file_path.count('.') != 1:
        raise Exception('File_path must contain exactly one period, usually in extension. IE: /home/test.wav')

    # 为每个分割片段创建新文件
    for idx, recording in enumerate(split_recordings):
        new_file_path = file_path.split('.')[0] + '_' + str(idx) + ".wav"
        scipy.io.wavfile.write(new_file_path, rate, recording)
        print(f"已创建文件: {new_file_path}")