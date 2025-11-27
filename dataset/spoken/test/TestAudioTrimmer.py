import unittest
import numpy as np
import scipy.io.wavfile
import os
from dataset.spoken.trimmer import trim_silence, trim_silence_file, split_multiple_recordings, split_multiple_recordings_file


class TestAudioTrimmer(unittest.TestCase):

    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录存放测试文件
        self.test_dir = '/opt/datasets/spoken/optional/wav/'

        # 创建测试音频数据
        self.sample_rate = 8000

        # 创建连续的声音段（1秒）
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)

        # 生成连续的声音信号（正弦波）
        freq = 440  # A4音
        self.test_audio = 1000 * np.sin(2 * np.pi * freq * t)
        self.test_audio = self.test_audio.astype(np.int16)

    """ 测试静音修剪功能 """
    def test_trim_silence(self):
        # 添加前后静音
        silence_prefix = np.zeros(1000, dtype=np.int16)
        silence_suffix = np.zeros(1000, dtype=np.int16)
        audio_with_silence = np.concatenate([silence_prefix, self.test_audio, silence_suffix])

        trimmed = trim_silence(audio_with_silence, noise_threshold=150)

        # 检查修剪后的长度应该小于原始带静音的长度
        self.assertLess(len(trimmed), len(audio_with_silence))

        # 检查修剪后的音频应该大致等于原始测试音频
        self.assertAlmostEqual(len(trimmed), len(self.test_audio), delta=100)

    """ 测试多录音分割功能 """
    def test_split_multiple_recordings(self):
        """测试多录音分割功能"""

        print("\n=== 测试多录音分割 ===")

        # 创建明确的测试音频结构：声音 + 静音 + 声音 + 静音 + 声音
        silence_duration = 0.5  # 0.5秒静音
        silence_frames = int(self.sample_rate * silence_duration)
        silence_segment = np.zeros(silence_frames, dtype=np.int16)

        # 构建测试音频
        multi_audio = np.concatenate([
            self.test_audio,    # 第一个声音片段 (1秒)
            silence_segment,    # 静音间隔 (0.5秒)
            self.test_audio,    # 第二个声音片段 (1秒)
            silence_segment,    # 静音间隔 (0.5秒)
            self.test_audio     # 第三个声音片段 (1秒)
        ])

        print(f"测试音频总长度: {len(multi_audio)} 帧 ({len(multi_audio)/self.sample_rate:.2f} 秒)")
        print(f"期望分割: 3个片段，每个约 {len(self.test_audio)} 帧")

        splits = split_multiple_recordings(
            multi_audio,
            min_silence_duration=0.3,  # 最小静音时长0.3秒
            noise_threshold=150,
            sample_rate_hz=self.sample_rate
        )

        print(f"实际分割: {len(splits)} 个片段")

        # 应该分割成3个片段
        self.assertEqual(len(splits), 3, f"期望3个片段，但得到了{len(splits)}个")

        # 每个片段的长度应该大致等于原始测试音频段
        for i, split in enumerate(splits):
            expected_length = len(self.test_audio)
            actual_length = len(split)
            print(f"片段 {i}: {actual_length} 帧 (期望: {expected_length} 帧)")
            self.assertAlmostEqual(actual_length, expected_length, delta=500,
                                 msg=f"片段 {i} 长度 {actual_length} 不接近期望值 {expected_length}")

    def test_trim_silence_file(self):
        """测试文件静音修剪功能"""

        # 创建测试文件
        test_file = os.path.join(self.test_dir, "0_george_0.wav")
        scipy.io.wavfile.write(test_file, self.sample_rate, self.test_audio)

        # 获取原始文件信息
        original_stats = os.stat(test_file)

        # 执行修剪（虽然这个测试音频没函有静音，但数应该能正常运行）
        trim_silence_file(test_file, noise_threshold=150)

        # 检查文件仍然存在
        self.assertTrue(os.path.exists(test_file))

        # 重新读取验证文件仍然有效
        rate, audio = scipy.io.wavfile.read(test_file)
        self.assertEqual(rate, self.sample_rate)
        self.assertTrue(len(audio) > 0)

    def test_split_multiple_recordings_file(self):
        """测试文件分割功能"""

        # 创建包含多个片段的测试文件
        multi_audio = np.concatenate([
            self.test_audio,  # 第一个片段
            np.zeros(4000),  # 静音间隔（0.5秒）
            self.test_audio,  # 第二个片段
        ]).astype(np.int16)

        test_file = os.path.join(self.test_dir, "0_george_10.wav")
        scipy.io.wavfile.write(test_file, self.sample_rate, multi_audio)

        # 执行分割
        split_multiple_recordings_file(
            test_file,
            min_silence_duration=0.4,
            noise_threshold=150
        )

        # 检查分割后的文件
        file1 = os.path.join(self.test_dir, "0_george_0.wav")
        file2 = os.path.join(self.test_dir, "0_george_1.wav")

        self.assertTrue(os.path.exists(file1))
        self.assertTrue(os.path.exists(file2))

        # 验证分割文件的内容
        rate1, audio1 = scipy.io.wavfile.read(file1)
        rate2, audio2 = scipy.io.wavfile.read(file2)

        self.assertEqual(rate1, self.sample_rate)
        self.assertEqual(rate2, self.sample_rate)
        self.assertAlmostEqual(len(audio1), len(self.test_audio), delta=100)
        self.assertAlmostEqual(len(audio2), len(self.test_audio), delta=100)

    def test_invalid_file_path(self):
        """测试无效文件路径处理"""

        # 创建无效文件路径
        invalid_file = os.path.join(self.test_dir, "invalid.name.wav")
        scipy.io.wavfile.write(invalid_file, self.sample_rate, self.test_audio)

        # 应该抛出异常
        with self.assertRaises(Exception):
            split_multiple_recordings_file(invalid_file)

    # def tearDown(self):
    #     """测试后的清理工作"""
    #     # 删除测试目录及其内容
    #     import shutil
    #     shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)