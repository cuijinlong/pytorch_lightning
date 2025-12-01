# app.py
import os
import io
import torch
import hydra
import logging
import tempfile
from typing import Dict, Any, List
from omegaconf import DictConfig, OmegaConf
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import rootutils
import torchaudio
import numpy as np

# 设置项目根目录
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=False)

from trainer.lhgnn.models.utils import RankedLogger
from trainer.lhgnn.models.tagging_module import TaggingModule
from trainer.lhgnn.dataset.spoken_datamodule import SpokenDataModule

# 初始化日志
log = RankedLogger(__name__, rank_zero_only=True)


class PredictionService:
    """预测服务类"""

    def __init__(self, config_path: str = "./configs", config_name: str = "predict.yaml"):
        """初始化预测服务

        Args:
            config_path: 配置文件路径
            config_name: 配置文件名
        """
        self.config_path = config_path
        self.config_name = config_name
        self.model = None
        self.datamodule = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型和数据模块
        self._initialize()

    def _initialize(self):
        """初始化模型和数据模块"""
        try:
            # 使用Hydra加载配置
            with hydra.initialize(config_path=self.config_path, version_base="1.3"):
                self.config = hydra.compose(config_name=self.config_name)

            log.info("Initializing data module...")
            # 实例化数据模块
            self.datamodule: SpokenDataModule = hydra.utils.instantiate(self.config.data)

            # 获取音频配置参数
            self.audio_conf = {
                'sr': self.datamodule.sr,
                'fmin': self.datamodule.fmin,
                'fmax': self.datamodule.fmax,
                'num_mels': self.datamodule.num_mels,
                'window_type': self.datamodule.window_type,
                'target_len': self.datamodule.target_len,
                'freqm': 0,  # 预测时不使用数据增强
                'timem': 0,  # 预测时不使用数据增强
                'norm_mean': self.datamodule.norm_mean,
                'norm_std': self.datamodule.norm_std,
                'mixup': 0.0  # 预测时不使用mixup
            }

            log.info("Initializing model...")
            # 实例化完整的TaggingModule（包含优化器等）
            self.model: TaggingModule = hydra.utils.instantiate(self.config.model)

            # 加载模型权重
            if self.config.get("model_weights"):
                log.info(f"Loading model weights from: {self.config.model_weights}")
                checkpoint = torch.load(self.config.model_weights, map_location="cpu")

                # 处理不同的检查点格式
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    # 处理可能的prefix
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("net."):
                            new_state_dict[k[4:]] = v
                        else:
                            new_state_dict[k] = v
                    self.model.net.load_state_dict(new_state_dict)
                else:
                    self.model.net.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            log.info(f"Model initialized on device: {self.device}")
            log.info(f"Audio config: {self.audio_conf}")

        except Exception as e:
            log.error(f"Failed to initialize prediction service: {e}")
            raise

    def wav_to_fbank(self, filename: str):
        """将音频文件转换为fbank特征 - 与训练时保持一致

        Args:
            filename: 音频文件路径

        Returns:
            fbank特征张量
        """
        try:
            # 单音频处理 - 与SpokenDataset中的逻辑保持一致
            waveform, sr = torchaudio.load(filename)

            # 多声道转单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 重采样
            if sr != self.audio_conf['sr']:
                transform = torchaudio.transforms.Resample(sr, self.audio_conf['sr'])
                waveform = transform(waveform)

            # 归一化
            waveform = waveform - waveform.mean()

            # 提取fbank特征 - 使用与训练时相同的参数
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=self.audio_conf['sr'],
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.audio_conf['num_mels'],
                dither=0.0,
                frame_shift=10
            )

            # 调整长度到target_len
            n_frames = fbank.shape[0]
            if n_frames < self.audio_conf['target_len']:
                # 填充
                p = self.audio_conf['target_len'] - n_frames
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif n_frames > self.audio_conf['target_len']:
                # 裁剪
                fbank = fbank[:self.audio_conf['target_len'], :]

            # 归一化 - 使用与训练时相同的参数
            fbank = (fbank - self.audio_conf['norm_mean']) / self.audio_conf['norm_std']

            return fbank

        except Exception as e:
            log.error(f"Error in wav_to_fbank for {filename}: {e}")
            raise

    def preprocess_file(self, file_path: str) -> torch.Tensor:
        """预处理上传的文件 - 与训练时保持一致

        Args:
            file_path: 文件路径

        Returns:
            预处理后的张量，形状为 [1, target_len, num_mels]
        """
        try:
            log.info(f"Preprocessing file: {file_path}")

            # 使用与训练时相同的fbank提取方法
            fbank = self.wav_to_fbank(file_path)

            # 确保形状正确 [target_len, num_mels]
            log.info(f"Fbank shape: {fbank.shape}")

            # 添加批次维度 [1, target_len, num_mels]
            fbank = fbank.unsqueeze(0)

            log.info(f"Final preprocessed shape: {fbank.shape}")
            return fbank

        except Exception as e:
            log.error(f"Error preprocessing file {file_path}: {e}")
            log.error(f"Error details: {type(e).__name__}: {str(e)}")
            raise

    def predict(self, file_path: str) -> Dict[str, Any]:
        """对单个文件进行预测

        Args:
            file_path: 文件路径

        Returns:
            预测结果字典
        """
        try:
            # 预处理文件
            input_tensor = self.preprocess_file(file_path)
            log.info(f"Input tensor shape for prediction: {input_tensor.shape}")

            input_tensor = input_tensor.to(self.device)

            # 进行预测
            with torch.no_grad():
                try:
                    # 使用模型的forward方法
                    preds = self.model.net(input_tensor)

                    log.info(f"Model output shape: {preds.shape}")

                    # 处理模型输出
                    predictions = self._process_output(preds)

                    return {
                        "success": True,
                        "file": os.path.basename(file_path),
                        "input_shape": list(input_tensor.shape),
                        "predictions": predictions,
                        "message": "Prediction completed successfully"
                    }

                except Exception as e:
                    log.error(f"Error during model forward pass: {e}")
                    log.error(f"Input tensor shape: {input_tensor.shape}")
                    log.error(f"Error details: {type(e).__name__}: {str(e)}")

                    return {
                        "success": False,
                        "file": os.path.basename(file_path),
                        "error": str(e),
                        "message": "Model prediction failed"
                    }

        except Exception as e:
            log.error(f"Error during prediction for {file_path}: {e}")
            return {
                "success": False,
                "file": os.path.basename(file_path),
                "error": str(e),
                "message": "Prediction failed"
            }

    def _process_output(self, output) -> Dict[str, Any]:
        """处理模型输出"""
        predictions = {}

        # 根据模型输出类型处理
        if isinstance(output, torch.Tensor):
            # 应用sigmoid或softmax取决于损失函数
            if hasattr(self.model, 'criterion'):
                if isinstance(self.model.criterion, torch.nn.BCEWithLogitsLoss):
                    # 对于BCEWithLogitsLoss，需要应用sigmoid
                    probabilities = torch.sigmoid(output)
                elif isinstance(self.model.criterion, torch.nn.CrossEntropyLoss):
                    # 对于CrossEntropyLoss，需要应用softmax
                    probabilities = torch.softmax(output, dim=-1)
                else:
                    probabilities = torch.sigmoid(output)  # 默认
            else:
                probabilities = torch.sigmoid(output)  # 默认

            # 获取预测类别和置信度
            if probabilities.dim() == 2:
                if probabilities.shape[1] > 1:  # 多分类
                    confidence, predicted_class = torch.max(probabilities, 1)
                    predictions = {
                        "predicted_class": int(predicted_class.item()),
                        "confidence": float(confidence.item()),
                        "all_probabilities": probabilities.cpu().numpy().tolist()[0]
                    }
                else:  # 二分类
                    confidence = probabilities.squeeze()
                    predicted_class = 1 if confidence > 0.5 else 0
                    predictions = {
                        "predicted_class": int(predicted_class),
                        "confidence": float(confidence.item()),
                        "all_probabilities": [1 - confidence.item(), confidence.item()]
                    }

        return predictions

    def batch_predict(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量预测

        Args:
            file_paths: 文件路径列表

        Returns:
            批量预测结果
        """
        results = []
        for file_path in file_paths:
            result = self.predict(file_path)
            results.append(result)

        return {
            "success": True,
            "total_files": len(file_paths),
            "results": results
        }


# 创建Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 文件大小限制

# 初始化预测服务
prediction_service = None


def get_prediction_service():
    """获取预测服务实例"""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service


# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        service = get_prediction_service()
        return jsonify({
            "status": "healthy",
            "message": "Prediction service is running",
            "device": str(service.device),
            "model_type": service.model.__class__.__name__
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """单个文件预测端点"""
    try:
        # 检查文件是否在请求中
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400

        file = request.files['file']

        # 检查文件名
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        if file and allowed_file(file.filename):
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                # 进行预测
                service = get_prediction_service()
                result = service.predict(temp_path)

                # 清理临时文件
                os.unlink(temp_path)

                return jsonify(result)

            except Exception as e:
                # 确保清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
        else:
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

    except Exception as e:
        log.error(f"Error in predict endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict_endpoint():
    """批量预测端点"""
    try:
        if 'files' not in request.files:
            return jsonify({
                "success": False,
                "error": "No files provided"
            }), 400

        files = request.files.getlist('files')

        if len(files) == 0:
            return jsonify({
                "success": False,
                "error": "No files selected"
            }), 400

        # 保存临时文件
        temp_paths = []
        valid_files = []

        for file in files:
            if file and allowed_file(file.filename):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                file.save(temp_file.name)
                temp_paths.append(temp_file.name)
                valid_files.append(temp_file.name)
            else:
                log.warning(f"Skipping invalid file: {file.filename}")

        if not valid_files:
            return jsonify({
                "success": False,
                "error": "No valid files provided"
            }), 400

        try:
            # 进行批量预测
            service = get_prediction_service()
            result = service.batch_predict(valid_files)

            # 清理临时文件
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            return jsonify(result)

        except Exception as e:
            # 确保清理临时文件
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            raise e

    except Exception as e:
        log.error(f"Error in batch predict endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息端点"""
    try:
        service = get_prediction_service()

        info = {
            "model_type": service.model.__class__.__name__,
            "net_type": service.model.net.__class__.__name__,
            "device": str(service.device),
            "audio_config": service.audio_conf,
            "num_classes": service.model.net.num_classes
        }

        return jsonify({
            "success": True,
            "model_info": info
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    # 初始化预测服务
    try:
        prediction_service = PredictionService()
        log.info("Starting Flask prediction service...")

        # 启动Flask应用
        app.run(
            host=os.getenv('HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', 5000)),
            debug=os.getenv('DEBUG', 'False').lower() == 'true'
        )

    except Exception as e:
        log.error(f"Failed to start prediction service: {e}")