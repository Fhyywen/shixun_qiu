import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelUtils:
    """模型工具类"""

    @staticmethod
    def load_model_safely(model_path: str, model_class=AutoModel, **kwargs):
        """安全加载模型"""
        try:
            model = model_class.from_pretrained(model_path, **kwargs)
            logger.info(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    @staticmethod
    def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
        """计算模型大小"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2

        return {
            "total_mb": size_all_mb,
            "parameters_mb": param_size / 1024 ** 2,
            "buffers_mb": buffer_size / 1024 ** 2,
            "num_parameters": sum(p.numel() for p in model.parameters())
        }

    @staticmethod
    def optimize_model_for_inference(model: torch.nn.Module, use_half_precision: bool = True):
        """优化模型用于推理"""
        model.eval()

        if use_half_precision and not isinstance(model, torch.nn.DataParallel):
            try:
                model.half()
                logger.info("Converted model to half precision")
            except Exception as e:
                logger.warning(f"Could not convert model to half precision: {e}")

        # 启用推理模式
        with torch.no_grad():
            pass

        return model

    @staticmethod
    def check_gpu_memory() -> Dict[str, float]:
        """检查 GPU 内存使用情况"""
        if not torch.cuda.is_available():
            return {"available": False}

        memory_info = {
            "available": True,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 2,
            "allocated_mb": torch.cuda.memory_allocated() / 1024 ** 2,
            "cached_mb": torch.cuda.memory_reserved() / 1024 ** 2,
            "free_mb": (torch.cuda.get_device_properties(0).total_memory -
                        torch.cuda.memory_allocated()) / 1024 ** 2
        }

        return memory_info

    @staticmethod
    def clear_gpu_cache():
        """清空 GPU 缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")


class EmbeddingUtils:
    """嵌入工具类"""

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """归一化嵌入向量"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除以零
        return embeddings / norms

    @staticmethod
    def calculate_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        embeddings1_norm = EmbeddingUtils.normalize_embeddings(embeddings1)
        embeddings2_norm = EmbeddingUtils.normalize_embeddings(embeddings2)

        return np.dot(embeddings1_norm, embeddings2_norm.T)

    @staticmethod
    def find_similar_embeddings(query_embedding: np.ndarray, corpus_embeddings: np.ndarray,
                                top_k: int = 5, threshold: float = 0.7) -> List[Tuple[int, float]]:
        """查找相似的嵌入向量"""
        similarities = EmbeddingUtils.calculate_cosine_similarity(query_embedding, corpus_embeddings)
        similarities = similarities.flatten()

        # 获取最相似的结果
        indices = np.argsort(similarities)[::-1]
        results = []

        for idx in indices:
            if similarities[idx] >= threshold and len(results) < top_k:
                results.append((idx, float(similarities[idx])))

        return results