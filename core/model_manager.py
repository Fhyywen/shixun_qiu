import os
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """模型基类"""

    @abstractmethod
    def load(self, model_path: str, **kwargs):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        pass


class EmbeddingModel(BaseModel):
    """嵌入模型"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None

    def load(self, model_path: str = None, **kwargs):
        if model_path is None:
            model_path = settings.model.embedding_model

        try:
            self.model = AutoModel.from_pretrained(model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model_path = model_path
            logger.info(f"Loaded embedding model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def embed(self, text: str) -> List[float]:
        if not self.is_loaded():
            raise ValueError("Model not loaded")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=settings.model.max_length, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class GenerationModel(BaseModel):
    """生成模型"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_path = None

    def load(self, model_path: str = None, **kwargs):
        if model_path is None:
            model_path = settings.model.generation_model

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            self.model_path = model_path
            logger.info(f"Loaded generation model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            raise

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_path = None

    def is_loaded(self) -> bool:
        return self.generator is not None

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded():
            raise ValueError("Model not loaded")

        generation_config = {
            "max_length": kwargs.get("max_length", settings.model.max_length),
            "temperature": kwargs.get("temperature", settings.model.temperature),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        result = self.generator(prompt, **generation_config)
        return result[0]["generated_text"]


class ModelManager:
    """模型管理器"""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.generation_model = GenerationModel()
        self.loaded_models: Dict[str, BaseModel] = {}

    def initialize_models(self):
        """初始化所有模型"""
        try:
            self.embedding_model.load()
            self.generation_model.load()

            self.loaded_models = {
                "embedding": self.embedding_model,
                "generation": self.generation_model
            }

            logger.info("All models initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False

    def get_model(self, model_type: str) -> Optional[BaseModel]:
        """获取指定类型的模型"""
        return self.loaded_models.get(model_type)

    def unload_all(self):
        """卸载所有模型"""
        for model in self.loaded_models.values():
            model.unload()
        self.loaded_models.clear()
        logger.info("All models unloaded")