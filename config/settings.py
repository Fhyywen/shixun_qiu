import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    generation_model: str = "microsoft/DialoGPT-medium"
    reranker_model: str = "BAAI/bge-reranker-base"
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    max_length: int = 512
    temperature: float = 0.1


@dataclass
class KnowledgeBaseConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    supported_formats: List[str] = None
    index_type: str = "hnsw"
    similarity_metric: str = "cosine"

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".txt", ".pdf", ".md", ".docx", ".csv", ".json"]


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10


@dataclass
class InferenceConfig:
    top_k: int = 5
    score_threshold: float = 0.7
    max_retries: int = 3
    timeout: int = 30


class Settings:
    def __init__(self):
        self.model = ModelConfig()
        self.knowledge_base = KnowledgeBaseConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.data_dir = "data"
        self.model_dir = "models"
        self.log_dir = "logs"

        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


settings = Settings()