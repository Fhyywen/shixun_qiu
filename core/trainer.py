import logging
import torch
from typing import Dict, Any, List
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = None

    def setup_training(self, output_dir: str, **kwargs):
        """设置训练参数"""
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=kwargs.get("num_epochs", settings.training.num_epochs),
            per_device_train_batch_size=kwargs.get("batch_size", settings.training.batch_size),
            per_device_eval_batch_size=kwargs.get("batch_size", settings.training.batch_size),
            learning_rate=kwargs.get("learning_rate", settings.training.learning_rate),
            evaluation_strategy="steps" if kwargs.get("eval_steps") else "no",
            eval_steps=kwargs.get("eval_steps", settings.training.eval_steps),
            save_steps=kwargs.get("save_steps", settings.training.save_steps),
            logging_steps=kwargs.get("logging_steps", settings.training.logging_steps),
            logging_dir=settings.log_dir,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

    def train_sft(self, train_dataset, eval_dataset=None, **kwargs):
        """监督微调训练"""
        if self.training_args is None:
            self.setup_training(**kwargs)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting SFT training...")
        trainer.train()

        # 保存模型
        trainer.save_model()
        logger.info(f"Model saved to {self.training_args.output_dir}")

        return trainer

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # 简化版指标计算
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}