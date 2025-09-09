import logging
import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理工具类"""

    @staticmethod
    def split_data(data: List[Any], train_ratio: float = 0.8, val_ratio: float = 0.1,
                   test_ratio: float = 0.1, random_state: int = 42) -> Tuple[List, List, List]:
        """分割数据集"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("比例之和必须为 1.0")

        # 先分割训练集
        train_data, temp_data = train_test_split(
            data, train_size=train_ratio, random_state=random_state
        )

        # 再分割验证集和测试集
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, train_size=val_size, random_state=random_state
        )

        return train_data, val_data, test_data

    @staticmethod
    def create_train_val_test_splits(data: List[Dict[str, Any]], **kwargs) -> Dict[str, List]:
        """创建训练/验证/测试分割"""
        train_data, val_data, test_data = DataProcessor.split_data(data, **kwargs)

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }

    @staticmethod
    def balance_dataset(data: List[Any], target_key: str = None) -> List[Any]:
        """平衡数据集（简化版）"""
        if not data:
            return data

        if target_key:
            # 按目标值分组
            groups = {}
            for item in data:
                target_value = item.get(target_key)
                if target_value not in groups:
                    groups[target_value] = []
                groups[target_value].append(item)

            # 找到最小分组大小
            min_size = min(len(group) for group in groups.values())

            # 平衡数据
            balanced_data = []
            for group in groups.values():
                balanced_data.extend(random.sample(group, min_size))

            return balanced_data

        return data

    @staticmethod
    def augment_text_data(texts: List[str], augmentation_factor: int = 2) -> List[str]:
        """文本数据增强（简化版）"""
        augmented_texts = []

        for text in texts:
            augmented_texts.append(text)

            # 简单的增强方法
            words = text.split()
            if len(words) > 3:
                # 随机打乱词语顺序
                for _ in range(augmentation_factor - 1):
                    shuffled_words = words.copy()
                    random.shuffle(shuffled_words)
                    augmented_texts.append(' '.join(shuffled_words))

        return augmented_texts

    @staticmethod
    def calculate_dataset_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算数据集统计信息"""
        if not data:
            return {}

        stats = {
            "total_samples": len(data),
            "avg_text_length": 0,
            "max_text_length": 0,
            "min_text_length": float('inf'),
        }

        text_lengths = []
        for item in data:
            if 'text' in item:
                length = len(item['text'])
                text_lengths.append(length)
                stats['max_text_length'] = max(stats['max_text_length'], length)
                stats['min_text_length'] = min(stats['min_text_length'], length)

        if text_lengths:
            stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)

        return stats


class DatasetValidator:
    """数据集验证器"""

    @staticmethod
    def validate_dataset_format(data: List[Dict[str, Any]], required_fields: List[str]) -> bool:
        """验证数据集格式"""
        if not data:
            return False

        for item in data:
            for field in required_fields:
                if field not in item:
                    return False

        return True

    @staticmethod
    def check_data_quality(data: List[Dict[str, Any]], text_field: str = "text") -> Dict[str, Any]:
        """检查数据质量"""
        quality_report = {
            "total_samples": len(data),
            "empty_texts": 0,
            "short_texts": 0,
            "duplicate_texts": 0,
            "quality_score": 0.0
        }

        texts_seen = set()

        for item in data:
            text = item.get(text_field, "")

            if not text.strip():
                quality_report["empty_texts"] += 1
            elif len(text.strip()) < 10:
                quality_report["short_texts"] += 1

            if text in texts_seen:
                quality_report["duplicate_texts"] += 1
            else:
                texts_seen.add(text)

        # 计算质量分数
        total_issues = (quality_report["empty_texts"] +
                        quality_report["short_texts"] +
                        quality_report["duplicate_texts"])

        quality_report["quality_score"] = 1.0 - (total_issues / quality_report["total_samples"])

        return quality_report