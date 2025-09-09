import logging
import random
from typing import List, Dict, Any, Tuple
import json

from config.settings import settings
from core.model_manager import GenerationModel

logger = logging.getLogger(__name__)


class DataConstructor:
    """数据构建器"""

    def __init__(self, generation_model: GenerationModel):
        self.generation_model = generation_model

    def generate_queries_from_documents(self, documents: List[str], num_queries_per_doc: int = 3) -> List[str]:
        """从文档生成查询问题"""
        queries = []

        for doc_content in documents:
            prompt = f"""根据以下文档内容，生成{num_queries_per_doc}个可能的问题。每个问题应该：
            1. 基于文档内容
            2. 清晰具体
            3. 覆盖文档的不同方面

            文档内容:
            {doc_content[:1000]}

            请直接输出问题，每个问题一行:"""

            try:
                response = self.generation_model.generate(prompt, max_length=500)
                generated_queries = [q.strip() for q in response.split('\n') if q.strip()]
                queries.extend(generated_queries[:num_queries_per_doc])
            except Exception as e:
                logger.error(f"Error generating queries: {e}")
                continue

        return queries

    def create_sft_data(self, documents: List[str], queries: List[str]) -> List[Dict[str, Any]]:
        """创建监督微调数据"""
        training_data = []

        for query in queries:
            # 简化版：选择最相关的文档
            best_doc = None
            best_score = 0

            for doc in documents:
                # 简单的关键词匹配评分
                score = sum(1 for word in query.split() if word.lower() in doc.lower())
                if score > best_score:
                    best_score = score
                    best_doc = doc

            if best_doc:
                training_data.append({
                    "query": query,
                    "document": best_doc,
                    "instruction": "基于给定的文档回答问题",
                    "input": f"文档: {best_doc}\n\n问题: {query}",
                    "output": "[需要LLM生成的答案]"
                })

        return training_data

    def create_negative_samples(self, queries: List[str], documents: List[str], num_negatives: int = 3) -> List[
        Dict[str, Any]]:
        """创建负样本数据"""
        negative_samples = []

        for query in queries:
            # 随机选择不相关的文档作为负样本
            negative_docs = random.sample(documents, min(num_negatives, len(documents)))

            for neg_doc in negative_docs:
                negative_samples.append({
                    "query": query,
                    "positive_document": "",  # 需要正样本信息
                    "negative_document": neg_doc,
                    "score": 0.0  # 低相似度分数
                })

        return negative_samples

    def save_training_data(self, data: List[Dict[str, Any]], file_path: str):
        """保存训练数据"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(data)} training examples to {file_path}")

    def load_training_data(self, file_path: str) -> List[Dict[str, Any]]:
        """加载训练数据"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            logger.info(f"Loaded {len(data)} training examples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading training data: {e}")

        return data