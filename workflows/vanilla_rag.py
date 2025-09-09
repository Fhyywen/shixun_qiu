import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from core.rag_system import RAGSystem, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class VanillaRAGConfig:
    top_k: int = 5
    score_threshold: float = 0.7
    max_context_length: int = 4000
    temperature: float = 0.1


class VanillaRAGWorkflow:
    """Vanilla RAG 工作流 - 基础检索增强生成"""

    def __init__(self, rag_system: RAGSystem, config: VanillaRAGConfig = None):
        self.rag_system = rag_system
        self.config = config or VanillaRAGConfig()

    def build_prompt(self, query: str, documents: List[Any]) -> str:
        """构建提示模板"""
        context = "\n\n".join([doc.content for doc in documents])

        prompt = f"""基于以下提供的背景信息，请回答用户的问题。如果信息不足以回答问题，请如实说明。

背景信息:
{context[:self.config.max_context_length]}

用户问题: {query}

请提供准确、完整且基于背景信息的回答:"""

        return prompt

    def query(self, query: str, **kwargs) -> QueryResult:
        """执行 Vanilla RAG 查询"""
        # 合并配置
        config = {**self.config.__dict__, **kwargs}

        # 检索相关文档
        documents = self.rag_system.retrieve(
            query,
            top_k=config['top_k'],
            threshold=config['score_threshold']
        )

        # 构建提示
        prompt = self.build_prompt(query, documents)

        # 生成回答
        generation_model = self.rag_system.model_manager.get_model("generation")
        answer = generation_model.generate(
            prompt,
            temperature=config['temperature'],
            max_length=1024
        )

        return QueryResult(
            query=query,
            documents=documents,
            answer=answer,
            confidence=0.8,  # 基于检索分数计算
            processing_time=0.0  # 实际使用时需要计时
        )

    def batch_query(self, queries: List[str], **kwargs) -> List[QueryResult]:
        """批量查询"""
        return [self.query(q, **kwargs) for q in queries]