import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config.settings import settings
from core.model_manager import ModelManager
from core.knowledge_manager import KnowledgeManager, Document

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query: str
    documents: List[Document]
    answer: str
    confidence: float
    processing_time: float


class RAGSystem:
    """RAG 系统核心"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.knowledge_manager = KnowledgeManager(self.model_manager)
        self.current_kb = None

    def initialize(self) -> bool:
        """初始化系统"""
        try:
            if not self.model_manager.initialize_models():
                return False

            # 创建默认知识库
            self.current_kb = self.knowledge_manager.create_knowledge_base("default")
            logger.info("RAG system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False

    def add_documents(self, file_paths: List[str], kb_name: str = "default") -> bool:
        """添加文档到知识库"""
        try:
            kb = self.knowledge_manager.get_knowledge_base(kb_name)
            if not kb:
                kb = self.knowledge_manager.create_knowledge_base(kb_name)

            for file_path in file_paths:
                kb.add_document(file_path)

            kb.build_index()
            self.current_kb = kb
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Document]:
        """检索相关文档"""
        if not self.current_kb:
            raise ValueError("No knowledge base loaded")

        return self.current_kb.search(query, top_k, threshold)

    def query(self, query: str, **kwargs) -> QueryResult:
        """执行完整查询"""
        import time
        start_time = time.time()

        # 检索相关文档
        documents = self.retrieve(query, **kwargs)

        # 构建提示
        context = "\n\n".join([doc.content for doc in documents])
        prompt = f"""基于以下背景信息回答问题。如果信息不足，请说明无法回答。

背景信息:
{context}

问题: {query}

请提供准确、简洁的回答:"""

        # 生成回答
        generation_model = self.model_manager.get_model("generation")
        answer = generation_model.generate(prompt)

        processing_time = time.time() - start_time

        return QueryResult(
            query=query,
            documents=documents,
            answer=answer,
            confidence=0.8,  # 简化版置信度
            processing_time=processing_time
        )

    def batch_query(self, queries: List[str], **kwargs) -> List[QueryResult]:
        """批量查询"""
        return [self.query(q, **kwargs) for q in queries]

    def shutdown(self):
        """关闭系统"""
        self.model_manager.unload_all()
        logger.info("RAG system shutdown complete")