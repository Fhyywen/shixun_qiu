import os
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from utils.file_utils import load_document, split_text

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None


class KnowledgeBase:
    """知识库管理"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
        self.index = None

    def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """添加文档到知识库"""
        try:
            content = load_document(file_path)
            if not content:
                raise ValueError(f"Failed to load content from {file_path}")

            # 分割文本
            chunks = split_text(content, settings.knowledge_base.chunk_size,
                                settings.knowledge_base.chunk_overlap)

            doc_ids = []
            for i, chunk in enumerate(chunks):
                doc_id = f"{os.path.basename(file_path)}_chunk_{i}"
                document = Document(
                    id=doc_id,
                    content=chunk,
                    metadata=metadata or {}
                )
                self.documents.append(document)
                doc_ids.append(doc_id)

            logger.info(f"Added {len(chunks)} chunks from {file_path}")
            return doc_ids

        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            raise

    def build_index(self):
        """构建索引"""
        if not self.documents:
            raise ValueError("No documents in knowledge base")

        # 生成嵌入向量
        texts = [doc.content for doc in self.documents]
        embeddings = self.embedding_model.embed_batch(texts)

        # 存储嵌入向量
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding

        self.embeddings = np.array(embeddings)
        logger.info(f"Built index with {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Document]:
        """搜索相关文档"""
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        # 生成查询嵌入
        query_embedding = self.embedding_model.embed(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # 获取最相关的文档
        indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in indices:
            if similarities[idx] >= threshold:
                results.append((self.documents[idx], similarities[idx]))

        return [doc for doc, score in sorted(results, key=lambda x: x[1], reverse=True)]

    def save(self, path: str):
        """保存知识库"""
        data = {
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding
                }
                for doc in self.documents
            ]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Knowledge base saved to {path}")

    def load(self, path: str):
        """加载知识库"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.documents = [
                Document(
                    id=doc_data["id"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    embedding=doc_data["embedding"]
                )
                for doc_data in data["documents"]
            ]

            if self.documents and self.documents[0].embedding:
                self.embeddings = np.array([doc.embedding for doc in self.documents])

            logger.info(f"Knowledge base loaded from {path} with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise


class KnowledgeManager:
    """知识管理器"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}

    def create_knowledge_base(self, name: str) -> KnowledgeBase:
        """创建新的知识库"""
        embedding_model = self.model_manager.get_model("embedding")
        if not embedding_model or not embedding_model.is_loaded():
            raise ValueError("Embedding model not available")

        kb = KnowledgeBase(embedding_model)
        self.knowledge_bases[name] = kb
        logger.info(f"Created knowledge base: {name}")
        return kb

    def get_knowledge_base(self, name: str) -> Optional[KnowledgeBase]:
        """获取知识库"""
        return self.knowledge_bases.get(name)

    def list_knowledge_bases(self) -> List[str]:
        """列出所有知识库"""
        return list(self.knowledge_bases.keys())