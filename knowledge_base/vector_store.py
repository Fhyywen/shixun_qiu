import faiss
import numpy as np
import pickle
import os
from config import config


class VectorStore:
    def __init__(self, dimension=None):
        self.dimension = dimension or config.EMBEDDING_DIMENSION
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []

    def add_embeddings(self, embeddings, texts, metadata=None):
        """添加嵌入向量到索引"""
        if not embeddings or not texts:
            return

        if metadata is None:
            metadata = [{}] * len(texts)
        elif len(metadata) != len(texts):
            metadata = [{}] * len(texts)

        # 转换为numpy数组
        embeddings = np.array(embeddings).astype('float32')

        # 添加到FAISS索引
        if len(embeddings) > 0:
            self.index.add(embeddings)

        # 存储文本和元数据
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_embedding, k=5):
        """搜索最相似的k个结果"""
        if len(self.texts) == 0:
            return []

        k = min(k, len(self.texts))
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts) and idx >= 0:  # 确保索引有效
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i]),
                    'similarity': float(1 - distances[0][i])  # 添加相似度分数
                })

        return results

    def save(self, file_path):
        """保存向量存储到文件"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, f"{file_path}.index")

            # 保存文本和元数据
            with open(f"{file_path}.data", 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'metadata': self.metadata,
                    'dimension': self.dimension
                }, f)
            return True
        except Exception as e:
            print(f"保存向量存储失败: {e}")
            return False

    def load(self, file_path):
        """从文件加载向量存储"""
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(f"{file_path}.index")

            # 加载文本和元数据
            with open(f"{file_path}.data", 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                if 'dimension' in data:
                    self.dimension = data['dimension']
            return True
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return False

    def get_stats(self):
        """获取向量存储统计信息"""
        return {
            'num_vectors': len(self.texts),
            'dimension': self.dimension,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }