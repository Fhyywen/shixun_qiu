import faiss
import numpy as np
import pickle
import os
from config import Config


class VectorStore:
    def __init__(self, dimension=None):
        self.dimension = dimension or Config.EMBEDDING_DIMENSION
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

    def get(self, where=None, include=None):
        """
        获取向量存储中的文档

        Args:
            where: 过滤条件字典
            include: 要包含的字段列表，只能是 ['documents', 'embeddings', 'metadatas']

        Returns:
            包含查询结果的字典
        """
        if include is None:
            include = ["documents"]

        # 验证include参数
        valid_include = ["documents", "embeddings", "metadatas", "distances", "uris", "data"]
        for item in include:
            if item not in valid_include:
                raise ValueError(f"Expected include item to be one of {valid_include}, got {item} in get.")

        results = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'embeddings': []
        }

        # 根据where条件过滤文档
        if where:
            # 找到匹配的索引
            matched_indices = []
            for i, meta in enumerate(self.metadata):
                match = True
                for key, value in where.items():
                    if meta.get(key) != value:
                        match = False
                        break
                if match:
                    matched_indices.append(i)

            # 构建结果
            results['ids'] = [str(i) for i in matched_indices]
            if "documents" in include:
                results['documents'] = [self.texts[i] for i in matched_indices]
            if "metadatas" in include:
                results['metadatas'] = [self.metadata[i] for i in matched_indices]
            if "embeddings" in include and hasattr(self.index, 'reconstruct'):
                try:
                    embeddings = []
                    for i in matched_indices:
                        embedding = self.index.reconstruct(i)
                        embeddings.append(embedding)
                    results['embeddings'] = embeddings
                except:
                    results['embeddings'] = []
        else:
            # 返回所有文档
            results['ids'] = [str(i) for i in range(len(self.texts))]
            if "documents" in include:
                results['documents'] = self.texts[:]
            if "metadatas" in include:
                results['metadatas'] = self.metadata[:]
            if "embeddings" in include:
                results['embeddings'] = []  # FAISS IndexFlatL2不支持直接获取向量

        return results

    def delete(self, ids=None, where=None):
        """
        从向量存储中删除文档

        Args:
            ids: 要删除的文档ID列表
            where: 删除条件字典

        Returns:
            删除操作的结果
        """
        indices_to_delete = set()

        # 根据ids删除
        if ids:
            for id_str in ids:
                try:
                    idx = int(id_str)
                    if 0 <= idx < len(self.texts):
                        indices_to_delete.add(idx)
                except (ValueError, TypeError):
                    continue

        # 根据where条件删除
        if where:
            for i, meta in enumerate(self.metadata):
                match = True
                for key, value in where.items():
                    if meta.get(key) != value:
                        match = False
                        break
                if match:
                    indices_to_delete.add(i)

        if not indices_to_delete:
            return {"success": True, "deleted_count": 0}

        # 转换为排序列表（降序），便于删除
        sorted_indices = sorted(list(indices_to_delete), reverse=True)

        # 删除文档和元数据
        for idx in sorted_indices:
            if idx < len(self.texts):
                self.texts.pop(idx)
                self.metadata.pop(idx)

        # 注意：FAISS索引中的向量无法轻易删除，需要重建索引
        # 这里只是简单地清空索引，实际使用中可能需要更好的解决方案
        print("注意：FAISS索引中的向量删除需要重建索引")

        return {"success": True, "deleted_count": len(indices_to_delete)}
