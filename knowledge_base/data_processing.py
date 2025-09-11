import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config


class DataProcessor:
    def __init__(self, model_name=None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def load_documents(self, directory_path):
        """从目录加载文档"""
        documents = []
        if not os.path.exists(directory_path):
            return documents

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in config.SUPPORTED_FILE_TYPES):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'title': filename,
                            'content': content,
                            'chunks': self.split_into_chunks(content)
                        })
                    print(f"成功加载文档: {filename}")
                except Exception as e:
                    print(f"加载文档 {filename} 时出错: {e}")
        return documents

    def split_into_chunks(self, text, chunk_size=None, overlap=None):
        """将文本分割成重叠的块"""
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP

        # 按句子分割，保持语义完整性
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size and current_chunk:
                # 保存当前块
                chunks.append(' '.join(current_chunk))
                # 保留重叠部分
                overlap_words = ' '.join(current_chunk).split()[-overlap:]
                current_chunk = [' '.join(overlap_words)] if overlap > 0 else []
                current_length = len(current_chunk[0].split()) if current_chunk else 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # 添加最后一个块
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_embeddings(self, texts):
        """为文本列表生成嵌入向量"""
        if not texts:
            return []
        return self.model.encode(texts, show_progress_bar=True)