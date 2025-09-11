from .data_processing import DataProcessor
from .vector_store import VectorStore
import os
from config import config


class TimeSeriesQA:
    def __init__(self,
                 data_dir=None,
                 vector_store_path=None,
                 embedding_model=None,
                 chunk_size=None,
                 chunk_overlap=None):

        self.data_dir = data_dir or config.DATA_DIR
        self.vector_store_path = vector_store_path or config.VECTOR_STORE_PATH
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        self.processor = DataProcessor(model_name=self.embedding_model)
        self.vector_store = VectorStore(dimension=config.EMBEDDING_DIMENSION)
        self.initialized = False

    def initialize(self):
        """初始化知识库"""
        if self.initialized:
            return True

        # 检查是否有保存的向量存储
        index_path = f"{self.vector_store_path}.index"
        data_path = f"{self.vector_store_path}.data"

        if os.path.exists(index_path) and os.path.exists(data_path):
            print("加载现有的向量存储...")
            try:
                self.vector_store.load(self.vector_store_path)
                self.initialized = True
                print("知识库加载完成!")
                return True
            except Exception as e:
                print(f"加载向量存储失败: {e}")
                # 继续创建新的向量存储

        print("创建新的向量存储...")
        # 加载和处理文档
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"警告: 数据目录 {self.data_dir} 不存在，已创建空目录")
            return False

        documents = self.processor.load_documents(self.data_dir)

        if not documents:
            print(f"警告: 在 {self.data_dir} 中没有找到文档")
            return False

        # 准备文本和元数据
        all_chunks = []
        all_metadata = []

        for doc in documents:
            for chunk in doc['chunks']:
                all_chunks.append(chunk)
                all_metadata.append({
                    'title': doc['title'],
                    'source': doc['title']
                })

        # 生成嵌入向量
        print("生成嵌入向量...")
        embeddings = self.processor.generate_embeddings(all_chunks)

        # 添加到向量存储
        print("构建向量索引...")
        self.vector_store.add_embeddings(embeddings, all_chunks, all_metadata)

        # 保存向量存储
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save(self.vector_store_path)

        self.initialized = True
        print("知识库初始化完成!")
        return True

    def ask(self, question, k=5):
        """回答问题"""
        if not self.initialized:
            if not self.initialize():
                return {
                    'question': question,
                    'results': [],
                    'answer': "知识库尚未初始化或没有可用的文档。"
                }

        # 生成问题的嵌入向量
        question_embedding = self.processor.generate_embeddings([question])[0]

        # 搜索最相关的文档块
        results = self.vector_store.search(question_embedding, k=k)

        # 构建回答
        answer = {
            'question': question,
            'results': results,
            'answer': self.generate_answer(question, results)
        }

        return answer

    def generate_answer(self, question, results):
        """基于检索结果生成回答"""
        if not results:
            return "抱歉，我没有找到相关的时间序列预测算法信息。"

        # 简单实现：返回最相关的结果
        context = "\n".join([f"[来自: {result['metadata']['title']}]\n{result['text']}"
                             for result in results])

        # 在实际应用中，这里可以使用LLM生成更精确的回答
        answer = f"基于时间序列预测知识库，我找到了以下相关信息：\n\n{context}"

        return answer

    def get_stats(self):
        """获取知识库统计信息"""
        return {
            'initialized': self.initialized,
            'document_count': len(self.vector_store.texts) if self.initialized else 0,
            'vector_dimension': config.EMBEDDING_DIMENSION
        }