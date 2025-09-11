import os
import chromadb
from typing import List, Dict, Any
from knowledge_base.data_processing import DataProcessor
from knowledge_base.llm_providers import LLMProvider
from config import Config
import requests
import json


class TimeSeriesQA:
    def __init__(self, config: Config = None):
        if config is None:
            self.config = Config()
        else:
            self.config = config

        self.processor = DataProcessor(model_name=self.config.EMBEDDING_MODEL)
        self.llm_provider = LLMProvider(self.config)
        self.chroma_client = chromadb.PersistentClient(path=self.config.CHROMA_DB_PATH)

        # 尝试获取或创建集合
        try:
            self.collection = self.chroma_client.get_collection("time_series_knowledge")
            print("连接到现有的向量数据库集合")
        except:
            self.collection = self.chroma_client.create_collection("time_series_knowledge")
            print("创建新的向量数据库集合")

    def build_knowledge_base(self, data_path: str = None):
        """构建知识库"""
        if data_path is None:
            data_path = self.config.DATA_PATH

        print("开始构建知识库...")
        documents = self.processor.load_documents(data_path)

        if not documents:
            print("没有找到任何文档，请检查数据路径")
            return 0

        chunks = self.processor.chunk_documents(documents)
        embeddings = self.processor.generate_embeddings([chunk["content"] for chunk in chunks])

        # 准备数据用于存储
        ids = [f"doc_{i}" for i in range(len(chunks))]
        documents_content = [chunk["content"] for chunk in chunks]
        metadatas = [{
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"]
        } for chunk in chunks]

        # 存储到ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents_content,
            metadatas=metadatas
        )

        print(f"知识库构建完成，共处理 {len(chunks)} 个文档块")
        return len(chunks)

    def search_similar_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        if top_k is None:
            top_k = self.config.TOP_K

        # 生成查询嵌入
        query_embedding = self.processor.generate_embeddings([query])[0].tolist()

        # 搜索相似文档
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        similar_docs = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                similarity = 1 - results['distances'][0][i]  # 转换距离为相似度
                if similarity >= self.config.SIMILARITY_THRESHOLD:
                    similar_docs.append({
                        "content": results['documents'][0][i],
                        "similarity": similarity,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][
                            0] else {}
                    })

        return similar_docs

    def generate_answer_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """基于上下文生成答案"""
        # 构建上下文
        context_text = "\n\n".join([f"相关文档 {i + 1} (相似度: {doc['similarity']:.2f}):\n{doc['content']}"
                                    for i, doc in enumerate(context)])

        # 构建提示词
        prompt = f"""你是一个时间序列分析专家。基于以下相关知识，请回答用户的问题。

相关背景知识：
{context_text}

用户问题：{query}

请根据上述知识提供专业、准确的回答。如果信息不足，可以基于你的专业知识进行补充。"""

        messages = [
            {"role": "system", "content": "你是一个时间序列分析专家，擅长金融数据分析、预测建模和统计学习。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_provider.generate_response(messages)
            return response
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def generate_answer_without_context(self, query: str) -> str:
        """当没有本地上下文时，使用LLM的一般知识回答"""
        prompt = f"""你是一个时间序列分析专家。请回答以下关于时间序列分析的问题。

用户问题：{query}

请基于你的专业知识提供准确、专业的回答。如果你是推测或不确定，请说明。"""

        messages = [
            {"role": "system", "content": "你是一个时间序列分析专家，擅长金融数据分析、预测建模和统计学习。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_provider.generate_response(messages)
            return response
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def search_web_knowledge(self, query: str) -> str:
        """模拟联网搜索（实际使用时可以接入真正的搜索API）"""
        # 这里可以接入真正的搜索API，如Serper、Google Search等
        # 目前先返回一个提示信息
        return "（注：此回答基于大模型的通用知识，如需更准确的信息建议联网搜索）"

    def ask_question(self, question: str) -> Dict[str, Any]:
        """提问并获取答案"""
        # 搜索相关文档
        similar_docs = self.search_similar_documents(question)

        if similar_docs:
            # 有相关文档，基于上下文生成答案
            answer = self.generate_answer_with_context(question, similar_docs)
            confidence = sum(doc["similarity"] for doc in similar_docs) / len(similar_docs)

            return {
                "answer": answer,
                "sources": [{"content": doc["content"][:200] + "...", "similarity": doc["similarity"]} for doc in
                            similar_docs],
                "confidence": confidence,
                "source_type": "knowledge_base"
            }
        else:
            # 没有相关文档，使用LLM的一般知识回答
            answer = self.generate_answer_without_context(question)
            web_info = self.search_web_knowledge(question)

            return {
                "answer": f"{answer}\n\n{web_info}",
                "sources": [],
                "confidence": 0.3,  # 通用知识的置信度较低
                "source_type": "general_knowledge"
            }