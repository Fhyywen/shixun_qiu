import os
import shutil

import chromadb
from typing import List, Dict, Any

from chromadb import Settings

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

        # 确保目录存在
        self._ensure_directories_exist()

        # 初始化 ChromaDB 客户端（修复重复初始化问题）
        self.client = self._init_chroma_client()

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB 集合 '{self.config.COLLECTION_NAME}' 初始化完成")

    def _ensure_directories_exist(self):
        """确保必要的目录存在"""
        os.makedirs(self.config.CHROMA_DB_PATH, exist_ok=True)
        os.makedirs(self.config.KNOWLEDGE_BASE_PATH, exist_ok=True)

    def _init_chroma_client(self):
        """初始化 ChromaDB 客户端，处理设置冲突"""
        try:
            # 首先尝试使用默认设置
            client = chromadb.PersistentClient(
                path=self.config.CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            return client
        except ValueError as e:
            if "different settings" in str(e):
                print("检测到 ChromaDB 设置冲突，正在清理并重新创建...")
                # 清理现有数据
                if os.path.exists(self.config.CHROMA_DB_PATH):
                    shutil.rmtree(self.config.CHROMA_DB_PATH)
                    os.makedirs(self.config.CHROMA_DB_PATH, exist_ok=True)

                # 重新创建客户端
                client = chromadb.PersistentClient(
                    path=self.config.CHROMA_DB_PATH,
                    settings=Settings(anonymized_telemetry=False)
                )
                return client
            else:
                raise e

    def build_knowledge_base(self, data_path: str = None):
        """构建知识库（支持增量更新）"""
        if data_path is None:
            data_path = self.config.KNOWLEDGE_BASE_PATH

        print(f"开始构建知识库，数据路径: {data_path}")

        # 检查数据目录
        if not os.path.exists(data_path) or not any(os.scandir(data_path)):
            print(f"数据路径不存在或为空: {data_path}")
            return 0

        # 加载已有的文件哈希记录
        file_hashes = self._load_file_hashes()

        # 扫描文件并识别变更
        new_or_modified_files = []
        all_files = []

        for root, _, files in os.walk(data_path):
            for file in files:
                if file.startswith('.'):  # 跳过隐藏文件
                    continue

                file_path = os.path.join(root, file)
                all_files.append(file_path)

                # 计算文件哈希
                current_hash = self._calculate_file_hash(file_path)

                # 检查文件是否新增或修改
                if file_path not in file_hashes or file_hashes[file_path] != current_hash:
                    new_or_modified_files.append(file_path)
                    file_hashes[file_path] = current_hash

        # 删除不存在的文件记录
        files_to_remove = [f for f in file_hashes.keys() if f not in all_files]
        for file_path in files_to_remove:
            if file_path in file_hashes:
                del file_hashes[file_path]
            # 从知识库中删除对应文档
            self._remove_documents_from_source(file_path)

        if not new_or_modified_files and not files_to_remove:
            print("没有检测到文件变更，跳过构建")
            return 0

        print(f"检测到 {len(new_or_modified_files)} 个新增/修改文件，{len(files_to_remove)} 个删除文件")

        # 处理新增/修改的文件
        documents = []
        for file_path in new_or_modified_files:
            try:
                # 方式1：复用 DataProcessor 已有的细分加载方法（推荐，更高效）
                file_ext = os.path.splitext(file_path)[1].lower()  # 获取文件后缀（如 .md, .txt）
                if file_ext in ('.txt', '.md', '.rst', '.markdown'):
                    # 文本类文件：调用 DataProcessor 的 _load_text_file 方法
                    content = self.processor._load_text_file(file_path)
                elif file_ext == '.csv':
                    # CSV文件：调用 _load_csv_file 方法
                    content = self.processor._load_csv_file(file_path)
                elif file_ext in ('.xlsx', '.xls'):
                    # Excel文件：调用 _load_excel_file 方法
                    content = self.processor._load_excel_file(file_path)
                elif file_ext == '.docx':  # 添加对Word文档的支持
                    # Word文档：调用 _load_word_file 方法
                    content = self.processor._load_word_file(file_path)
                elif file.endswith('.doc'):
                    content = self.processor._load_doc_file(file_path)
                else:
                    print(f"跳过不支持的文件格式: {file_path}")
                    continue

                # 将加载的内容添加到文档列表
                documents.append({
                    "content": content,
                    "source": file_path,
                    "type": file_ext  # 记录文件类型
                })
                print(f"加载文件: {file_path}")

            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")

        if not documents and not files_to_remove:
            print("没有需要处理的文档")
            return 0

        # 处理文档分块和嵌入
        chunks = self.processor.chunk_documents(documents)
        print(f"文档分块完成，共 {len(chunks)} 个块")

        # 生成嵌入向量
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.processor.generate_embeddings(chunk_texts)
        print("嵌入向量生成完成")

        # 准备数据用于存储
        ids = [f"doc_{hash(chunk['source'] + str(chunk['chunk_index'])):x}" for chunk in chunks]
        documents_content = [chunk["content"] for chunk in chunks]
        metadatas = [{
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "document_type": os.path.splitext(chunk["source"])[1] if "source" in chunk else "unknown",
            "file_hash": file_hashes.get(chunk["source"], "unknown")
        } for chunk in chunks]

        # 先删除已修改文件的旧文档
        for file_path in new_or_modified_files:
            self._remove_documents_from_source(file_path)

        # 添加新文档到集合
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents_content,
                metadatas=metadatas
            )

            # 保存文件哈希记录
            self._save_file_hashes(file_hashes)

            count = self.collection.count()
            print(f"知识库更新完成，当前文档总数: {count}")

            return len(chunks)

        except Exception as e:
            print(f"更新知识库时出错: {e}")
            return 0

    def _load_file_hashes(self):
        """加载文件哈希记录"""
        if os.path.exists(self.config.FILE_HASH_DB):
            try:
                with open(self.config.FILE_HASH_DB, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_file_hashes(self, file_hashes):
        """保存文件哈希记录"""
        os.makedirs(os.path.dirname(self.config.FILE_HASH_DB), exist_ok=True)
        with open(self.config.FILE_HASH_DB, 'w', encoding='utf-8') as f:
            json.dump(file_hashes, f, ensure_ascii=False, indent=2)

    def _calculate_file_hash(self, file_path):
        """计算文件哈希值"""
        import hashlib
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except:
            return "error"

    def _remove_documents_from_source(self, source_path):
        """删除指定来源的所有文档"""
        try:
            # 查询所有来自该来源的文档
            results = self.collection.get(
                where={"source": source_path}
                # 不需要指定 include=["ids"]，因为 ids 是默认返回的
                # 可以添加其他需要的字段，如 include=["metadatas"]
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"已删除 {len(results['ids'])} 个来自 {source_path} 的文档")

        except Exception as e:
            print(f"删除文档时出错: {e}")

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
                print("similarity=", similarity)
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
        #读取template文件 TODO
        try:
            with open(self.config.ANSWER_TEMPLATE, 'r', encoding='utf-8') as f:
                template = f.read()
            print(f"成功读取模板文件: {self.config.ANSWER_TEMPLATE}")
            print("文件内容为:",template)
        except FileNotFoundError:
            print(f"模板文件不存在: {self.config.ANSWER_TEMPLATE}")
            template = "# 默认模板\n\n这是一个默认的回答模板。"
        except Exception as e:
            print(f"读取模板文件时出错: {e}")
            template = "# 错误\n\n无法加载模板文件。"


        # 构建上下文
        context_text = "\n\n".join([f"相关文档 {i + 1} (相似度: {doc['similarity']:.2f}):\n{doc['content']}"
                                    for i, doc in enumerate(context)])

        # 构建提示词
        prompt = f"""你是一个社会调研专家。基于以下相关知识，请回答用户的问题。

相关背景知识：
{context_text}
套用模板：
{template}
用户问题：{query}

使用模板结合背景知识来生成回答。"""

        messages = [
            {"role": "system", "content": "你是一个社会调研专家。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_provider.generate_response(messages)
            return response
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def generate_answer_without_context(self, query: str) -> str:
        """当没有本地上下文时，使用LLM的一般知识回答"""
        prompt = f"""你是一个社会调研专家。请回答以下关于社会调研专家的问题。

用户问题：{query}

请基于你的专业知识提供准确、专业的回答。如果你是推测或不确定，请说明。"""

        messages = [
            {"role": "system", "content": "你是一个社会调研专家。"},
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