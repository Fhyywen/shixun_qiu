import os
import shutil
from datetime import datetime

import chromadb
from typing import List, Dict, Any

from chromadb import Settings

from knowledge_base.data_processing import DataProcessor
from knowledge_base.database_manager import DatabaseManager
from knowledge_base.knowledge_base_analyzer import KnowledgeBaseAnalyzer
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

        # 初始化数据库管理器
        self.db_manager = DatabaseManager({
            'host': self.config.MYSQL_HOST,
            'port': self.config.MYSQL_PORT,
            'user': self.config.MYSQL_USER,
            'password': self.config.MYSQL_PASSWORD,
            'database': self.config.MYSQL_DATABASE,
            'charset': 'utf8mb4'
        })


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

        self.current_knowledge_base = self.config.KNOWLEDGE_BASE_PATH
        self.default_knowledge_base = self.config.KNOWLEDGE_BASE_PATH


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
                elif file_ext == '.pdf':
                    content = self.processor._load_pdf_file(file_path)
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

    def generate_answer_with_context(self, query: str, context: List[Dict[str, Any]],report: List[Dict[str, Any]]) -> str:
        """基于上下文生成答案"""
        #读取template文件 TODO
        try:
            template = self._read_template_file()
            print(f"成功读取模板文件: {self.config.ANSWER_TEMPLATE}")
            print("文件内容为:",template)
        except FileNotFoundError:
            print(f"模板文件不存在: {self.config.ANSWER_TEMPLATE}")
            template = "# 默认模板\n\n这是一个默认的回答模板。"
        except Exception as e:
            print(f"读取模板文件时出错: {e}")
            template = "# 错误\n\n无法加载模板文件。"

        print("数据参考:",report)

        # 构建上下文
        context_text = "\n\n".join([f"相关文档 {i + 1} (相似度: {doc['similarity']:.2f}):\n{doc['content']}"
                                    for i, doc in enumerate(context)])

        # 构建提示词
        prompt = f"""你是一个社会调研专家。基于以下相关知识，请回答用户的问题。

相关背景知识：
{context_text}
套用模板：
{template}
参考数据：
{report}
用户问题：{query}

使用模板结合背景知识来生成回答，并且在模板里面引用相关数据，具体数据用参考数据里面的数据。"""

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

        #读取template文件 TODO
        try:
            template = self._read_template_file()
            print(f"成功读取模板文件: {self.config.ANSWER_TEMPLATE}")
            print("文件内容为:",template)
        except FileNotFoundError:
            print(f"模板文件不存在: {self.config.ANSWER_TEMPLATE}")
            template = "# 默认模板\n\n这是一个默认的回答模板。"
        except Exception as e:
            print(f"读取模板文件时出错: {e}")
            template = "# 错误\n\n无法加载模板文件。"

        """当没有本地上下文时，使用LLM的一般知识回答"""
        prompt = f"""你是一个社会调研专家。请回答以下关于社会调研专家的问题。

用户问题：{query}
使用模板:{template}
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

    def analyze_knowledge_base(self, knowledge_base_path: str = None) -> Dict[str, Any]:
        """分析知识库"""
        try:
            # 验证输入路径
            if not knowledge_base_path:
                return {"status": "error", "message": "知识库路径不能为空"}

            # 正确处理路径（移除可能的重复..）
            clean_path = os.path.normpath(knowledge_base_path.replace("..", ""))

            # 检查路径是否存在
            if not os.path.exists(clean_path):
                return {"status": "error", "message": f"知识库路径不存在: {clean_path}"}

            # 创建分析器实例
            analyzer = KnowledgeBaseAnalyzer()

            # 先进行分析
            print("开始分析知识库...")
            stats = analyzer.analyze_knowledge_base(clean_path)

            # 检查是否有错误
            if "error" in stats:
                error_msg = f"分析失败: {stats['error']}"
                print(error_msg)
                return {"status": "error", "message": error_msg}
            else:
                print("分析完成！")

                # 获取统计报告
                report = analyzer.get_statistics_report()
                print(report)

                # 确保返回完整的报告信息
                return {
                    "status": "success",
                    "message": "知识库分析完成",
                    "data": report,
                    "stats": stats
                }

        except Exception as e:
            # 捕获任何未预期的异常
            error_msg = f"分析过程中发生未预期错误: {str(e)}"
            print(error_msg)
            return {"status": "error", "message": error_msg}

    def ask_question(self, question: str, knowledge_base_path: str = None) -> Dict[str, Any]:

        report = self.analyze_knowledge_base(knowledge_base_path)

        """提问并获取答案，可指定知识库路径"""
        # 如果指定了知识库路径且与当前加载的不同，重新构建知识库
        if knowledge_base_path and knowledge_base_path != getattr(self, 'current_knowledge_base', None):
            try:
                print(f"切换到知识库: {knowledge_base_path}")
                # 使用build_knowledge_base来构建指定路径的知识库
                processed_count = self.build_knowledge_base(knowledge_base_path)
                if processed_count > 0:
                    self.current_knowledge_base = knowledge_base_path
                    print(f"成功切换到知识库: {knowledge_base_path}")
                else:
                    print(f"警告: 知识库 {knowledge_base_path} 没有文档或构建失败，使用当前知识库")
            except Exception as e:
                print(f"切换知识库失败: {e}")
                # 继续使用当前知识库

        # 搜索相关文档
        similar_docs = self.search_similar_documents(question)

        if similar_docs:
            # 有相关文档，基于上下文生成答案
            answer = self.generate_answer_with_context(question, similar_docs, report)
            confidence = sum(doc["similarity"] for doc in similar_docs) / len(similar_docs)

            return {
                "answer": answer,
                "sources": [{"content": doc["content"][:200] + "...", "similarity": doc["similarity"]} for doc in
                            similar_docs],
                "confidence": confidence,
                "source_type": "knowledge_base",
                "knowledge_base_used": getattr(self, 'current_knowledge_base', self.default_knowledge_base)
            }
        else:
            # 没有相关文档，使用LLM的一般知识回答
            answer = self.generate_answer_without_context(question)
            web_info = self.search_web_knowledge(question)

            return {
                "answer": f"{answer}\n\n{web_info}",
                "sources": [],
                "confidence": 0.3,
                "source_type": "general_knowledge",
                "knowledge_base_used": getattr(self, 'current_knowledge_base', self.default_knowledge_base)
            }

    def switch_knowledge_base(self, knowledge_base_path: str) -> Dict[str, Any]:
        """切换知识库"""
        try:
            if not os.path.exists(knowledge_base_path):
                return {"success": False, "message": f"知识库路径不存在: {knowledge_base_path}"}

            processed_count = self.build_knowledge_base(knowledge_base_path)
            if processed_count > 0:
                self.current_knowledge_base = knowledge_base_path
                return {"success": True, "message": f"成功切换到知识库: {knowledge_base_path}",
                        "documents_processed": processed_count}
            else:
                return {"success": False, "message": f"知识库 {knowledge_base_path} 没有文档或构建失败"}

        except Exception as e:
            return {"success": False, "message": f"切换知识库时出错: {str(e)}"}

    def get_current_knowledge_base(self) -> str:
        """获取当前使用的知识库路径"""
        return getattr(self, 'current_knowledge_base', self.default_knowledge_base)

    def list_available_knowledge_bases(self, base_directory: str = None) -> List[str]:
        """列出可用的知识库（目录）"""
        if base_directory is None:
            base_directory = os.path.dirname(self.default_knowledge_base)

        available_bases = []

        if os.path.exists(base_directory):
            for item in os.listdir(base_directory):
                item_path = os.path.join(base_directory, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # 检查目录是否包含文档文件
                    has_documents = any(
                        not f.startswith('.') and os.path.isfile(os.path.join(item_path, f))
                        for f in os.listdir(item_path)
                    )
                    if has_documents:
                        available_bases.append(item_path)

        return available_bases

    def _read_template_file(self):
        """读取模板文件，保持原格式处理"""
        try:
            file_ext = os.path.splitext(self.config.ANSWER_TEMPLATE.lower())[1]
            print(f"读取模板文件: {self.config.ANSWER_TEMPLATE} (格式: {file_ext})")

            # 根据文件类型使用对应读取方法
            if file_ext in ('.txt', '.md', '.rst', '.markdown'):
                with open(self.config.ANSWER_TEMPLATE, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_ext == '.csv':
                return self.processor._load_csv_file(self.config.ANSWER_TEMPLATE)
            elif file_ext in ('.xlsx', '.xls'):
                return self.processor._load_excel_file(self.config.ANSWER_TEMPLATE)
            elif file_ext == '.docx':
                return self.processor._load_word_file(self.config.ANSWER_TEMPLATE)
            elif file_ext == '.pdf':
                return self.processor._load_pdf_file(self.config.ANSWER_TEMPLATE)
            else:
                raise ValueError(f"不支持的模板文件格式: {file_ext}")

        except FileNotFoundError:
            print(f"模板文件不存在: {self.config.ANSWER_TEMPLATE}")
            return "# 默认模板\n\n这是一个默认的回答模板。"
        except Exception as e:
            print(f"读取模板文件时出错: {e}")
            return "# 错误\n\n无法加载模板文件。"

    def stream_answer_sse(self, question: str, knowledge_base_path: str = None):
        """流式生成答案，直接产出 SSE 帧字符串（data:...\n\n）。

        说明：
        - 优先尝试底层 LLM 的原生流（token 级别），逐 token 推送。
        - 若失败，则回退为一次性生成完整答案并按块切分模拟流。
        - 末尾补充 sources / confidence / knowledge_base_used 等元信息。
        """
        # 起始帧
        yield f"data:{json.dumps({'type': 'start'})}\n\n"

        # 优先尝试原生流
        try:
            # 准备上下文与提示词，尽量与非流式逻辑保持一致
            similar_docs = self.search_similar_documents(question)
            report = self.analyze_knowledge_base(knowledge_base_path)
            if similar_docs:
                context_text = "\n\n".join([
                    f"相关文档 {i + 1} (相似度: {doc['similarity']:.2f}):\n{doc['content']}"
                    for i, doc in enumerate(similar_docs)
                ])
                try:
                    template = self._read_template_file()
                except Exception:
                    template = "# 默认模板\n\n这是一个默认的回答模板。"
                prompt = (
                    "你是一个社会调研专家。基于以下相关知识，请回答用户的问题。\n\n"
                    f"相关背景知识：\n{context_text}\n"
                    "套用模板：\n"
                    f"{template}\n"
                    f"用户问题：{question}\n"
                    f"数据参考：{report}\n"
                    "使用模板结合背景知识来生成回答,要在回答里面放入具体数据，具体数据中的数据可以套用,一定要保证数据的真实性。"
                )
            else:
                try:
                    template = self._read_template_file()
                except Exception:
                    template = "# 默认模板\n\n这是一个默认的回答模板。"
                prompt = (
                    "你是一个社会调研专家。请回答以下关于社会调研专家的问题。\n\n"
                    f"用户问题：{question}\n"
                    f"使用模板:{template}\n"
                    "请基于你的专业知识提供准确、专业的回答。如果你是推测或不确定，请说明。"
                )

            messages = [
                {"role": "system", "content": "你是一个社会调研专家。"},
                {"role": "user", "content": prompt}
            ]

            for token in self.llm_provider.stream_response(messages):
                if token:
                    frame = {"type": "chunk", "content": token}
                    yield f"data:{json.dumps(frame, ensure_ascii=False)}\n\n"

            # 末尾补充 meta 信息（基于非流式最终结果）
            final_result = self.ask_question(question, knowledge_base_path)
            meta = {
                "type": "end",
                "sources": final_result.get("sources", []),
                "knowledge_base_used": final_result.get("knowledge_base_used", ""),
                "confidence": final_result.get("confidence", 0)
            }
            yield f"data:{json.dumps(meta, ensure_ascii=False)}\n\n"
            return
        except Exception:
            # 原生流失败时回退到模拟流
            pass

        # 回退：一次性生成并切块输出
        result = self.ask_question(question, knowledge_base_path)
        answer_text = result.get("answer", "")
        chunk_size = 120
        for i in range(0, len(answer_text), chunk_size):
            chunk = answer_text[i:i + chunk_size]
            frame = {"type": "chunk", "content": chunk}
            yield f"data:{json.dumps(frame, ensure_ascii=False)}\n\n"

        meta = {
            "type": "end",
            "sources": result.get("sources", []),
            "knowledge_base_used": result.get("knowledge_base_used", ""),
            "confidence": result.get("confidence", 0)
        }
        yield f"data:{json.dumps(meta, ensure_ascii=False)}\n\n"

    def create_session(self, user_id: str = "anonymous", knowledge_base_path: str = None,
                       title: str = "新对话") -> str:
        """创建新的对话会话"""
        return self.db_manager.create_session(user_id, knowledge_base_path, title)

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """从数据库获取对话历史"""
        return self.db_manager.get_conversation_history(session_id, limit)

    def stream_answer_sse(self, question: str, knowledge_base_path: str = None,
                          session_id: str = None) -> str:
        """流式生成答案，支持多轮对话，使用数据库存储历史"""

        # 获取对话历史
        conversation_history = []
        if session_id:
            conversation_history = self.get_conversation_history(session_id)

        # 起始帧
        yield f"data:{json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

        # 构建当前对话上下文
        similar_docs = self.search_similar_documents(question)
        report = self.analyze_knowledge_base(knowledge_base_path)

        # 准备系统提示词和上下文
        try:
            template = self._read_template_file()
        except Exception:
            template = "# 默认模板\n\n这是一个默认的回答模板。"

        # 构建基础系统提示
        system_prompt = """你是一个社会调研专家。请基于以下上下文进行对话："""

        # 如果有相关文档，添加上下文
        context_text = ""
        if similar_docs:
            context_text = "\n\n".join([
                f"相关文档 {i + 1} (相似度: {doc['similarity']:.2f}):\n{doc['content']}"
                for i, doc in enumerate(similar_docs)
            ])
            system_prompt += f"""

相关背景知识：
{context_text}

套用模板：
{template}

数据参考：
{report}

请使用模板结合背景知识来生成回答，在回答中放入具体数据，保证数据的真实性。"""

        # 构建完整的消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 添加对话历史（确保不超过token限制）
        max_history_length = 6  # 限制历史对话轮数
        recent_history = conversation_history[-max_history_length * 2:]

        # 添加历史对话到消息中
        for msg in recent_history:
            messages.append(msg)

        # 添加当前问题
        messages.append({"role": "user", "content": question})

        # 记录知识库使用情况
        if session_id and similar_docs:
            avg_similarity = sum(doc["similarity"] for doc in similar_docs) / len(similar_docs)
            self.db_manager.record_knowledge_base_usage(
                session_id,
                knowledge_base_path or self.get_current_knowledge_base(),
                question,
                len(similar_docs),
                avg_similarity
            )

        # 优先尝试原生流式响应
        full_response = ""
        try:
            for token in self.llm_provider.stream_response(messages):
                if token:
                    full_response += token
                    frame = {"type": "chunk", "content": token}
                    yield f"data:{json.dumps(frame, ensure_ascii=False)}\n\n"

            # 保存消息到数据库
            if session_id:
                # 保存用户问题
                self.db_manager.add_message(
                    session_id,
                    "user",
                    question,
                    {"knowledge_base": knowledge_base_path, "timestamp": datetime.now().isoformat()}
                )
                # 保存助手回答
                self.db_manager.add_message(
                    session_id,
                    "assistant",
                    full_response,
                    {"sources_count": len(similar_docs), "timestamp": datetime.now().isoformat()}
                )

                # 更新会话标题（如果这是第一轮对话）
                if len(conversation_history) == 0:
                    # 使用问题前20个字符作为标题
                    title = question[:20] + "..." if len(question) > 20 else question
                    self.db_manager.update_session_title(session_id, title)

            # 末尾补充 meta 信息
            meta = {
                "type": "end",
                "session_id": session_id,
                "sources": [{"content": doc["content"][:200] + "...", "similarity": doc["similarity"]}
                            for doc in similar_docs] if similar_docs else [],
                "knowledge_base_used": knowledge_base_path or self.get_current_knowledge_base(),
                "confidence": sum(doc["similarity"] for doc in similar_docs) / len(
                    similar_docs) if similar_docs else 0.3,
                "conversation_turn": len(conversation_history) // 2 + 1
            }
            yield f"data:{json.dumps(meta, ensure_ascii=False)}\n\n"

        except Exception as e:
            # 原生流失败时回退到模拟流
            print(f"流式响应失败，回退到模拟流: {e}")

            # 一次性生成完整答案
            result = self.ask_question(question, knowledge_base_path)
            answer_text = result.get("answer", "")
            full_response = answer_text

            # 模拟流式输出
            chunk_size = 120
            for i in range(0, len(answer_text), chunk_size):
                chunk = answer_text[i:i + chunk_size]
                frame = {"type": "chunk", "content": chunk}
                yield f"data:{json.dumps(frame, ensure_ascii=False)}\n\n"

            # 保存消息到数据库
            if session_id:
                self.db_manager.add_message(session_id, "user", question)
                self.db_manager.add_message(session_id, "assistant", full_response)

            meta = {
                "type": "end",
                "session_id": session_id,
                "sources": result.get("sources", []),
                "knowledge_base_used": result.get("knowledge_base_used", ""),
                "confidence": result.get("confidence", 0),
                "conversation_turn": len(conversation_history) // 2 + 1
            }
            yield f"data:{json.dumps(meta, ensure_ascii=False)}\n\n"

    def get_user_sessions(self, user_id: str = "anonymous", limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户的会话列表"""
        return self.db_manager.get_user_sessions(user_id, limit)

    def close_session(self, session_id: str):
        """关闭会话"""
        self.db_manager.close_session(session_id)

    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指定会话的详细消息列表"""
        conn = self.db_manager.get_connection()
        try:
            with conn.cursor(dictionary=True) as cursor:
                sql = """
                SELECT role, content, metadata, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                LIMIT %s
                """
                cursor.execute(sql, (session_id, limit))
                messages = cursor.fetchall()

                # 格式化返回数据
                formatted_messages = []
                for msg in messages:
                    formatted_msg = {
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["created_at"].isoformat() if msg["created_at"] else None
                    }
                    if msg["metadata"]:
                        formatted_msg["metadata"] = json.loads(msg["metadata"])
                    formatted_messages.append(formatted_msg)

                return formatted_messages
        except Exception as e:
            print(f"获取会话消息失败: {e}")
            return []
        finally:
            conn.close()

    # 原有的其他方法保持不变...
    def _ensure_directories_exist(self):
        """确保必要的目录存在"""
        os.makedirs(self.config.CHROMA_DB_PATH, exist_ok=True)
        os.makedirs(self.config.KNOWLEDGE_BASE_PATH, exist_ok=True)

    def _init_chroma_client(self):
        """初始化 ChromaDB 客户端，处理设置冲突"""
        try:
            client = chromadb.PersistentClient(
                path=self.config.CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            return client
        except ValueError as e:
            if "different settings" in str(e):
                print("检测到 ChromaDB 设置冲突，正在清理并重新创建...")
                if os.path.exists(self.config.CHROMA_DB_PATH):
                    shutil.rmtree(self.config.CHROMA_DB_PATH)
                    os.makedirs(self.config.CHROMA_DB_PATH, exist_ok=True)

                client = chromadb.PersistentClient(
                    path=self.config.CHROMA_DB_PATH,
                    settings=Settings(anonymized_telemetry=False)
                )
                return client
            else:
                raise e