import numpy as np
from typing import List, Dict, Any
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import Config  # 正确的导入方式


class DataProcessor:
    def __init__(self, model_name: str = None):
        if model_name is None:
            config = Config()
            self.model_name = config.EMBEDDING_MODEL
        else:
            self.model_name = model_name

        self.model = None
        print(f"初始化数据处理器，使用模型: {self.model_name}")

    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            print("正在加载嵌入模型...")
            self.model = SentenceTransformer(self.model_name)
            print("模型加载完成")

    def check_existing_documents(self, vector_store, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检查并向量库中已存在的文档，避免重复处理
        """
        unique_documents = []
        for doc in documents:
            # 基于文件路径或其他唯一标识符检查是否已存在
            try:
                existing = vector_store.get(
                    where={"source": doc["source"]}
                    # 移除 include=["documents"]，因为默认会返回 ids
                    # 如果确实需要 documents 内容，可以使用 include=["documents", "metadatas"]
                )

                if len(existing['ids']) == 0:
                    unique_documents.append(doc)
                else:
                    print(f"文档已存在，跳过: {doc['source']}")
            except Exception as e:
                print(f"检查文档存在性时出错: {e}")
                # 出错时仍然添加文档，避免遗漏
                unique_documents.append(doc)

        return unique_documents

    def load_documents(self, data_path: str) -> List[Dict[str, Any]]:
        """加载多种格式的文档"""
        documents = []

        if not os.path.exists(data_path):
            print(f"数据路径不存在: {data_path}")
            return documents

        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith(('.txt', '.md', '.rst')):
                        content = self._load_text_file(file_path)
                    elif file.endswith('.csv'):
                        content = self._load_csv_file(file_path)
                    elif file.endswith(('.xlsx', '.xls')):
                        content = self._load_excel_file(file_path)
                    elif file.endswith('.docx'):  # 添加Word文档支持
                        content = self._load_word_file(file_path)
                    elif file.endswith('.doc'):
                        content = self._load_doc_file(file_path)
                    else:
                        print(f"跳过不支持的文件格式: {file}")
                        continue

                    documents.append({
                        "content": content,
                        "source": file_path,
                        "type": os.path.splitext(file)[1]
                    })
                    print(f"成功加载文件: {file}")

                except Exception as e:
                    print(f"加载文件 {file_path} 时出错: {e}")

        print(f"共加载 {len(documents)} 个文档")
        return documents

    def _load_doc_file(self, file_path: str) -> str:
        """加载.doc文件并转换为文本"""
        try:
            # 尝试使用win32com读取.doc文件
            import win32com.client

            text = f"Word文档(.doc): {os.path.basename(file_path)}\n"
            text += "=" * 50 + "\n"

            # 创建Word应用程序对象
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False  # 不显示Word界面

            try:
                # 打开文档
                doc = word.Documents.Open(file_path)

                # 获取文档内容
                content = doc.Content.Text
                text += content

                # 关闭文档
                doc.Close()

            finally:
                # 退出Word应用程序
                word.Quit()

            return text

        except ImportError:
            # 如果win32com不可用，尝试使用antiword（需要安装antiword工具）
            try:
                import subprocess

                # 使用antiword提取文本内容
                result = subprocess.run(['antiword', file_path],
                                        capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    text = f"Word文档(.doc): {os.path.basename(file_path)}\n"
                    text += "=" * 50 + "\n"
                    text += result.stdout
                    return text
                else:
                    return f"使用antiword读取.doc文件失败: {result.stderr}"

            except (FileNotFoundError, subprocess.TimeoutExpired):
                return f"错误: 无法读取.doc文件 {file_path}。请确保已安装win32com或antiword"

        except Exception as e:
            return f"读取.doc文件 {file_path} 出错: {str(e)}"

    def _load_text_file(self, file_path: str) -> str:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_csv_file(self, file_path: str) -> str:
        """加载CSV文件并转换为文本"""
        try:
            df = pd.read_csv(file_path)
            # 将DataFrame转换为描述性文本
            text = f"CSV文件: {os.path.basename(file_path)}\n"
            text += f"列名: {', '.join(df.columns)}\n"
            text += f"行数: {len(df)}\n"
            text += f"前3行数据:\n{df.head(3).to_string()}\n"
            return text
        except Exception as e:
            return f"读取CSV文件出错: {str(e)}"

    def _load_word_file(self, file_path: str) -> str:
        """加载Word文档文件并转换为文本"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = f"Word文档: {os.path.basename(file_path)}\n"
            text += "=" * 50 + "\n"

            # 提取段落文本
            paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            text += "\n".join(paragraphs)

            # 可选：提取表格内容
            if doc.tables:
                text += "\n\n文档中的表格:\n"
                for i, table in enumerate(doc.tables):
                    text += f"\n表格 {i + 1}:\n"
                    for row in table.rows:
                        row_text = " | ".join([cell.text.strip() for cell in row.cells])
                        text += row_text + "\n"

            return text
        except ImportError:
            return f"错误: 未安装python-docx库，无法读取Word文档 {file_path}"
        except Exception as e:
            return f"读取Word文档 {file_path} 出错: {str(e)}"

    def _load_excel_file(self, file_path: str) -> str:
        """加载Excel文件并转换为文本"""
        try:
            df = pd.read_excel(file_path)
            text = f"Excel文件: {os.path.basename(file_path)}\n"
            text += f"列名: {', '.join(df.columns)}\n"
            text += f"行数: {len(df)}\n"
            text += f"前3行数据:\n{df.head(3).to_string()}\n"
            return text
        except Exception as e:
            return f"读取Excel文件出错: {str(e)}"

    def chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50) -> List[
        Dict[str, Any]]:
        """将文档分块"""
        chunks = []

        for doc in documents:
            content = doc["content"]
            source = doc["source"]

            # 简单的文本分块
            words = content.split()
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "content": chunk_text,
                    "source": source,
                    "chunk_index": len(chunks)
                })

        print(f"文档分块完成，共 {len(chunks)} 个块")
        return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """生成文本嵌入向量"""
        self._load_model()  # 确保模型已加载

        print(f"正在为 {len(texts)} 个文本生成嵌入...")
        embeddings = self.model.encode(texts)
        print("嵌入生成完成")
        return embeddings