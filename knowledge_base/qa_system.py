import os
import shutil
from datetime import datetime

import chromadb
from typing import List, Dict, Any
import re
import html

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

    def _bing_search(self, query: str) -> List[Dict[str, str]]:
        """使用必应国内版进行搜索并解析前若干结果（无额外第三方依赖）。"""
        if not getattr(self.config, 'WEB_SEARCH_ENABLED', True):
            return []
        try:
            params = {
                'q': query,
                'mkt': getattr(self.config, 'WEB_SEARCH_MKT', 'zh-CN')
            }
            url = getattr(self.config, 'WEB_SEARCH_ENGINE_URL', 'https://cn.bing.com/search')
            timeout = getattr(self.config, 'WEB_SEARCH_TIMEOUT', 8)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36'
            }
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            text = resp.text
            results = []
            
            print(f"正在搜索: {query}")  # 添加调试信息
            
            # 模式1：查找h2标签中的链接
            for m in re.finditer(r'<h2[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>([^<]+)</a>.*?</h2>', text, re.DOTALL):
                link = html.unescape(m.group(1).strip())
                title = html.unescape(m.group(2).strip())
                
                # 过滤掉必应内部链接和无效链接
                if (link.startswith('http') and 
                    'bing.com' not in link and 
                    title and 
                    len(title) > 3):
                    results.append({'title': title, 'snippet': '', 'link': link})
                    if len(results) >= getattr(self.config, 'WEB_SEARCH_TOPN', 3):
                        break
            
            # 模式1.5：查找更复杂的h2结构
            if not results:
                for m in re.finditer(r'<h2[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>([^<]+)</a>.*?</h2>', text, re.DOTALL):
                    link = html.unescape(m.group(1).strip())
                    title = html.unescape(m.group(2).strip())
                    
                    # 过滤掉必应内部链接和无效链接
                    if (link.startswith('http') and 
                        'bing.com' not in link and 
                        title and 
                        len(title) > 3):
                        results.append({'title': title, 'snippet': '', 'link': link})
                        if len(results) >= getattr(self.config, 'WEB_SEARCH_TOPN', 3):
                            break
            
            # 如果新模式没有结果，尝试原来的模式
            if not results:
                # 解析常见结果块：<li class="b_algo"> ...
                for m in re.finditer(r'<li class=\"b_algo\"[\s\S]*?<h2>[\s\S]*?<a href=\"(.*?)\"[^>]*>([\s\S]*?)</a>[\s\S]*?</h2>[\s\S]*?(?:<p[^>]*>([\s\S]*?)</p>)?', text):
                    link = html.unescape(re.sub(r'\s+', ' ', m.group(1) or '').strip())
                    title_raw = re.sub(r'<.*?>', '', m.group(2) or '')
                    title = html.unescape(re.sub(r'\s+', ' ', title_raw).strip())
                    snippet_raw = re.sub(r'<.*?>', '', m.group(3) or '')
                    snippet = html.unescape(re.sub(r'\s+', ' ', snippet_raw).strip())
                    if link and title:
                        results.append({'title': title, 'snippet': snippet, 'link': link})
                    if len(results) >= getattr(self.config, 'WEB_SEARCH_TOPN', 3):
                        break
                        
            # 回退：若未匹配到b_algo，尝试通用a标签解析
            if not results:
                for m in re.finditer(r'<a href=\"(https?://[^\"]+)\"[^>]*>([\s\S]*?)</a>', text):
                    link = html.unescape(m.group(1))
                    title = html.unescape(re.sub(r'<.*?>', '', m.group(2) or '').strip())
                    if 'bing.com' in link:
                        continue
                    if title:
                        results.append({'title': title, 'snippet': '', 'link': link})
                    if len(results) >= getattr(self.config, 'WEB_SEARCH_TOPN', 3):
                        break
            
            print(f"搜索完成，获得 {len(results)} 个结果")  # 添加调试信息
            return results
        except Exception as e:
            print(f"搜索异常: {e}")  # 添加调试信息
            return []

    def _summarize_web_results(self, results: List[Dict[str, str]]) -> str:
        """将搜索结果压缩为简短中文摘要，并列出可引用的关键信息。"""
        if not results:
            return "未检索到可用的网络资料。"
        bullets = []
        for i, r in enumerate(results, 1):
            title = r.get('title', '')
            snippet = r.get('snippet', '')
            link = r.get('link', '')
            piece = f"{i}. {title} — {snippet}"
            bullets.append(piece.strip(' —'))
        # 简要合成说明
        summary = "\n".join(bullets)
        return summary

    def _build_search_query(self, question: str, context_text: str, template_text: str) -> str:
        """构建更精准的搜索查询"""
        # 1. 基础问题处理
        base_query = question.strip()
        
        # 2. 分析问题类型
        question_type = self._analyze_question_type(base_query)
        
        # 3. 提取核心概念（更智能的方法）
        core_concepts = self._extract_core_concepts(base_query)
        
        # 4. 构建搜索查询
        if core_concepts:
            # 使用核心概念构建查询
            if question_type == "definition":
                # 定义类问题：使用更具体的查询
                search_query = f"{core_concepts[0]} 是什么 含义 概念"
            elif question_type == "how_to":
                # 方法类问题：使用核心概念 + "方法"
                search_query = f"{core_concepts[0]} 方法 步骤 如何"
            elif question_type == "why":
                # 原因类问题：使用核心概念 + "原因"
                search_query = f"{core_concepts[0]} 原因 影响 为什么"
            elif question_type == "comparison":
                # 比较类问题：使用所有核心概念
                search_query = " ".join(core_concepts[:2])
            else:
                # 默认：使用最重要的核心概念 + 相关词汇
                search_query = f"{core_concepts[0]} 法律 规定 条款"
        else:
            # 如果没有提取到核心概念，使用简化的问题
            search_query = self._simplify_question(base_query)
        
        # 5. 限制查询长度
        if len(search_query) > 50:
            search_query = search_query[:50]
        
        print(f"问题类型: {question_type}, 核心概念: {core_concepts}, 搜索查询: {search_query}")
        return search_query
    
    def _extract_core_concepts(self, question: str) -> List[str]:
        """提取问题的核心概念"""
        concepts = []
        
        # 1. 移除问句标记和停用词
        cleaned_question = question.replace("？", "").replace("?", "")
        
        # 2. 定义法律专业术语词典
        legal_terms = {
            # 基础法律概念
            "法律": "法律",
            "法治": "法治",
            "宪法": "宪法",
            "民法": "民法",
            "刑法": "刑法",
            "行政法": "行政法",
            "商法": "商法",
            "劳动法": "劳动法",
            "知识产权法": "知识产权法",
            "环境法": "环境法",
            "国际法": "国际法",
            
            # 案件相关
            "案件": "案件",
            "诉讼": "诉讼",
            "起诉": "起诉",
            "上诉": "上诉",
            "申诉": "申诉",
            "仲裁": "仲裁",
            "调解": "调解",
            "判决": "判决",
            "裁定": "裁定",
            "执行": "执行",
            "强制执行": "强制执行",
            
            # 法律主体
            "法院": "法院",
            "检察院": "检察院",
            "公安机关": "公安机关",
            "律师": "律师",
            "法官": "法官",
            "检察官": "检察官",
            "当事人": "当事人",
            "原告": "原告",
            "被告": "被告",
            "第三人": "第三人",
            "代理人": "代理人",
            "辩护人": "辩护人",
            
            # 法律程序
            "立案": "立案",
            "审理": "审理",
            "开庭": "开庭",
            "举证": "举证",
            "质证": "质证",
            "辩论": "辩论",
            "合议": "合议",
            "宣判": "宣判",
            "送达": "送达",
            "保全": "保全",
            "先予执行": "先予执行",
            
            # 法律责任
            "民事责任": "民事责任",
            "刑事责任": "刑事责任",
            "行政责任": "行政责任",
            "违约责任": "违约责任",
            "侵权责任": "侵权责任",
            "赔偿": "赔偿",
            "补偿": "补偿",
            "罚款": "罚款",
            "拘留": "拘留",
            "有期徒刑": "有期徒刑",
            "无期徒刑": "无期徒刑",
            "死刑": "死刑",
            
            # 法律权利
            "权利": "权利",
            "义务": "义务",
            "人身权": "人身权",
            "财产权": "财产权",
            "知识产权": "知识产权",
            "继承权": "继承权",
            "监护权": "监护权",
            "抚养权": "抚养权",
            "探视权": "探视权",
            "名誉权": "名誉权",
            "隐私权": "隐私权",
            "肖像权": "肖像权",
            
            # 合同相关
            "合同": "合同",
            "协议": "协议",
            "契约": "契约",
            "要约": "要约",
            "承诺": "承诺",
            "履行": "履行",
            "违约": "违约",
            "解除": "解除",
            "终止": "终止",
            "变更": "变更",
            "转让": "转让",
            
            # 婚姻家庭
            "婚姻": "婚姻",
            "结婚": "结婚",
            "离婚": "离婚",
            "夫妻": "夫妻",
            "家庭": "家庭",
            "子女": "子女",
            "父母": "父母",
            "配偶": "配偶",
            "夫妻共同财产": "夫妻共同财产",
            "婚前财产": "婚前财产",
            "婚后财产": "婚后财产",
            
            # 公司企业
            "公司": "公司",
            "企业": "企业",
            "法人": "法人",
            "股东": "股东",
            "董事会": "董事会",
            "监事会": "监事会",
            "股东大会": "股东大会",
            "公司章程": "公司章程",
            "注册资本": "注册资本",
            "股权": "股权",
            "股份": "股份",
            "上市": "上市",
            
            # 金融法律
            "银行": "银行",
            "贷款": "贷款",
            "担保": "担保",
            "抵押": "抵押",
            "质押": "质押",
            "保证": "保证",
            "保险": "保险",
            "证券": "证券",
            "基金": "基金",
            "投资": "投资",
            "融资": "融资",
            
            # 劳动法律
            "劳动合同": "劳动合同",
            "工资": "工资",
            "加班": "加班",
            "休假": "休假",
            "社保": "社保",
            "公积金": "公积金",
            "工伤": "工伤",
            "职业病": "职业病",
            "解雇": "解雇",
            "辞职": "辞职",
            "经济补偿": "经济补偿",
            
            # 房地产
            "房地产": "房地产",
            "房屋": "房屋",
            "土地": "土地",
            "产权": "产权",
            "使用权": "使用权",
            "所有权": "所有权",
            "租赁": "租赁",
            "买卖": "买卖",
            "过户": "过户",
            "登记": "登记",
            "抵押贷款": "抵押贷款",
            
            # 刑事法律
            "犯罪": "犯罪",
            "罪名": "罪名",
            "量刑": "量刑",
            "缓刑": "缓刑",
            "假释": "假释",
            "减刑": "减刑",
            "自首": "自首",
            "立功": "立功",
            "累犯": "累犯",
            "共犯": "共犯",
            "主犯": "主犯",
            "从犯": "从犯",
            
            # 行政法律
            "行政处罚": "行政处罚",
            "行政许可": "行政许可",
            "行政复议": "行政复议",
            "行政诉讼": "行政诉讼",
            "行政强制": "行政强制",
            "行政监督": "行政监督",
            "政府": "政府",
            "行政机关": "行政机关",
            "公务员": "公务员",
            "公职人员": "公职人员"
        }
        
        # 3. 查找专业术语
        for term, standard_term in legal_terms.items():
            if term in cleaned_question:
                concepts.append(standard_term)
        
        # 4. 如果没有找到专业术语，提取关键词
        if not concepts:
            # 提取2-6字的中文词汇
            import re
            words = re.findall(r'[\u4e00-\u9fff]{2,6}', cleaned_question)
            
            # 过滤停用词
            stop_words = {'什么', '如何', '怎么', '为什么', '哪个', '哪些', '是', '的', '了', '在', '有', '和', '与', '或', '但', '然而', '因此', '所以', '因为', '如果', '当', '就', '都', '很', '非常', '比较', '更', '最', '还', '也', '又', '再', '已经', '正在', '将要', '可以', '能够', '应该', '必须', '需要', '要求', '希望', '想要', '喜欢', '不喜欢', '认为', '觉得', '知道', '了解', '明白', '理解', '学习', '研究', '分析', '讨论', '介绍', '说明', '解释', '描述', '总结', '概括', '区别', '差异', '比较', '对比', '应用', '用途', '作用', '影响', '意义', '价值', '重要性', '特点', '优势', '劣势', '优点', '缺点', '好处', '坏处', '风险', '机会', '前景', '未来', '现在', '过去', '历史', '现状', '情况', '状态', '水平', '程度', '范围', '领域', '行业', '市场', '经济', '社会', '政治', '文化', '教育', '科技', '医疗', '健康', '环境', '能源', '交通', '通信', '金融', '投资', '管理', '运营', '生产', '销售', '服务', '客户', '用户', '消费者', '企业', '公司', '组织', '机构', '政府', '部门', '单位', '团队', '个人', '专家', '学者', '研究人员', '分析师', '顾问', '咨询师', '工程师', '设计师', '开发者', '程序员', '产品经理', '项目经理', '销售经理', '市场经理', '人力资源', '财务', '会计', '法律', '律师', '医生', '护士', '教师', '学生', '家长', '孩子', '老人', '年轻人', '男性', '女性', '城市', '农村', '地区', '国家', '国际', '全球', '世界', '中国', '美国', '欧洲', '亚洲', '非洲', '南美洲', '北美洲', '大洋洲', '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '西安', '重庆', '天津', '青岛', '大连', '厦门', '苏州', '无锡', '宁波', '温州', '佛山', '东莞', '中山', '珠海', '江门', '肇庆', '惠州', '汕头', '湛江', '茂名', '韶关', '清远', '阳江', '河源', '梅州', '汕尾', '潮州', '揭阳', '云浮', '广西', '海南', '云南', '贵州', '四川', '重庆', '西藏', '新疆', '青海', '甘肃', '宁夏', '内蒙古', '黑龙江', '吉林', '辽宁', '河北', '山西', '陕西', '河南', '山东', '江苏', '安徽', '浙江', '福建', '江西', '湖南', '湖北', '广东', '台湾', '香港', '澳门'}
            
            for word in words:
                if word not in stop_words and len(word) >= 2:
                    concepts.append(word)
        
        # 5. 去重并按长度排序
        unique_concepts = []
        seen = set()
        for concept in concepts:
            if concept not in seen:
                unique_concepts.append(concept)
                seen.add(concept)
        
        # 按长度排序，优先选择较长的概念
        unique_concepts.sort(key=len, reverse=True)
        
        return unique_concepts[:3]  # 最多返回3个核心概念
    
    def _simplify_question(self, question: str) -> str:
        """简化问题，提取核心内容"""
        # 移除问句标记
        simplified = question.replace("？", "").replace("?", "")
        
        # 移除常见的问句开头
        question_starters = ["什么是", "如何", "怎么", "为什么", "哪个", "哪些", "什么时候", "哪里", "多少", "几个"]
        for starter in question_starters:
            if simplified.startswith(starter):
                simplified = simplified[len(starter):]
                break
        
        # 限制长度
        if len(simplified) > 20:
            simplified = simplified[:20]
        
        return simplified.strip()
    
    def _analyze_question_type(self, question: str) -> str:
        """分析问题类型"""
        question_lower = question.lower()
        
        # 定义类问题
        if any(word in question for word in ['是什么', '什么是', '定义', '含义', '概念']):
            return "definition"
        
        # 比较类问题
        elif any(word in question for word in ['比较', '对比', '区别', '差异', '哪个更好', '哪个更']):
            return "comparison"
        
        # 方法类问题
        elif any(word in question for word in ['如何', '怎么', '怎样', '方法', '步骤', '怎么做']):
            return "how_to"
        
        # 原因类问题
        elif any(word in question for word in ['为什么', '原因', '为什么', '为何', '导致']):
            return "why"
        
        # 时间类问题
        elif any(word in question for word in ['什么时候', '何时', '时间', '历史', '发展']):
            return "when"
        
        # 地点类问题
        elif any(word in question for word in ['哪里', '在哪', '地点', '位置', '地方']):
            return "where"
        
        # 数量类问题
        elif any(word in question for word in ['多少', '几个', '数量', '规模', '比例']):
            return "quantity"
        
        else:
            return "general"
    
    def _extract_keywords_from_question(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        keywords = []
        
        # 移除常见的停用词
        stop_words = {'什么', '如何', '怎么', '为什么', '哪个', '哪些', '是', '的', '了', '在', '有', '和', '与', '或', '但', '然而', '因此', '所以', '因为', '如果', '当', '就', '都', '很', '非常', '比较', '更', '最', '还', '也', '又', '再', '已经', '正在', '将要', '可以', '能够', '应该', '必须', '需要', '要求', '希望', '想要', '喜欢', '不喜欢', '认为', '觉得', '知道', '了解', '明白', '理解', '学习', '研究', '分析', '讨论', '介绍', '说明', '解释', '描述', '总结', '概括', '？', '?', '，', ',', '。', '.', '！', '!', '：', ':', '；', ';'}
        
        # 更智能的关键词提取
        # 1. 提取2-4字的中文词汇
        words_2_4 = re.findall(r'[\u4e00-\u9fff]{2,4}', question)
        for word in words_2_4:
            if word not in stop_words:
                keywords.append(word)
        
        # 2. 提取专业术语（5字以上的词）
        long_words = re.findall(r'[\u4e00-\u9fff]{5,}', question)
        for word in long_words:
            if word not in stop_words:
                keywords.append(word)
        
        # 3. 按长度和重要性排序
        keywords.sort(key=len, reverse=True)
        
        # 4. 去重
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:5]  # 最多返回5个关键词
    
    def _extract_keywords_from_context(self, context_text: str) -> List[str]:
        """从上下文中提取相关关键词"""
        keywords = []
        
        # 提取前200个字符中的关键词
        text_sample = context_text[:200]
        words = re.findall(r'[\u4e00-\u9fff]+', text_sample)
        
        # 过滤和排序
        stop_words = {'相关', '文档', '相似度', '内容', '信息', '数据', '资料', '报告', '分析', '研究', '调查', '统计', '结果', '结论', '建议', '方法', '技术', '系统', '应用', '发展', '趋势', '问题', '挑战', '机遇', '影响', '作用', '意义', '价值', '重要性', '特点', '优势', '劣势', '优点', '缺点', '好处', '坏处', '风险', '机会', '前景', '未来', '现在', '过去', '历史', '现状', '情况', '状态', '水平', '程度', '范围', '领域', '行业', '市场', '经济', '社会', '政治', '文化', '教育', '科技', '医疗', '健康', '环境', '能源', '交通', '通信', '金融', '投资', '管理', '运营', '生产', '销售', '服务', '客户', '用户', '消费者', '企业', '公司', '组织', '机构', '政府', '部门', '单位', '团队', '个人', '专家', '学者', '研究人员', '分析师', '顾问', '咨询师', '工程师', '设计师', '开发者', '程序员', '产品经理', '项目经理', '销售经理', '市场经理', '人力资源', '财务', '会计', '法律', '律师', '医生', '护士', '教师', '学生', '家长', '孩子', '老人', '年轻人', '男性', '女性', '城市', '农村', '地区', '国家', '国际', '全球', '世界', '中国', '美国', '欧洲', '亚洲', '非洲', '南美洲', '北美洲', '大洋洲', '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '西安', '重庆', '天津', '青岛', '大连', '厦门', '苏州', '无锡', '宁波', '温州', '佛山', '东莞', '中山', '珠海', '江门', '肇庆', '惠州', '汕头', '湛江', '茂名', '韶关', '清远', '阳江', '河源', '梅州', '汕尾', '潮州', '揭阳', '云浮', '广西', '海南', '云南', '贵州', '四川', '重庆', '西藏', '新疆', '青海', '甘肃', '宁夏', '内蒙古', '黑龙江', '吉林', '辽宁', '河北', '山西', '陕西', '河南', '山东', '江苏', '安徽', '浙江', '福建', '江西', '湖南', '湖北', '广东', '台湾', '香港', '澳门'}
        
        for word in words:
            if len(word) >= 2 and word not in stop_words:
                keywords.append(word)
        
        # 按长度排序
        keywords.sort(key=len, reverse=True)
        
        return keywords[:3]  # 最多返回3个上下文关键词

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

        # 联网搜索并摘要
        web_summary = ""
        web_sources = []
        if getattr(self.config, 'WEB_SEARCH_ENABLED', True):
            search_q = self._build_search_query(query, context_text, template)
            web_results = self._bing_search(search_q)
            web_summary = self._summarize_web_results(web_results)
            web_sources = web_results

        # 构建提示词
        prompt = f"""你是一个社会调研专家。基于以下相关知识与联网检索摘要，请回答用户的问题。

相关背景知识：
{context_text}
套用模板：
{template}
参考数据：
{report}
联网检索摘要（必应国内版）：
{web_summary}
用户问题：{query}

使用模板结合背景知识与联网摘要来生成回答，并在模板里引用相关数据和事实，保证数据真实性，并在必要处给出来源编号。"""

        messages = [
            {"role": "system", "content": "你是一个社会调研专家。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_provider.generate_response(messages)
            # 将来源附在结尾，便于前端查看
            if web_sources:
                refs = "\n".join([f"[{i+1}] {item.get('title','')} ({item.get('link','')})" for i, item in enumerate(web_sources)])
                response = f"{response}\n\n参考链接:\n{refs}"
            return response
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def generate_answer_without_context(self, query: str) -> str:

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

        """当没有本地上下文时，使用LLM的一般知识回答，同时加入联网摘要"""
        web_summary = ""
        web_sources = []
        if getattr(self.config, 'WEB_SEARCH_ENABLED', True):
            search_q = self._build_search_query(query, "", template)
            web_results = self._bing_search(search_q)
            web_summary = self._summarize_web_results(web_results)
            web_sources = web_results

        prompt = f"""你是一个社会调研专家。请回答以下问题，并参考联网检索摘要：

用户问题：{query}
使用模板:{template}
联网检索摘要（必应国内版）：
{web_summary}
请基于你的专业知识与联网摘要提供准确、专业的回答；若为推测或不确定，请说明。"""

        messages = [
            {"role": "system", "content": "你是一个社会调研专家。"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_provider.generate_response(messages)
            if web_sources:
                refs = "\n".join([f"[{i+1}] {item.get('title','')} ({item.get('link','')})" for i, item in enumerate(web_sources)])
                response = f"{response}\n\n参考链接:\n{refs}"
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
        system_prompt = """你是一个社会调研专家。请结合以下数据生成一个司法社会调研报告："""

        # 执行网络搜索（无论是否有本地文档）
        web_summary = ""
        web_results = []
        if getattr(self.config, 'WEB_SEARCH_ENABLED', True):
            # 构建更精准的搜索查询
            if similar_docs:
                # 有本地文档时，结合上下文构建查询
                context_text = "\n\n".join([
                    f"相关文档 {i + 1} (相似度: {doc['similarity']:.2f}):\n{doc['content']}"
                    for i, doc in enumerate(similar_docs)
                ])
                search_q = self._build_search_query(question, context_text, template)
            else:
                # 没有本地文档时，使用问题本身构建查询
                search_q = self._build_search_query(question, "", template)

            web_results = self._bing_search(search_q)
            web_summary = self._summarize_web_results(web_results)

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

联网检索摘要（必应国内版）：
{web_summary}

请使用模板结合背景知识与联网摘要来生成回答，在回答中放入具体数据，保证数据的真实性。"""
        else:
            # 没有本地文档时，也要添加网络搜索摘要
            system_prompt += f"""

套用模板：
{template}

联网检索摘要（必应国内版）：
{web_summary}

请基于你的专业知识与联网摘要提供准确、专业的回答；若为推测或不确定，请说明。"""

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
                "conversation_turn": len(conversation_history) // 2 + 1,
                "web_sources": web_results  # 添加网络搜索来源
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
                "conversation_turn": len(conversation_history) // 2 + 1,
                "web_sources": web_results  # 添加网络搜索来源
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