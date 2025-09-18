import os
from dotenv import load_dotenv
import json

load_dotenv()

class Config:
    # 模型配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai")

    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # 通义千问配置
    TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
    TONGYI_API_BASE = os.getenv("TONGYI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    TONGYI_MODEL = os.getenv("TONGYI_MODEL", "qwen-max")

    # Azure配置
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_BASE = os.getenv("AZURE_API_BASE")
    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")

    # 路径配置 - 修复路径组合
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else os.getcwd()
    DATA_PATH = os.getenv("DATA_PATH", "data")
    KNOWLEDGE_BASE_PATH = os.path.join(DATA_PATH, os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base"))
    CHROMA_DB_PATH = os.path.join(DATA_PATH, os.getenv("CHROMA_DB_PATH", "chroma_db"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")  # 新增集合名称配置

    # 分块配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # 搜索配置
    SEARCH_RESULTS_COUNT = int(os.getenv("SEARCH_RESULTS_COUNT", 3))
    TOP_K = SEARCH_RESULTS_COUNT
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
    # 在 Config 类中添加
    FILE_HASH_DB = os.path.join(DATA_PATH, "file_hashes.json")


    # Flask配置
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    # 模板文件配置 - 改为可动态设置
    ANSWER_TEMPLATE = os.getenv("ANSWER_TEMPLATE", "knowledge_base/answer_template.md")

    # 新增模板文件类型验证配置
    ALLOWED_TEMPLATE_EXTENSIONS = {'.txt', '.md', '.rst', '.csv', '.xlsx', '.xls', '.docx', '.pdf'}
    CONFIG_FILE = "config.json"

    @classmethod
    def ensure_directories_exist(cls):
        """确保必要的目录存在"""
        directories = [
            cls.DATA_PATH,
            cls.KNOWLEDGE_BASE_PATH,
            cls.CHROMA_DB_PATH
        ]

        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"创建目录: {directory}")

    @classmethod
    def init(cls):
        """初始化配置"""
        cls.ensure_directories_exist()
        print("配置初始化完成")
        print(f"知识库路径: {cls.KNOWLEDGE_BASE_PATH}")
        print(f"ChromaDB 路径: {cls.CHROMA_DB_PATH}")


    @classmethod
    def load_config(cls):
        """启动时加载保存的配置"""
        if os.path.exists(cls.CONFIG_FILE):
            with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                # 读取保存的模板路径（如果存在）
                if "answer_template_path" in config_data:
                    cls.ANSWER_TEMPLATE = config_data["answer_template_path"]
        else:
            # 首次运行，使用默认路径并创建配置文件
            cls.ANSWER_TEMPLATE = "knowledge_base/answer_template.md"
            cls.save_config()

    @classmethod
    def save_config(cls):
        """保存当前配置到文件"""
        config_data = {
            "answer_template_path": cls.ANSWER_TEMPLATE
        }
        with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def set_answer_template(cls, template_path):
        """动态设置模板路径并保存"""
        ext = os.path.splitext(template_path.lower())[1]
        if ext in cls.ALLOWED_TEMPLATE_EXTENSIONS:
            cls.ANSWER_TEMPLATE = template_path
            cls.save_config()
            return True
        return False


# 程序启动时自动加载配置
Config.load_config()