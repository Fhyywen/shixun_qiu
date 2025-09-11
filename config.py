import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # 模型配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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

    # 路径配置 - 添加缺失的配置项
    DATA_PATH = os.getenv("DATA_PATH", "data/")
    DATA_DIR = DATA_PATH  # 别名
    KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base/")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db/")
    VECTOR_STORE_PATH = CHROMA_DB_PATH  # 别名

    # 分块配置 - 添加缺失的配置项
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # 搜索配置 - 添加缺失的配置项
    SEARCH_RESULTS_COUNT = int(os.getenv("SEARCH_RESULTS_COUNT", 3))
    TOP_K = SEARCH_RESULTS_COUNT  # 别名
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

    # Flask配置 - 添加缺失的配置项
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    @staticmethod
    def ensure_directories_exist():
        """确保必要的目录存在"""
        directories = [
            Config.DATA_PATH,
            Config.CHROMA_DB_PATH,
            os.path.dirname(Config.KNOWLEDGE_BASE_PATH) if Config.KNOWLEDGE_BASE_PATH else None
        ]

        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")