import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """应用配置类"""

    # Flask配置
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-please-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

    # 知识库配置
    KNOWLEDGE_BASE_DIR = os.getenv('KNOWLEDGE_BASE_DIR', 'knowledge_base')
    DATA_DIR = os.getenv('DATA_DIR', 'data/time_series_docs')
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', 'knowledge_base/time_series')

    # 模型配置
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '384'))

    # 检索配置
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    SEARCH_RESULTS_COUNT = int(os.getenv('SEARCH_RESULTS_COUNT', '5'))

    # LLM配置（可选，用于未来扩展）
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'none')  # openai, huggingface, none
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')

    # 文件类型配置
    SUPPORTED_FILE_TYPES = ['.md', '.txt', '.pdf', '.docx']

    @classmethod
    def ensure_directories_exist(cls):
        """确保必要的目录存在"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.KNOWLEDGE_BASE_DIR, exist_ok=True)


# 创建配置实例
config = Config()