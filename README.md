OCR依赖下载：
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html

https://github.com/PaddlePaddle/PaddleOCR/blob/main/readme/README_cn.md

快捷下载（确保依赖版本与requirement.txt相符）：
CPU版本：python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

GPU版本：python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

python -m pip install "paddleocr[all]"

运行ocr_test.py成功运行即安装完成


ResearchReport -Knowledge-Base/
├── app.py                      # Flask Web应用主入口
├── config.py                   # 应用配置文件
├── .env                        # 环境变量文件
├── requirements.txt            # Python依赖包列表
├── README.md                   # 项目说明文档
│
├── knowledge_base/             # 知识库核心模块
│   ├── __init__.py            # 包初始化文件
│   ├── data_processing.py     # 数据处理和文本分块
│   ├── vector_store.py        # FAISS向量存储管理
│   ├── qa_system.py           # 问答系统核心逻辑
│   ├── utils.py               # 工具函数
│   └── llm_integration.py     # LLM集成模块（可选）
│
├── data/                       # 数据存储目录
│   └── time_series_docs/      # 时间序列相关文档
│       ├── arima.md           # ARIMA算法文档
│       ├── prophet.md         # Prophet算法文档
│       ├── lstm.md            # LSTM算法文档
│       ├── ets.md             # ETS算法文档
│       └── preprocessing.md   # 数据预处理文档
│
├── knowledge_base_files/       # 向量存储文件目录
│   ├── time_series.index      # FAISS索引文件
│   └── time_series.data       # 元数据文件
│
├── static/                     # 静态资源文件
│   ├── css/
│   │   └── style.css          # 样式表
│   ├── js/
│   │   └── script.js          # JavaScript文件
│   └── images/                # 图片资源
│
├── templates/                  # HTML模板文件
│   ├── base.html              # 基础模板
│   ├── index.html             # 主页面
│   ├── results.html           # 结果展示页面
│   └── error.html             # 错误页面
│
├── logs/                       # 日志文件目录
│   └── app.log                # 应用日志
│
├── tests/                      # 测试目录
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_vector_store.py
│   ├── test_qa_system.py
│   └── test_app.py
│
└── scripts/                    # 脚本工具目录
    ├── build_knowledge_base.py # 知识库构建脚本
    ├── add_document.py         # 文档添加脚本
    └── evaluate_system.py      # 系统评估脚本

ask接口测试用例：
{
    "question": "帮我生产一份东城的社会调研报告，限制500字左右，简单一点",
    "knowledge_base_path": "/data/knowledge_base/test3"
}


-- 创建数据库
CREATE DATABASE IF NOT EXISTS knowledge_base_chat DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE knowledge_base_chat;

-- 创建会话表
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) DEFAULT 'anonymous',
    knowledge_base_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    title VARCHAR(200) DEFAULT '新对话',
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);

-- 创建对话消息表
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    role ENUM('system', 'user', 'assistant') NOT NULL,
    content TEXT NOT NULL,
    tokens INT DEFAULT 0,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

-- 创建知识库使用记录表
CREATE TABLE IF NOT EXISTS knowledge_base_usage (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    knowledge_base_path VARCHAR(500) NOT NULL,
    question TEXT NOT NULL,
    similar_docs_count INT DEFAULT 0,
    average_similarity FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_knowledge_base (knowledge_base_path(200)),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

在config.py中修改mysql数据库的用户名，密码，端口号等等数据