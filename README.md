# shixun_qiu

UltraRAG-Clone/
├── core/
│   ├── __init__.py
│   ├── model_manager.py      # 模型管理模块
│   ├── knowledge_manager.py  # 知识库管理模块
│   ├── data_constructor.py   # 数据构建模块
│   ├── trainer.py           # 训练模块
│   ├── evaluator.py         # 评估推理模块
│   └── rag_system.py        # RAG 系统核心
├── webui/
│   ├── __init__.py
│   ├── app.py              # Web界面主程序
│   ├── static/             # 静态文件
│   └── templates/          # HTML模板
│       ├── base.html
│       ├── index.html
│       ├── knowledge.html
│       ├── training.html
│       └── inference.html
├── workflows/              # 预设工作流
│   ├── __init__.py
│   ├── vanilla_rag.py
│   ├── deepnote.py
│   └── rag_adaptation.py
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── data_utils.py
│   ├── model_utils.py
│   └── file_utils.py
├── config/
│   ├── __init__.py
│   └── settings.py        # 配置文件
├── models/                # 模型存储
├── data/                  # 数据存储
├── requirements.txt
└── main.py               # 主入口文件

启动方式
# 列出所有可用的流水线
python main.py --list-pipelines

# 使用特定流水线执行查询
python main.py --pipeline deepnote_pipeline --query "什么是人工智能?"

# 使用默认流水线执行查询
python main.py --query "机器学习的基本概念"

# 启动带YAML支持的Web界面
python main.py --webui