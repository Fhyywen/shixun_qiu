import logging
import argparse
from core.rag_system import RAGSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="UltraRAG-Clone - RAG System")
    parser.add_argument("--webui", action="store_true", help="启动Web界面")
    parser.add_argument("--cli", action="store_true", help="启动命令行界面")
    parser.add_argument("--add-docs", nargs='+', help="添加文档到知识库")
    parser.add_argument("--query", help="执行查询")

    args = parser.parse_args()

    # 初始化RAG系统
    rag_system = RAGSystem()
    if not rag_system.initialize():
        logger.error("Failed to initialize RAG system")
        return

    try:
        if args.add_docs:
            # 添加文档模式
            success = rag_system.add_documents(args.add_docs)
            if success:
                logger.info("Documents added successfully")
            else:
                logger.error("Failed to add documents")

        elif args.query:
            # 查询模式
            result = rag_system.query(args.query)
            print(f"问题: {result.query}")
            print(f"回答: {result.answer}")
            print(f"处理时间: {result.processing_time:.2f}秒")
            print("\n参考文档:")
            for i, doc in enumerate(result.documents):
                print(f"{i + 1}. {doc.content[:100]}...")

        elif args.webui:
            # 启动Web界面
            from webui.app import run_webui
            run_webui(rag_system)

        elif args.cli:
            # 命令行交互模式
            print("UltraRAG-Clone 命令行界面 (输入 'quit' 退出)")
            while True:
                query = input("\n请输入问题: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if query:
                    result = rag_system.query(query)
                    print(f"\n回答: {result.answer}")
                    if result.documents:
                        print("\n参考文档:")
                        for i, doc in enumerate(result.documents):
                            print(f"{i + 1}. {doc.content[:100]}...")

        else:
            parser.print_help()

    finally:
        rag_system.shutdown()


if __name__ == "__main__":
    main()