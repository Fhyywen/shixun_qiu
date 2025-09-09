import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def setup_yaml_pipelines(rag_system):
    """设置 YAML 流水线"""
    from workflows.pipeline_manager import PipelineManager

    pipeline_manager = PipelineManager(rag_system)

    # 加载所有 YAML 流水线
    workflows_dir = Path("workflows")
    yaml_files = list(workflows_dir.glob("*_pipeline.yaml"))

    for yaml_file in yaml_files:
        try:
            pipeline_manager.load_pipeline(str(yaml_file))
        except Exception as e:
            logging.warning(f"Failed to load pipeline {yaml_file}: {e}")

    return pipeline_manager


def main():
    parser = argparse.ArgumentParser(description="UltraRAG-Clone - 支持 YAML 流水线")
    parser.add_argument("--webui", action="store_true", help="启动Web界面")
    parser.add_argument("--cli", action="store_true", help="启动命令行界面")
    parser.add_argument("--pipeline", help="使用指定的YAML流水线执行查询")
    parser.add_argument("--query", help="执行查询")
    parser.add_argument("--list-pipelines", action="store_true", help="列出所有可用的流水线")

    args = parser.parse_args()

    # 初始化RAG系统
    from core.rag_system import RAGSystem
    rag_system = RAGSystem()

    if not rag_system.initialize():
        logging.error("Failed to initialize RAG system")
        return

    try:
        # 设置YAML流水线
        pipeline_manager = setup_yaml_pipelines(rag_system)

        if args.list_pipelines:
            # 列出所有流水线
            print("可用的流水线:")
            for pipeline_name in pipeline_manager.pipelines.keys():
                print(f"  - {pipeline_name}")

        elif args.pipeline and args.query:
            # 使用指定流水线执行查询
            result = pipeline_manager.execute_pipeline(args.pipeline, args.query)
            print(f"回答: {result['results'].get('generate', {}).get('answer', 'No answer generated')}")

        elif args.query:
            # 使用默认流水线
            result = pipeline_manager.execute_pipeline("vanilla_rag_pipeline", args.query)
            print(f"回答: {result['results'].get('generate', {}).get('answer', 'No answer generated')}")

        elif args.webui:
            from webui.app import run_webui
            run_webui(rag_system, pipeline_manager)

        elif args.cli:
            print("UltraRAG-Clone 命令行界面 (输入 'quit' 退出)")
            while True:
                query = input("\n请输入问题: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if query:
                    result = pipeline_manager.execute_pipeline("vanilla_rag_pipeline", query)
                    answer = result['results'].get('generate', {}).get('answer', 'No answer generated')
                    print(f"\n回答: {answer}")

        else:
            parser.print_help()

    finally:
        rag_system.shutdown()


if __name__ == "__main__":
    main()