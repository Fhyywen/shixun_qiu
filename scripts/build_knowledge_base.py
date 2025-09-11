#!/usr/bin/env python3
"""
知识库构建脚本 - 用于初始化和更新向量知识库
"""
import argparse
from knowledge_base.qa_system import TimeSeriesQA


def main():
    parser = argparse.ArgumentParser(description='构建时间序列知识库')
    parser.add_argument('--force', action='store_true', help='强制重新构建知识库')
    args = parser.parse_args()

    qa_system = TimeSeriesQA()
    if args.force:
        # 删除现有索引文件
        import os
        index_files = [
            f"{qa_system.vector_store_path}.index",
            f"{qa_system.vector_store_path}.data"
        ]
        for file in index_files:
            if os.path.exists(file):
                os.remove(file)

    success = qa_system.initialize()
    if success:
        print("知识库构建成功!")
    else:
        print("知识库构建失败!")


if __name__ == '__main__':
    main()