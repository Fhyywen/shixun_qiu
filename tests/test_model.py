#!/usr/bin/env python3
"""
检查模型路径的独立脚本
"""

import os
import torch
from sentence_transformers import SentenceTransformer
from config import Config


def check_model_paths():
    print("=" * 60)
    print("🔍 模型路径检查工具")
    print("=" * 60)

    # 获取配置
    config = Config()
    model_name = config.EMBEDDING_MODEL

    print(f"📋 配置的模型: {model_name}")
    print(f"📁 Torch 缓存路径: {torch.hub.get_dir()}")
    print(f"📁 当前工作目录: {os.getcwd()}")

    # 检查环境变量
    print("\n🌐 环境变量:")
    for var in ['TRANSFORMERS_CACHE', 'SENTENCE_TRANSFORMERS_HOME', 'HF_HOME']:
        value = os.getenv(var)
        print(f"   {var}: {value if value else '未设置'}")

    # 尝试加载模型来发现路径
    print(f"\n🚀 尝试加载模型: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ 模型加载成功!")
        print(f"📁 模型存储路径: {model._model_path}")

        # 检查文件
        if os.path.exists(model._model_path):
            print(f"\n📄 模型文件列表:")
            total_size = 0
            for file in os.listdir(model._model_path):
                file_path = os.path.join(model._model_path, file)
                size = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size
                print(f"   - {file}: {size:.2f} MB")

            print(f"\n💾 总大小: {total_size:.2f} MB")
        else:
            print("❌ 模型路径不存在")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 提示: 模型将自动下载到上述缓存路径")


if __name__ == "__main__":
    check_model_paths()