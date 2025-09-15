import chromadb

# 用默认配置加载已存在的实例（关键：不指定新配置，让Chroma自动读取目录中的旧配置）
existing_client = chromadb.PersistentClient(path="data/chroma_db/")

# 打印现有实例的配置（重点关注需要复用的参数）
print("现有Chroma配置：")
print(f"persist_directory: {existing_client.settings.persist_directory}")
print(f"chroma_db_impl: {existing_client.settings.chroma_db_impl}")
print(f"timeout: {existing_client.settings.timeout}")