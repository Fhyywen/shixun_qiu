from knowledge_base.knowledge_base_analyzer import KnowledgeBaseAnalyzer

# 创建分析器实例
analyzer = KnowledgeBaseAnalyzer()

# 先进行分析
print("开始分析知识库...")
stats = analyzer.analyze_knowledge_base("../data/knowledge_base/调研报告素材")

# 检查是否有错误
if "error" in stats:
    print(f"分析失败: {stats['error']}")
else:
    print("分析完成！")

    # 获取统计报告
    report = analyzer.get_statistics_report()
    print(report)

    # 也可以直接查看统计数据的详细信息
    print("\n详细统计数据:")
    import json

    print(json.dumps(stats, ensure_ascii=False, indent=2))