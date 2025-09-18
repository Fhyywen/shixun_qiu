import os
import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime


class KnowledgeBaseAnalyzer:
    def __init__(self, config=None):
        if config is None:
            from config import Config
            self.config = Config()
        else:
            self.config = config

        # 确保输出目录存在
        os.makedirs(self.config.KNOWLEDGE_BASE_PATH, exist_ok=True)
        self.stats_file_path = os.path.join(self.config.KNOWLEDGE_BASE_PATH, "knowledge_base_statistics.json")

    def analyze_knowledge_base(self, knowledge_base_path: str = None) -> Dict[str, Any]:
        """分析知识库并生成统计信息"""
        if knowledge_base_path is None:
            knowledge_base_path = self.config.KNOWLEDGE_BASE_PATH

        if not os.path.exists(knowledge_base_path):
            return {"error": f"知识库路径不存在: {knowledge_base_path}"}

        print(f"开始分析知识库: {knowledge_base_path}")

        # 收集统计信息
        stats = {
            "analysis_date": datetime.now().isoformat(),
            "knowledge_base_path": knowledge_base_path,
            "file_statistics": self._analyze_files(knowledge_base_path),
            "content_statistics": self._analyze_content(knowledge_base_path),
            "case_statistics": self._extract_case_statistics(knowledge_base_path),
            "summary": {}
        }

        # 生成总结
        stats["summary"] = self._generate_summary(stats)

        # 保存统计结果
        save_success = self._save_statistics(stats)
        if not save_success:
            print("警告：保存统计结果到JSON文件失败")

        print(f"知识库分析完成！共分析 {stats['file_statistics']['total_files']} 个文件")
        return stats

    def _analyze_files(self, base_path: str) -> Dict[str, Any]:
        """分析文件统计信息"""
        file_stats = {
            "total_files": 0,
            "file_types": {},
            "file_sizes": {
                "total_size_bytes": 0,
                "average_size_bytes": 0
            },
            "files_by_extension": {}
        }

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith('.'):  # 跳过隐藏文件
                    continue

                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # 文件类型统计
                file_stats["total_files"] += 1

                # 按扩展名统计
                if file_ext in file_stats["files_by_extension"]:
                    file_stats["files_by_extension"][file_ext] += 1
                else:
                    file_stats["files_by_extension"][file_ext] = 1

                # 文件大小统计
                try:
                    file_size = os.path.getsize(file_path)
                    file_stats["file_sizes"]["total_size_bytes"] += file_size
                except:
                    continue

        # 计算平均文件大小
        if file_stats["total_files"] > 0:
            file_stats["file_sizes"]["average_size_bytes"] = (
                    file_stats["file_sizes"]["total_size_bytes"] / file_stats["total_files"]
            )

        return file_stats

    def _analyze_content(self, base_path: str) -> Dict[str, Any]:
        """分析内容统计信息"""
        content_stats = {
            "total_documents": 0,
            "total_words": 0,
            "total_characters": 0,
            "documents_by_type": {
                "cases": 0,
                "reports": 0,
                "statistics": 0,
                "other": 0
            },
            "word_frequency": {}
        }

        # 简单的关键词匹配来分类文档类型
        case_keywords = ["案件", "案例", "事故", "事件", "纠纷", "诉讼"]
        report_keywords = ["报告", "总结", "分析", "研究", "调查"]
        stats_keywords = ["统计", "数据", "数字", "比例", "百分比"]

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                try:
                    content = self._read_file_content(file_path, file_ext)
                    if not content:
                        continue

                    content_stats["total_documents"] += 1
                    content_stats["total_words"] += len(content.split())
                    content_stats["total_characters"] += len(content)

                    # 分类文档类型
                    content_lower = content.lower()
                    if any(keyword in content_lower for keyword in case_keywords):
                        content_stats["documents_by_type"]["cases"] += 1
                    elif any(keyword in content_lower for keyword in report_keywords):
                        content_stats["documents_by_type"]["reports"] += 1
                    elif any(keyword in content_lower for keyword in stats_keywords):
                        content_stats["documents_by_type"]["statistics"] += 1
                    else:
                        content_stats["documents_by_type"]["other"] += 1

                    # 简单的词频统计（示例）
                    words = content.split()[:50]  # 只取前50个词作为示例
                    for word in words:
                        if len(word) > 2:  # 只统计长度大于2的词
                            if word in content_stats["word_frequency"]:
                                content_stats["word_frequency"][word] += 1
                            else:
                                content_stats["word_frequency"][word] = 1

                except Exception as e:
                    print(f"分析文件 {file_path} 时出错: {e}")
                    continue

        # 按词频排序
        content_stats["word_frequency"] = dict(
            sorted(content_stats["word_frequency"].items(),
                   key=lambda x: x[1], reverse=True)[:20]  # 取前20个
        )

        return content_stats

    def _extract_case_statistics(self, base_path: str) -> Dict[str, Any]:
        """提取案件相关统计信息"""
        case_stats = {
            "total_cases": 0,
            "cases_by_type": {},
            "cases_by_year": {},
            "cases_by_region": {},
            "case_resolution": {
                "resolved": 0,
                "pending": 0,
                "dismissed": 0
            }
        }

        # 案件类型关键词映射
        case_type_patterns = {
            "民事案件": ["民事", "合同", "债务", "侵权", "婚姻", "继承"],
            "刑事案件": ["刑事", "犯罪", "盗窃", "抢劫", "诈骗", "伤害"],
            "行政案件": ["行政", "处罚", "许可", "复议"],
            "经济案件": ["经济", "金融", "证券", "保险", "破产"],
            "劳动案件": ["劳动", "雇佣", "工资", "工伤", "仲裁"]
        }

        # 地区关键词（示例）
        regions = ["北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "重庆", "西安"]

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                try:
                    content = self._read_file_content(file_path, file_ext)
                    if not content:
                        continue

                    content_lower = content.lower()

                    # 检测是否为案件文档
                    case_keywords = ["案件", "案例", "案号", "原告", "被告", "法院", "判决"]
                    if any(keyword in content_lower for keyword in case_keywords):
                        case_stats["total_cases"] += 1

                        # 分析案件类型
                        for case_type, keywords in case_type_patterns.items():
                            if any(keyword in content_lower for keyword in keywords):
                                if case_type in case_stats["cases_by_type"]:
                                    case_stats["cases_by_type"][case_type] += 1
                                else:
                                    case_stats["cases_by_type"][case_type] = 1
                                break

                        # 分析年份（简单匹配4位数字）
                        import re
                        year_matches = re.findall(r'\b(20\d{2})\b', content)
                        if year_matches:
                            year = year_matches[0]
                            if year in case_stats["cases_by_year"]:
                                case_stats["cases_by_year"][year] += 1
                            else:
                                case_stats["cases_by_year"][year] = 1

                        # 分析地区
                        for region in regions:
                            if region in content:
                                if region in case_stats["cases_by_region"]:
                                    case_stats["cases_by_region"][region] += 1
                                else:
                                    case_stats["cases_by_region"][region] = 1
                                break

                        # 分析处理状态
                        if "已结" in content or "终结" in content or "执行完毕" in content:
                            case_stats["case_resolution"]["resolved"] += 1
                        elif "未结" in content or "审理中" in content or "待处理" in content:
                            case_stats["case_resolution"]["pending"] += 1
                        elif "驳回" in content or "撤诉" in content:
                            case_stats["case_resolution"]["dismissed"] += 1

                except Exception as e:
                    print(f"分析案件文件 {file_path} 时出错: {e}")
                    continue

        return case_stats

    def _read_file_content(self, file_path: str, file_ext: str) -> str:
        """读取文件内容"""
        try:
            if file_ext in ['.txt', '.md', '.rst', '.markdown']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                return df.to_string()
            elif file_ext == '.docx':
                try:
                    import docx
                    doc = docx.Document(file_path)
                    return "\n".join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    print("请安装 python-docx 包以支持 .docx 文件读取")
                    return ""
            elif file_ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    print("请安装 PyPDF2 包以支持 .pdf 文件读取")
                    return ""
            else:
                # 尝试以文本方式读取其他文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except:
                    return ""
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return ""

    def _generate_summary(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """生成总结信息"""
        summary = {
            "overview": f"知识库包含 {stats['file_statistics']['total_files']} 个文件，"
                        f"{stats['content_statistics']['total_documents']} 个文档，"
                        f"共 {stats['content_statistics']['total_words']} 个词",
            "case_summary": f"共发现 {stats['case_statistics']['total_cases']} 个案件",
            "main_file_types": dict(sorted(
                stats['file_statistics']['files_by_extension'].items(),
                key=lambda x: x[1], reverse=True
            )[:5]),  # 前5个文件类型
            "main_case_types": dict(sorted(
                stats['case_statistics']['cases_by_type'].items(),
                key=lambda x: x[1], reverse=True
            )[:5]) if stats['case_statistics']['cases_by_type'] else "无案件数据"
        }
        return summary

    def _save_statistics(self, stats: Dict[str, Any]) -> bool:
        """保存统计结果到JSON文件，返回是否成功"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.stats_file_path), exist_ok=True)

            with open(self.stats_file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"统计结果已保存到: {self.stats_file_path}")
            return True
        except Exception as e:
            print(f"保存统计结果时出错: {e}")
            return False

    def get_statistics_report(self, knowledge_base_path: str = None) -> str:
        """获取统计报告文本"""
        # 如果指定了知识库路径，先进行分析
        if knowledge_base_path:
            stats = self.analyze_knowledge_base(knowledge_base_path)
            if "error" in stats:
                return f"分析失败: {stats['error']}"

        if not os.path.exists(self.stats_file_path):
            return "暂无统计报告，请先运行 analyze_knowledge_base() 方法进行分析"

        try:
            with open(self.stats_file_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            report = f"""知识库统计分析报告
生成时间: {stats['analysis_date']}
知识库路径: {stats['knowledge_base_path']}

文件统计:
- 总文件数: {stats['file_statistics']['total_files']}
- 总大小: {stats['file_statistics']['file_sizes']['total_size_bytes'] / 1024 / 1024:.2f} MB
- 主要文件类型: {', '.join([f'{k}: {v}' for k, v in stats['file_statistics']['files_by_extension'].items()][:5])}

内容统计:
- 总文档数: {stats['content_statistics']['total_documents']}
- 总词数: {stats['content_statistics']['total_words']}
- 文档类型分布: 案件({stats['content_statistics']['documents_by_type']['cases']}), 
              报告({stats['content_statistics']['documents_by_type']['reports']}), 
              统计({stats['content_statistics']['documents_by_type']['statistics']})

案件统计:
- 总案件数: {stats['case_statistics']['total_cases']}
- 案件类型分布: {', '.join([f'{k}: {v}' for k, v in stats['case_statistics']['cases_by_type'].items()])}
- 年度分布: {', '.join([f'{k}: {v}' for k, v in stats['case_statistics']['cases_by_year'].items()])}
- 处理状态: 已结({stats['case_statistics']['case_resolution']['resolved']}), 
          未结({stats['case_statistics']['case_resolution']['pending']}), 
          驳回({stats['case_statistics']['case_resolution']['dismissed']})
"""
            return report
        except Exception as e:
            return f"读取统计报告时出错: {e}"

    def debug_save_function(self):
        """调试保存功能"""
        test_data = {
            "test": "这是一个测试数据",
            "timestamp": datetime.now().isoformat()
        }

        try:
            # 检查目录权限
            dir_path = os.path.dirname(self.stats_file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"创建目录: {dir_path}")

            # 检查文件写入权限
            with open(self.stats_file_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)

            print(f"测试文件成功写入: {self.stats_file_path}")

            # 读取验证
            with open(self.stats_file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                print(f"读取验证: {content}")

            return True
        except Exception as e:
            print(f"调试失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = KnowledgeBaseAnalyzer()

    # 首先调试保存功能
    print("调试保存功能...")
    if analyzer.debug_save_function():
        print("保存功能正常")
    else:
        print("保存功能有问题，请检查目录权限")

    # 分析知识库
    print("\n开始分析知识库...")
    stats = analyzer.analyze_knowledge_base("/data/knowledge_base/调研报告素材")

    # 获取统计报告
    report = analyzer.get_statistics_report()
    print(report)

    # 检查文件是否真的保存了
    if os.path.exists(analyzer.stats_file_path):
        print(f"\n统计文件已成功创建: {analyzer.stats_file_path}")
        print(f"文件大小: {os.path.getsize(analyzer.stats_file_path)} 字节")
    else:
        print(f"\n错误: 统计文件未创建: {analyzer.stats_file_path}")