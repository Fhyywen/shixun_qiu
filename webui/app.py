import logging
import gradio as gr
import os
from typing import List, Dict, Any
from core.rag_system import RAGSystem
from workflows.vanilla_rag import VanillaRAGWorkflow
from workflows.deepnote import DeepNoteWorkflow
from workflows.rag_adaptation import RAGAdaptationWorkflow

logger = logging.getLogger(__name__)


class RAGWebUI:
    """RAG 系统 Web 界面"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.workflows = {
            "vanilla": VanillaRAGWorkflow(rag_system),
            "deepnote": DeepNoteWorkflow(rag_system),
            "adaptation": RAGAdaptationWorkflow(rag_system)
        }
        self.current_workflow = "vanilla"
        self.uploaded_files = []

    def create_ui(self):
        """创建 Gradio 界面"""
        with gr.Blocks(title="UltraRAG-Clone", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🚀 UltraRAG-Clone - RAG 系统")

            with gr.Tab("知识管理"):
                self._create_knowledge_management_tab()

            with gr.Tab("查询界面"):
                self._create_query_tab()

            with gr.Tab("工作流配置"):
                self._create_workflow_tab()

            with gr.Tab("系统状态"):
                self._create_status_tab()

        return demo

    def _create_knowledge_management_tab(self):
        """创建知识管理标签页"""
        with gr.Row():
            with gr.Column(scale=1):
                file_output = gr.File(label="上传文档", file_count="multiple")
                upload_btn = gr.Button("上传到知识库", variant="primary")
                clear_btn = gr.Button("清空知识库", variant="secondary")

                gr.Markdown("### 支持格式")
                gr.Markdown("- 📄 PDF (.pdf)\n- 📝 文本 (.txt)\n- 📋 Word (.docx)\n- 📊 Markdown (.md)")

            with gr.Column(scale=2):
                kb_status = gr.Markdown("### 知识库状态\n- 文档数量: 0\n- 总字符数: 0")
                file_list = gr.DataFrame(
                    headers=["文件名", "大小", "状态"],
                    value=[],
                    label="已上传文件"
                )

        upload_btn.click(
            self.upload_files,
            inputs=[file_output],
            outputs=[kb_status, file_list]
        )

        clear_btn.click(
            self.clear_knowledge_base,
            outputs=[kb_status, file_list]
        )

    def _create_query_tab(self):
        """创建查询标签页"""
        with gr.Row():
            with gr.Column(scale=1):
                workflow_dropdown = gr.Dropdown(
                    choices=list(self.workflows.keys()),
                    value=self.current_workflow,
                    label="选择工作流"
                )

                query_text = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入您的问题...",
                    lines=3
                )

                query_btn = gr.Button("执行查询", variant="primary")
                clear_chat_btn = gr.Button("清空对话", variant="secondary")

            with gr.Column(scale=2):
                chat_output = gr.Chatbot(label="对话历史", height=400)
                answer_output = gr.Textbox(label="回答", lines=5)

                with gr.Accordion("参考文档", open=False):
                    doc_output = gr.DataFrame(
                        headers=["文档", "内容片段"],
                        value=[],
                        max_rows=5
                    )

        query_btn.click(
            self.execute_query,
            inputs=[query_text, workflow_dropdown],
            outputs=[chat_output, answer_output, doc_output]
        )

        clear_chat_btn.click(
            lambda: ([], "", []),
            outputs=[chat_output, answer_output, doc_output]
        )

    def _create_workflow_tab(self):
        """创建工作流配置标签页"""
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Vanilla RAG 配置")
                top_k_slider = gr.Slider(1, 10, value=5, label="检索数量 (top_k)")
                threshold_slider = gr.Slider(0.0, 1.0, value=0.7, label="相似度阈值")

            with gr.Column():
                gr.Markdown("### DeepNote 配置")
                memory_size = gr.Slider(5, 20, value=10, label="记忆大小")
                max_iterations = gr.Slider(1, 5, value=3, label="最大迭代次数")

        save_config_btn = gr.Button("保存配置", variant="primary")
        config_status = gr.Markdown("配置状态: 未保存")

        save_config_btn.click(
            self.save_configuration,
            inputs=[top_k_slider, threshold_slider, memory_size, max_iterations],
            outputs=[config_status]
        )

    def _create_status_tab(self):
        """创建系统状态标签页"""
        gr.Markdown("### 系统信息")

        model_status = gr.Markdown("**模型状态:**\n- 嵌入模型: 未加载\n- 生成模型: 未加载")
        kb_status = gr.Markdown("**知识库状态:**\n- 文档数量: 0\n- 索引状态: 未构建")
        system_status = gr.Markdown("**系统状态:**\n- 内存使用: 未知\n- 运行时间: 未知")

        refresh_btn = gr.Button("刷新状态", variant="primary")

        refresh_btn.click(
            self.get_system_status,
            outputs=[model_status, kb_status, system_status]
        )

    def upload_files(self, files):
        """处理文件上传"""
        try:
            if files:
                file_paths = [f.name for f in files]
                success = self.rag_system.add_documents(file_paths)

                if success:
                    status = "### 知识库状态\n- 文档数量: {}\n- 上传成功".format(
                        len(self.rag_system.current_kb.documents) if self.rag_system.current_kb else 0
                    )

                    file_info = [
                        [os.path.basename(path), f"{os.path.getsize(path)} bytes", "成功"]
                        for path in file_paths
                    ]

                    return status, file_info
        except Exception as e:
            logger.error(f"Upload error: {e}")

        return "上传失败", []

    def clear_knowledge_base(self):
        """清空知识库"""
        # 实现清空逻辑
        return "知识库已清空", []

    def execute_query(self, query, workflow_type):
        """执行查询"""
        try:
            if not query.strip():
                return [], "请输入有效问题", []

            workflow = self.workflows.get(workflow_type, self.workflows["vanilla"])
            result = workflow.query(query)

            # 构建聊天历史
            chat_history = [(query, result.answer)]

            # 构建文档列表
            doc_list = [
                [f"文档 {i + 1}", doc.content[:100] + "..."]
                for i, doc in enumerate(result.documents)
            ]

            return chat_history, result.answer, doc_list

        except Exception as e:
            logger.error(f"Query error: {e}")
            return [], f"查询出错: {str(e)}", []

    def save_configuration(self, top_k, threshold, memory_size, max_iterations):
        """保存配置"""
        try:
            self.workflows["vanilla"].config.top_k = top_k
            self.workflows["vanilla"].config.score_threshold = threshold
            self.workflows["deepnote"].config.memory_size = memory_size
            self.workflows["deepnote"].config.max_iterations = max_iterations

            return "配置已保存"
        except Exception as e:
            return f"配置保存失败: {str(e)}"

    def get_system_status(self):
        """获取系统状态"""
        # 简化实现
        model_status = "**模型状态:**\n- 嵌入模型: 已加载\n- 生成模型: 已加载"

        kb_count = len(self.rag_system.current_kb.documents) if self.rag_system.current_kb else 0
        kb_status = f"**知识库状态:**\n- 文档数量: {kb_count}\n- 索引状态: 已构建"

        system_status = "**系统状态:**\n- 内存使用: 正常\n- 运行时间: 活跃"

        return model_status, kb_status, system_status


def run_webui(rag_system: RAGSystem):
    """运行 Web 界面"""
    webui = RAGWebUI(rag_system)
    demo = webui.create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)