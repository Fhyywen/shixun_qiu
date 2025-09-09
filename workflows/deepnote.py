import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class MemoryNote:
    id: str
    content: str
    relevance_score: float
    timestamp: float


@dataclass
class DeepNoteConfig:
    top_k: int = 5
    memory_size: int = 10
    relevance_threshold: float = 0.6
    max_iterations: int = 3
    temperature: float = 0.1


class DeepNoteWorkflow:
    """DeepNote 工作流 - 带有记忆机制的迭代检索"""

    def __init__(self, rag_system, config: DeepNoteConfig = None):
        self.rag_system = rag_system
        self.config = config or DeepNoteConfig()
        self.memory: List[MemoryNote] = []

    def update_memory(self, new_notes: List[MemoryNote]):
        """更新记忆"""
        # 添加新记忆
        self.memory.extend(new_notes)

        # 按相关性排序并截断
        self.memory.sort(key=lambda x: x.relevance_score, reverse=True)
        self.memory = self.memory[:self.config.memory_size]

    def build_iterative_prompt(self, query: str, documents: List[Any],
                               previous_notes: List[MemoryNote] = None) -> str:
        """构建迭代提示"""
        context = "\n\n".join([doc.content for doc in documents])

        memory_context = ""
        if previous_notes:
            memory_context = "\n".join([f"- {note.content}" for note in previous_notes])

        prompt = f"""你是一个具有记忆能力的AI助手。基于以下背景信息和之前的记忆，请回答用户的问题。

背景信息:
{context}

{'之前的记忆:' if memory_context else ''}
{memory_context}

用户问题: {query}

请逐步思考，并基于所有可用信息提供准确回答:"""

        return prompt

    def query(self, query: str, **kwargs) -> QueryResult:
        """执行 DeepNote 查询"""
        import time
        start_time = time.time()

        config = {**self.config.__dict__, **kwargs}
        all_documents = []

        # 迭代检索和生成
        for iteration in range(config['max_iterations']):
            # 检索相关文档
            current_docs = self.rag_system.retrieve(
                query,
                top_k=config['top_k'],
                threshold=config['relevance_threshold']
            )
            all_documents.extend(current_docs)

            # 构建提示
            prompt = self.build_iterative_prompt(query, current_docs, self.memory)

            # 生成回答
            generation_model = self.rag_system.model_manager.get_model("generation")
            answer = generation_model.generate(
                prompt,
                temperature=config['temperature'],
                max_length=1024
            )

            # 评估是否需要继续迭代
            if self._should_stop_iteration(answer, iteration):
                break

            # 更新记忆
            new_note = MemoryNote(
                id=f"note_{int(time.time())}_{iteration}",
                content=f"Query: {query}\nAnswer: {answer[:200]}...",
                relevance_score=0.8,
                timestamp=time.time()
            )
            self.update_memory([new_note])

        processing_time = time.time() - start_time

        return QueryResult(
            query=query,
            documents=all_documents,
            answer=answer,
            confidence=0.9,
            processing_time=processing_time
        )

    def _should_stop_iteration(self, answer: str, iteration: int) -> bool:
        """判断是否停止迭代"""
        # 简单的停止条件：答案足够完整或达到最大迭代次数
        stop_phrases = ["信息不足", "无法回答", "根据以上信息", "综上所述"]
        if any(phrase in answer for phrase in stop_phrases):
            return True

        if iteration >= self.config.max_iterations - 1:
            return True

        return False

    def clear_memory(self):
        """清空记忆"""
        self.memory = []

    def save_memory(self, file_path: str):
        """保存记忆到文件"""
        memory_data = [
            {
                "id": note.id,
                "content": note.content,
                "relevance_score": note.relevance_score,
                "timestamp": note.timestamp
            }
            for note in self.memory
        ]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)

    def load_memory(self, file_path: str):
        """从文件加载记忆"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            self.memory = [
                MemoryNote(
                    id=item["id"],
                    content=item["content"],
                    relevance_score=item["relevance_score"],
                    timestamp=item["timestamp"]
                )
                for item in memory_data
            ]
        except Exception as e:
            logger.error(f"Error loading memory: {e}")