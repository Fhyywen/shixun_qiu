import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RAGAdaptationConfig:
    top_k: int = 7
    confidence_threshold: float = 0.8
    adaptation_strength: float = 0.5
    use_confidence_weighting: bool = True


class RAGAdaptationWorkflow:
    """RAG Adaptation 工作流 - 自适应知识适应"""

    def __init__(self, rag_system, config: RAGAdaptationConfig = None):
        self.rag_system = rag_system
        self.config = config or RAGAdaptationConfig()
        self.domain_knowledge = {}

    def adapt_retrieval(self, query: str, documents: List[Any]) -> List[Any]:
        """自适应检索调整"""
        adapted_documents = []

        for doc in documents:
            # 计算领域适应性分数
            domain_score = self._calculate_domain_relevance(doc.content, query)

            # 应用适应性调整
            if domain_score > self.config.confidence_threshold:
                adapted_doc = doc
                # 可以在这里修改文档内容或元数据
                adapted_documents.append(adapted_doc)

        return adapted_documents

    def adapt_generation(self, prompt: str, context: str) -> str:
        """自适应生成调整"""
        # 分析上下文中的领域特定内容
        domain_features = self._extract_domain_features(context)

        # 修改提示以包含领域指导
        adapted_prompt = self._enhance_prompt_with_domain(prompt, domain_features)

        return adapted_prompt

    def query(self, query: str, **kwargs) -> QueryResult:
        """执行自适应 RAG 查询"""
        import time
        start_time = time.time()

        config = {**self.config.__dict__, **kwargs}

        # 检索文档
        documents = self.rag_system.retrieve(query, top_k=config['top_k'])

        # 自适应检索调整
        adapted_documents = self.adapt_retrieval(query, documents)

        # 构建上下文
        context = "\n\n".join([doc.content for doc in adapted_documents])

        # 构建基础提示
        base_prompt = f"""基于以下背景信息回答问题。请特别注意领域特定的知识和术语。

背景信息:
{context}

问题: {query}

请提供专业、准确的回答:"""

        # 自适应生成调整
        adapted_prompt = self.adapt_generation(base_prompt, context)

        # 生成回答
        generation_model = self.rag_system.model_manager.get_model("generation")
        answer = generation_model.generate(adapted_prompt)

        processing_time = time.time() - start_time

        return QueryResult(
            query=query,
            documents=adapted_documents,
            answer=answer,
            confidence=self._calculate_answer_confidence(answer, context),
            processing_time=processing_time
        )

    def _calculate_domain_relevance(self, content: str, query: str) -> float:
        """计算领域相关性"""
        # 简化实现 - 实际应使用更复杂的领域检测
        domain_keywords = self._get_domain_keywords()

        content_words = set(content.lower().split())
        query_words = set(query.lower().split())

        domain_match = len(content_words & domain_keywords) / len(domain_keywords) if domain_keywords else 0
        query_match = len(content_words & query_words) / len(query_words) if query_words else 0

        return (domain_match + query_match) / 2

    def _get_domain_keywords(self) -> set:
        """获取领域关键词"""
        # 这里应该从领域知识库中获取关键词
        # 简化实现
        return set(["法律", "条款", "法规", "案例", "判决", "合同"])

    def _extract_domain_features(self, context: str) -> Dict[str, Any]:
        """提取领域特征"""
        # 简化实现
        return {
            "has_legal_terms": any(term in context for term in ["法律", "法规", "条款"]),
            "has_technical_terms": any(term in context for term in ["技术", "系统", "算法"]),
            "length": len(context)
        }

    def _enhance_prompt_with_domain(self, prompt: str, domain_features: Dict[str, Any]) -> str:
        """使用领域信息增强提示"""
        if domain_features["has_legal_terms"]:
            enhanced_prompt = prompt + "\n\n请注意：这是一个法律领域的问题，请确保回答准确且符合法律规范。"
        elif domain_features["has_technical_terms"]:
            enhanced_prompt = prompt + "\n\n请注意：这是一个技术领域的问题，请提供专业且准确的技术解释。"
        else:
            enhanced_prompt = prompt

        return enhanced_prompt

    def _calculate_answer_confidence(self, answer: str, context: str) -> float:
        """计算回答置信度"""
        # 基于回答与上下文的一致性
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0
        return min(overlap * 1.5, 1.0)  # 缩放并限制在 0-1 范围内

    def add_domain_knowledge(self, domain: str, knowledge: Dict[str, Any]):
        """添加领域知识"""
        self.domain_knowledge[domain] = knowledge

    def get_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """获取领域知识"""
        return self.domain_knowledge.get(domain, {})