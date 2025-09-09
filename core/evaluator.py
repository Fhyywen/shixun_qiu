import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """RAG 评估器"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def evaluate_retrieval(self, test_queries: List[str], ground_truths: List[List[str]],
                           top_k: int = 5) -> Dict[str, float]:
        """评估检索性能"""
        if len(test_queries) != len(ground_truths):
            raise ValueError("测试查询和真实值数量必须相同")

        precisions = []
        recalls = []
        f1_scores = []

        for query, gt_doc_ids in zip(test_queries, ground_truths):
            # 执行检索
            retrieved_docs = self.rag_system.retrieve(query, top_k=top_k)
            retrieved_ids = [doc.id for doc in retrieved_docs]

            # 计算指标
            relevant_retrieved = set(retrieved_ids) & set(gt_doc_ids)

            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(relevant_retrieved) / len(gt_doc_ids) if gt_doc_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {
            "precision@k": np.mean(precisions),
            "recall@k": np.mean(recalls),
            "f1_score@k": np.mean(f1_scores),
            "precision_std": np.std(precisions),
            "recall_std": np.std(recalls),
        }

    def evaluate_generation(self, test_queries: List[str], reference_answers: List[str]) -> Dict[str, float]:
        """评估生成性能"""
        if len(test_queries) != len(reference_answers):
            raise ValueError("测试查询和参考答案数量必须相同")

        # 简化版评估 - 实际应使用更复杂的指标如 ROUGE, BLEU 等
        exact_matches = 0
        for query, ref_answer in zip(test_queries, reference_answers):
            result = self.rag_system.query(query)
            if result.answer.strip().lower() == ref_answer.strip().lower():
                exact_matches += 1

        return {
            "exact_match": exact_matches / len(test_queries),
            "total_queries": len(test_queries),
        }

    def evaluate_end_to_end(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """端到端评估"""
        retrieval_metrics = []
        generation_metrics = []

        for test_case in test_data:
            query = test_case["query"]
            gt_doc_ids = test_case.get("relevant_docs", [])
            ref_answer = test_case.get("reference_answer", "")

            # 评估检索
            if gt_doc_ids:
                retrieved_docs = self.rag_system.retrieve(query)
                retrieved_ids = [doc.id for doc in retrieved_docs]
                relevant_retrieved = set(retrieved_ids) & set(gt_doc_ids)

                precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
                recall = len(relevant_retrieved) / len(gt_doc_ids) if gt_doc_ids else 0

                retrieval_metrics.append({
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                })

            # 评估生成
            if ref_answer:
                result = self.rag_system.query(query)
                # 简化版文本相似度评估
                answer = result.answer.strip().lower()
                ref = ref_answer.strip().lower()

                # 计算词重叠率
                answer_words = set(answer.split())
                ref_words = set(ref.split())
                if ref_words:
                    overlap = len(answer_words & ref_words) / len(ref_words)
                    generation_metrics.append(overlap)

        final_metrics = {}
        if retrieval_metrics:
            final_metrics.update({
                "retrieval_precision": np.mean([m["precision"] for m in retrieval_metrics]),
                "retrieval_recall": np.mean([m["recall"] for m in retrieval_metrics]),
                "retrieval_f1": np.mean([m["f1"] for m in retrieval_metrics]),
            })

        if generation_metrics:
            final_metrics["generation_overlap"] = np.mean(generation_metrics)

        return final_metrics