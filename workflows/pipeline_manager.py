import yaml
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineManager:
    """流水线管理器 - 支持 YAML 配置的 UltraRAG 2.0 风格"""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.pipelines: Dict[str, Any] = {}

    def load_pipeline(self, yaml_path: str) -> Dict[str, Any]:
        """加载 YAML 流水线配置"""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                pipeline_config = yaml.safe_load(f)

            pipeline_name = Path(yaml_path).stem
            self.pipelines[pipeline_name] = pipeline_config
            logger.info(f"Loaded pipeline '{pipeline_name}' from {yaml_path}")
            return pipeline_config

        except Exception as e:
            logger.error(f"Error loading pipeline from {yaml_path}: {e}")
            raise

    def execute_pipeline(self, pipeline_name: str, query: str, **kwargs) -> Any:
        """执行流水线"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not loaded")

        pipeline_config = self.pipelines[pipeline_name]
        return self._execute_steps(pipeline_config['steps'], query, **kwargs)

    def _execute_steps(self, steps: List[Dict[str, Any]], query: str, **kwargs) -> Any:
        """执行流水线步骤"""
        context = {'query': query, 'results': {}}

        for step_index, step in enumerate(steps):
            step_name = step.get('name', f'step_{step_index}')
            step_type = step['type']

            logger.info(f"Executing step {step_index}: {step_name} ({step_type})")

            try:
                if step_type == 'retrieve':
                    result = self._execute_retrieve(step, context, **kwargs)
                elif step_type == 'generate':
                    result = self._execute_generate(step, context, **kwargs)
                elif step_type == 'conditional':
                    result = self._execute_conditional(step, context, **kwargs)
                elif step_type == 'loop':
                    result = self._execute_loop(step, context, **kwargs)
                else:
                    raise ValueError(f"Unknown step type: {step_type}")

                context['results'][step_name] = result
                context['last_result'] = result

            except Exception as e:
                logger.error(f"Error executing step {step_name}: {e}")
                if step.get('continue_on_error', False):
                    continue
                raise

        return context

    def _execute_retrieve(self, step: Dict[str, Any], context: Dict[str, Any], **kwargs) -> Any:
        """执行检索步骤"""
        query = context['query']
        top_k = step.get('top_k', 5)
        threshold = step.get('threshold', 0.7)

        # 从知识库检索
        documents = self.rag_system.retrieve(query, top_k=top_k, threshold=threshold)
        return {
            'documents': documents,
            'count': len(documents),
            'query': query
        }

    def _execute_generate(self, step: Dict[str, Any], context: Dict[str, Any], **kwargs) -> Any:
        """执行生成步骤"""
        # 构建提示模板
        prompt_template = step.get('prompt_template', """
基于以下信息回答问题：
{context}

问题：{query}

请提供准确的回答：
""")

        # 获取上下文信息
        documents = context.get('results', {}).get('retrieve', {}).get('documents', [])
        context_text = "\n\n".join([doc.content for doc in documents])

        # 渲染提示
        prompt = prompt_template.format(
            context=context_text,
            query=context['query']
        )

        # 生成回答
        generation_model = self.rag_system.model_manager.get_model("generation")
        answer = generation_model.generate(prompt)

        return {
            'answer': answer,
            'prompt': prompt,
            'document_count': len(documents)
        }

    def _execute_conditional(self, step: Dict[str, Any], context: Dict[str, Any], **kwargs) -> Any:
        """执行条件判断步骤"""
        condition = step['condition']
        then_steps = step.get('then', [])
        else_steps = step.get('else', [])

        # 简化版条件评估 - 实际应使用更复杂的逻辑
        condition_met = self._evaluate_condition(condition, context)

        if condition_met:
            logger.info(f"Condition met, executing 'then' branch")
            return self._execute_steps(then_steps, context['query'], **kwargs)
        else:
            logger.info(f"Condition not met, executing 'else' branch")
            return self._execute_steps(else_steps, context['query'], **kwargs)

    def _execute_loop(self, step: Dict[str, Any], context: Dict[str, Any], **kwargs) -> Any:
        """执行循环步骤"""
        max_iterations = step.get('max_iterations', 3)
        steps_to_loop = step['steps']
        break_condition = step.get('break_condition')

        results = []
        for iteration in range(max_iterations):
            logger.info(f"Loop iteration {iteration + 1}/{max_iterations}")

            # 执行循环中的步骤
            iteration_result = self._execute_steps(
                steps_to_loop, context['query'], **kwargs
            )
            results.append(iteration_result)

            # 检查中断条件
            if break_condition and self._evaluate_condition(break_condition, {
                **context,
                'iteration': iteration,
                'iteration_result': iteration_result
            }):
                logger.info(f"Break condition met at iteration {iteration}")
                break

        return {'iterations': len(results), 'results': results}

    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        # 简化版条件评估
        condition_type = condition.get('type', 'document_count')

        if condition_type == 'document_count':
            min_docs = condition.get('min_documents', 1)
            doc_count = len(context.get('results', {}).get('retrieve', {}).get('documents', []))
            return doc_count >= min_docs

        elif condition_type == 'confidence_threshold':
            threshold = condition.get('threshold', 0.8)
            # 这里需要实际的置信度计算
            return True  # 简化实现

        return False


# 示例使用
if __name__ == "__main__":
    # 示例化并使用PipelineManager
    from core.rag_system import RAGSystem

    rag_system = RAGSystem()
    rag_system.initialize()

    pipeline_manager = PipelineManager(rag_system)

    # 加载并执行流水线
    pipeline_config = pipeline_manager.load_pipeline("workflows/vanilla_rag_pipeline.yaml")
    result = pipeline_manager.execute_pipeline("vanilla_rag", "什么是人工智能?")

    print(f"最终回答: {result['results']['generate']['answer']}")