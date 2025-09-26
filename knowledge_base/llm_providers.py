import openai
from openai import OpenAI
import dashscope
from dashscope import Generation
import requests
import json
from typing import Dict, Any, List, Optional
from config import Config


class LLMProvider:
    def __init__(self, config: Config):
        self.config = config
        self.setup_providers()

    def setup_providers(self):
        """设置各个API提供商"""
        if self.config.OPENAI_API_KEY:
            self.openai_client = OpenAI(
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_BASE
            )

        if self.config.TONGYI_API_KEY:
            dashscope.api_key = self.config.TONGYI_API_KEY

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """根据配置的模型类型生成响应"""
        model_type = self.config.LLM_MODEL.lower()

        if model_type == "openai":
            return self._openai_chat(messages, **kwargs)
        elif model_type == "tongyi":
            return self._tongyi_chat(messages, **kwargs)
        elif model_type == "azure":
            return self._azure_chat(messages, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def _openai_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """OpenAI API调用"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API调用失败: {str(e)}")

    def _tongyi_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """通义千问API调用"""
        try:
            # 确保温度参数合理
            temperature = kwargs.get('temperature', 0.7)
            temperature = max(0.1, min(temperature, 1.0))  # 限制在0.1-1.0之间

            response = Generation.call(
                model=self.config.TONGYI_MODEL,
                messages=messages,
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.8),
                max_tokens=min(kwargs.get('max_tokens', 1000), 8192),  # 限制在API允许范围内
                result_format='message'
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                error_msg = f"通义千问API错误: {response.message}"
                if response.code == "InvalidApiKey":
                    error_msg += "，请检查API密钥是否正确"
                raise Exception(error_msg)

        except Exception as e:
            raise Exception(f"通义千问API调用失败: {str(e)}")

    def _azure_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Azure OpenAI API调用"""
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.config.AZURE_API_KEY
            }

            payload = {
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.7),
                "max_tokens": kwargs.get('max_tokens', 1000),
                "stream": False
            }

            endpoint = f"{self.config.AZURE_API_BASE}/openai/deployments/{self.config.AZURE_DEPLOYMENT}/chat/completions?api-version={self.config.AZURE_API_VERSION}"

            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except Exception as e:
            raise Exception(f"Azure OpenAI API调用失败: {str(e)}")

    def stream_response(self, messages: List[Dict[str, str]], **kwargs):
        """基于所选模型提供商的流式响应生成器。

        目前实现 OpenAI 的原生流，逐步产出 content 片段（字符串）。
        对于未实现流的提供商，将抛出异常。
        """
        model_type = self.config.LLM_MODEL.lower()

        if model_type == "openai":
            try:
                stream = self.openai_client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=kwargs.get('max_tokens', 1000),
                    stream=True
                )

                for event in stream:
                    if not hasattr(event, 'choices') or not event.choices:
                        continue
                    delta = event.choices[0].delta
                    if delta and getattr(delta, 'content', None):
                        yield delta.content
            except Exception as e:
                raise Exception(f"OpenAI 流式接口调用失败: {str(e)}")

        elif model_type == "tongyi":
            # 通义千问流式输出
            try:
                stream = Generation.call(
                    model=self.config.TONGYI_MODEL,
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.8),
                    max_tokens=kwargs.get('max_tokens', 2000),
                    result_format='message',
                    stream=True,
                    incremental_output=True
                )

                for event in stream:
                    # 正常 token 增量
                    try:
                        if getattr(event, 'output', None) and event.output and \
                           getattr(event.output, 'choices', None) and event.output.choices:
                            message = event.output.choices[0].message
                            content = getattr(message, 'content', None)
                            if content:
                                yield content
                    except Exception:
                        # 忽略单次解析异常，继续流
                        continue
            except Exception as e:
                raise Exception(f"通义千问流式接口调用失败: {str(e)}")

        elif model_type == "azure":
            # 如需支持 Azure OpenAI 流式，可在此实现 SSE/流式解析
            raise Exception("当前未实现 Azure 的流式输出")

        else:
            raise ValueError(f"不支持的模型类型(流式): {model_type}")