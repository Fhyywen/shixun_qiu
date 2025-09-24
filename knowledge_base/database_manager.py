# knowledge_base/database_manager.py
import mysql.connector
from mysql.connector import pooling
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging


class DatabaseManager:
    def __init__(self, config: dict):
        self.config = config
        self.pool = self._create_connection_pool()
        self.logger = logging.getLogger(__name__)

    def _create_connection_pool(self):
        """创建数据库连接池"""
        try:
            pool = pooling.MySQLConnectionPool(
                pool_name="chat_pool",
                pool_size=5,
                **self.config
            )
            return pool
        except Exception as e:
            self.logger.error(f"创建数据库连接池失败: {e}")
            raise

    def get_connection(self):
        """从连接池获取连接"""
        return self.pool.get_connection()

    def create_session(self, user_id: str = "anonymous", knowledge_base_path: str = None,
                       title: str = "新对话") -> str:
        """创建新的对话会话"""
        session_id = str(uuid.uuid4())

        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO chat_sessions (session_id, user_id, knowledge_base_path, title)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (session_id, user_id, knowledge_base_path, title))
                conn.commit()
            return session_id
        except Exception as e:
            self.logger.error(f"创建会话失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_message(self, session_id: str, role: str, content: str, metadata: dict = None):
        """添加消息到对话历史"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO chat_messages (session_id, role, content, metadata)
                VALUES (%s, %s, %s, %s)
                """
                metadata_json = json.dumps(metadata) if metadata else None
                cursor.execute(sql, (session_id, role, content, metadata_json))
                conn.commit()
        except Exception as e:
            self.logger.error(f"添加消息失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取对话历史"""
        conn = self.get_connection()
        try:
            with conn.cursor(dictionary=True) as cursor:
                sql = """
                SELECT role, content, metadata, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                LIMIT %s
                """
                cursor.execute(sql, (session_id, limit))
                messages = cursor.fetchall()

                # 转换为标准格式
                history = []
                for msg in messages:
                    history.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "metadata": json.loads(msg["metadata"]) if msg["metadata"] else {}
                    })
                return history
        except Exception as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []
        finally:
            conn.close()

    def update_session_title(self, session_id: str, title: str):
        """更新会话标题"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE chat_sessions SET title = %s, updated_at = NOW() WHERE session_id = %s"
                cursor.execute(sql, (title, session_id))
                conn.commit()
        except Exception as e:
            self.logger.error(f"更新会话标题失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户的会话列表"""
        conn = self.get_connection()
        try:
            with conn.cursor(dictionary=True) as cursor:
                sql = """
                SELECT s.session_id, s.title, s.knowledge_base_path, s.created_at, s.updated_at,
                       (SELECT content FROM chat_messages 
                        WHERE session_id = s.session_id AND role = 'user' 
                        ORDER BY created_at ASC LIMIT 1) as first_question
                FROM chat_sessions s
                WHERE s.user_id = %s AND s.is_active = TRUE
                ORDER BY s.updated_at DESC
                LIMIT %s
                """
                cursor.execute(sql, (user_id, limit))
                return cursor.fetchall()
        except Exception as e:
            self.logger.error(f"获取用户会话列表失败: {e}")
            return []
        finally:
            conn.close()

    def record_knowledge_base_usage(self, session_id: str, knowledge_base_path: str,
                                    question: str, similar_docs_count: int, average_similarity: float):
        """记录知识库使用情况"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO knowledge_base_usage 
                (session_id, knowledge_base_path, question, similar_docs_count, average_similarity)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (session_id, knowledge_base_path, question,
                                     similar_docs_count, average_similarity))
                conn.commit()
        except Exception as e:
            self.logger.error(f"记录知识库使用情况失败: {e}")
            conn.rollback()
        finally:
            conn.close()

    def close_session(self, session_id: str):
        """关闭会话（软删除）"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE chat_sessions SET is_active = FALSE WHERE session_id = %s"
                cursor.execute(sql, (session_id,))
                conn.commit()
        except Exception as e:
            self.logger.error(f"关闭会话失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()