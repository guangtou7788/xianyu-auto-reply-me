"""
AI回复引擎模块
集成XianyuAutoAgent的AI回复功能到现有项目中
"""

import os
import json
import time
import sqlite3
from typing import List, Dict, Optional
from loguru import logger
import requests
from openai import OpenAI
from db_manager import db_manager


class DashScopeClient:
    """封装阿里云DashScope API的客户端"""
    def __init__(self, api_key: str, app_id: str):
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = f"https://dashscope.aliyuncs.com/api/v1/apps/{self.app_id}/completion"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, messages: List[Dict], parameters: Optional[Dict] = None):
        """调用DashScope的补全接口"""
        if parameters is None:
            parameters = {}

        prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt += f"{msg['content']}\n"
            elif msg['role'] == 'user':
                prompt += f"用户: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"助手: {msg['content']}\n"

        data = {
            "input": {
                "prompt": prompt
            },
            "parameters": parameters,
            "debug": {}
        }

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get('code'):
                raise Exception(f"DashScope API错误: {result.get('message')}")

            if result.get('output') and result['output'].get('text'):
                return result['output']['text']
            else:
                return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"DashScope API请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"DashScope API调用异常: {e}")
            raise


class AIReplyEngine:
    """AI回复引擎"""

    def __init__(self):
        self.clients = {}
        self.agents = {}
        self._init_default_prompts()

    def _init_default_prompts(self):
        """初始化默认提示词"""
        self.default_prompts = {
            'classify': '''请根据客户消息内容，判断其意图类型。
            - tech: 技术或服务相关
            - default: 预约或一般咨询

            只返回意图类型，不要其他内容。''',

            'tech': '''请回答客户关于牙齿护理、口腔健康、服务流程等专业问题。''',

            'default': '''请专注于处理客户的预约请求和常见问题。你的首要任务是确认客户意向的门店和时间，并询问个人信息以完成预约。'''
        }

    def get_client(self, cookie_id: str):
        """获取指定账号的AI客户端，根据配置选择不同服务商"""
        if cookie_id not in self.clients:
            settings = db_manager.get_ai_reply_settings(cookie_id)
            if not settings['ai_enabled'] or not settings['api_key']:
                return None

            api_key = settings['api_key']
            base_url = settings.get('base_url')

            try:
                if base_url and "dashscope.aliyuncs.com" in base_url:
                    try:
                        app_id = base_url.split("apps/")[1].split("/")[0]
                        if not app_id:
                            logger.error(f"DashScope APP ID 未在 base_url 中找到 {cookie_id}")
                            return None
                    except IndexError:
                        logger.error(f"DashScope API地址格式不正确 {cookie_id}")
                        return None

                    self.clients[cookie_id] = DashScopeClient(
                        api_key=api_key,
                        app_id=app_id
                    )
                    logger.info(f"为账号 {cookie_id} 创建DashScope客户端")
                else:
                    self.clients[cookie_id] = OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
                    logger.info(f"为账号 {cookie_id} 创建OpenAI客户端")
            except Exception as e:
                logger.error(f"创建AI客户端失败 {cookie_id}: {e}")
                return None

        return self.clients.get(cookie_id)

    def is_ai_enabled(self, cookie_id: str) -> bool:
        """检查指定账号是否启用AI回复"""
        settings = db_manager.get_ai_reply_settings(cookie_id)
        return settings['ai_enabled']

    def detect_intent(self, message: str, cookie_id: str) -> str:
        """检测用户消息意图"""
        client = self.get_client(cookie_id)
        if not client:
            return 'default'

        try:
            settings = db_manager.get_ai_reply_settings(cookie_id)
            custom_prompts = json.loads(settings['custom_prompts']) if settings['custom_prompts'] else {}
            classify_prompt = custom_prompts.get('classify', self.default_prompts['classify'])

            messages = [
                {"role": "system", "content": classify_prompt},
                {"role": "user", "content": message}
            ]

            if isinstance(client, DashScopeClient):
                response_text = client.chat_completion(messages)
            else:
                response = client.chat.completions.create(
                    model=settings['model_name'],
                    messages=messages,
                    max_tokens=10,
                    temperature=0.1
                )
                response_text = response.choices[0].message.content.strip()

            intent = response_text.lower()
            if intent in ['tech', 'default']:
                return intent
            else:
                return 'default'

        except Exception as e:
            logger.error(f"意图检测失败 {cookie_id}: {e}")
            return 'default'

    def generate_reply(self, message: str, item_info: dict, chat_id: str,
                      cookie_id: str, user_id: str, item_id: str) -> Optional[str]:
        """生成AI回复"""
        if not self.is_ai_enabled(cookie_id):
            return None

        client = self.get_client(cookie_id)
        if not client:
            return None

        try:
            settings = db_manager.get_ai_reply_settings(cookie_id)
            is_dashscope = isinstance(client, DashScopeClient)

            intent = self.detect_intent(message, cookie_id)
            logger.info(f"检测到意图: {intent} (账号: {cookie_id})")

            context = self.get_conversation_context(chat_id, cookie_id)

            custom_prompts = json.loads(settings['custom_prompts']) if settings['custom_prompts'] else {}
            system_prompt = custom_prompts.get(intent, self.default_prompts[intent])

            item_desc = f"商品标题: {item_info.get('title', '未知')}\n"
            item_desc += f"商品价格: {item_info.get('price', '未知')}元\n"
            item_desc += f"商品描述: {item_info.get('desc', '无')}"

            context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context[-10:]])

            user_prompt = f"""商品信息：
{item_desc}
对话历史：
{context_str}
用户消息：{message}
请根据以上信息生成回复："""

            messages_to_send = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            if is_dashscope:
                reply = client.chat_completion(messages_to_send)
            else:
                response = client.chat.completions.create(
                    model=settings['model_name'],
                    messages=messages_to_send,
                    max_tokens=100,
                    temperature=0.7
                )
                reply = response.choices[0].message.content.strip()

            self.save_conversation(chat_id, cookie_id, user_id, item_id, "user", message, intent)
            self.save_conversation(chat_id, cookie_id, user_id, item_id, "assistant", reply, intent)

            logger.info(f"AI回复生成成功 (账号: {cookie_id}): {reply}")
            return reply

        except Exception as e:
            logger.error(f"AI回复生成失败 {cookie_id}: {e}")
            return None

    def get_conversation_context(self, chat_id: str, cookie_id: str, limit: int = 20) -> List[Dict]:
        """获取对话上下文"""
        try:
            with db_manager.lock:
                cursor = db_manager.conn.cursor()
                cursor.execute('''
                SELECT role, content FROM ai_conversations 
                WHERE chat_id = ? AND cookie_id = ? 
                ORDER BY created_at DESC LIMIT ?
                ''', (chat_id, cookie_id, limit))

                results = cursor.fetchall()
                context = [{"role": row[0], "content": row[1]} for row in reversed(results)]
                return context
        except Exception as e:
            logger.error(f"获取对话上下文失败: {e}")
            return []

    def save_conversation(self, chat_id: str, cookie_id: str, user_id: str,
                         item_id: str, role: str, content: str, intent: str = None):
        """保存对话记录"""
        try:
            with db_manager.lock:
                cursor = db_manager.conn.cursor()
                cursor.execute('''
                INSERT INTO ai_conversations 
                (cookie_id, chat_id, user_id, item_id, role, content, intent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (cookie_id, chat_id, user_id, item_id, role, content, intent))
                db_manager.conn.commit()
        except Exception as e:
            logger.error(f"保存对话记录失败: {e}")

    def get_bargain_count(self, chat_id: str, cookie_id: str) -> int:
        """此方法已弃用，议价功能已移除"""
        return 0

    def increment_bargain_count(self, chat_id: str, cookie_id: str):
        """此方法已弃用，议价功能已移除"""
        pass

    def clear_client_cache(self, cookie_id: str = None):
        """清理客户端缓存"""
        if cookie_id:
            self.clients.pop(cookie_id, None)
            logger.info(f"清理账号 {cookie_id} 的客户端缓存")
        else:
            self.clients.clear()
            logger.info("清理所有客户端缓存")


# 全局AI回复引擎实例
ai_reply_engine = AIReplyEngine()
