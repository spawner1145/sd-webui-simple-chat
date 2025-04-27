#==========================================================================================
# 代码见https://github.com/spawner1145/llm-api-backup.git
#==========================================================================================
import httpx
import json
import mimetypes
import asyncio
import base64
import os
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Union, Callable
import aiofiles
import logging
import tempfile
from openai import AsyncOpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAIAPI:
    def __init__(
        self,
        apikey: str,
        baseurl: str = "https://api-inference.modelscope.cn",
        model: str = "deepseek-ai/DeepSeek-R1",
        proxies: Optional[Dict[str, str]] = None
    ):
        self.apikey = apikey
        self.baseurl = baseurl.rstrip('/')
        self.model = model
        self.client = AsyncOpenAI(
            api_key=apikey,
            base_url=baseurl,
            http_client=httpx.AsyncClient(proxies=proxies, timeout=60.0) if proxies else None
        )

    async def upload_file(self, file_path: str, display_name: Optional[str] = None) -> Dict[str, Union[str, None]]:
        """上传单个文件，使用 client.files.create，目的为 user_data"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 32 * 1024 * 1024:  # 32MB 限制
                raise ValueError(f"文件 {file_path} 大小超过 32MB 限制")
        except FileNotFoundError:
            logger.error(f"文件 {file_path} 不存在")
            return {"fileId": None, "mimeType": None, "error": f"文件 {file_path} 不存在"}

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
            logger.warning(f"无法检测文件 {file_path} 的 MIME 类型，使用默认值: {mime_type}")

        supported_mime_types = [
            "application/pdf", "image/jpeg", "image/png", "image/webp", "image/gif"
        ]
        if mime_type not in supported_mime_types:
            logger.warning(f"MIME 类型 {mime_type} 可能不受支持，可能导致处理失败")

        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file = await self.client.files.create(
                    file=(display_name or os.path.basename(file_path), await f.read(), mime_type),
                    purpose="user_data"
                )
                file_id = file.id
                logger.info(f"文件 {file_path} 上传成功，ID: {file_id}")
                return {"fileId": file_id, "mimeType": mime_type, "error": None}
        except Exception as e:
            logger.error(f"文件 {file_path} 上传失败: {str(e)}")
            return {"fileId": None, "mimeType": mime_type, "error": str(e)}

    async def upload_files(self, file_paths: List[str], display_names: Optional[List[str]] = None) -> List[Dict[str, Union[str, None]]]:
        """并行上传多个文件"""
        if not file_paths:
            raise ValueError("文件路径列表不能为空")

        if display_names and len(display_names) != len(file_paths):
            raise ValueError("display_names 长度必须与 file_paths 一致")

        tasks = [self.upload_file(file_paths[idx], display_names[idx] if display_names else None) for idx in range(len(file_paths))]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"上传文件 {file_paths[idx]} 失败: {str(result)}")
                final_results.append({"fileId": None, "mimeType": None, "error": str(result)})
            else:
                final_results.append(result)
        return final_results

    async def prepare_inline_image(self, file_path: str, detail: str = "auto") -> Dict[str, Union[Dict, None]]:
        """将单个图片转换为 Base64 编码的 input_image"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 20 * 1024 * 1024:  # 20MB 限制
                raise ValueError(f"文件 {file_path} 过大，超过 20MB 限制")

            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type or mime_type not in ["image/jpeg", "image/png", "image/webp", "image/gif"]:
                mime_type = "image/jpeg"
                logger.warning(f"无效图片 MIME 类型，使用默认值: {mime_type}")

            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            base64_data = base64.b64encode(file_content).decode('utf-8')
            return {
                "input_image": {
                    "image_url": f"data:{mime_type};base64,{base64_data}",
                    "detail": detail
                }
            }
        except Exception as e:
            logger.error(f"处理图片 {file_path} 失败: {str(e)}")
            return {"input_image": None, "error": str(e)}

    async def prepare_inline_image_batch(self, file_paths: List[str], detail: str = "auto") -> List[Dict[str, Union[Dict, None]]]:
        """将多个图片转换为 Base64 编码的 input_image 列表"""
        if not file_paths:
            raise ValueError("文件路径列表不能为空")

        results = []
        for file_path in file_paths:
            result = await self.prepare_inline_image(file_path, detail)
            results.append(result)
        return results

    async def _execute_tool(
        self,
        tool_calls: List[Dict],
        tools: Dict[str, Callable]
    ) -> List[Dict]:
        """执行工具调用并返回响应，遵循 OpenAI 格式"""
        tool_responses = []
        for tool_call in tool_calls:
            name = tool_call.function.name
            if not name:
                logger.error(f"工具调用缺少名称: {tool_call}")
                continue
            tool_call_id = tool_call.id or f"call_{uuid.uuid4()}"
            args = json.loads(tool_call.function.arguments)
            logger.info(f"执行工具调用: {name}, 参数: {args}, ID: {tool_call_id}")
            func = tools.get(name)
            if func:
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = func(**args)
                    logger.info(f"工具结果: {name} 返回 {result}, ID: {tool_call_id}")
                    tool_response = {
                        "role": "tool",
                        "content": json.dumps({"result": result}),  # 包装为 JSON
                        "tool_call_id": tool_call_id
                    }
                    tool_responses.append((tool_response, tool_call_id))
                except Exception as e:
                    result = f"函数 {name} 执行失败: {str(e)}"
                    logger.error(f"工具错误: {result}, ID: {tool_call_id}")
                    tool_response = {
                        "role": "tool",
                        "content": json.dumps({"error": result}),
                        "tool_call_id": tool_call_id
                    }
                    tool_responses.append((tool_response, tool_call_id))
            else:
                logger.error(f"未找到工具: {name}, ID: {tool_call_id}")
                tool_response = {
                    "role": "tool",
                    "content": json.dumps({"error": f"未找到工具 {name}"}),
                    "tool_call_id": tool_call_id
                }
                tool_responses.append((tool_response, tool_call_id))
        return tool_responses

    async def _chat_api(
        self,
        messages: List[Dict],
        stream: bool,
        tools: Optional[Dict[str, Callable]] = None,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        response_format: Optional[Dict] = None,
        seed: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
        logprobs: Optional[int] = None,
        retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """核心 API 调用逻辑，遵循 OpenAI 标准，支持 reasoning_content 但不记录到历史"""
        original_model = self.model

        # 验证参数
        if topp is not None and (topp < 0 or topp > 1):
            raise ValueError("top_p 必须在 0 到 1 之间")
        if temperature is not None and (temperature < 0 or temperature > 2):
            raise ValueError("temperature 必须在 0 到 2 之间")
        if presence_penalty is not None and (presence_penalty < -2 or presence_penalty > 2):
            raise ValueError("presence_penalty 必须在 -2 到 2 之间")
        if frequency_penalty is not None and (frequency_penalty < -2 or frequency_penalty > 2):
            raise ValueError("frequency_penalty 必须在 -2 到 2 之间")
        if logprobs is not None and (logprobs < 0 or logprobs > 20):
            raise ValueError("logprobs 必须在 0 到 20 之间")

        # 构造消息
        api_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                api_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                api_content = []
                for part in content:
                    if "text" in part:
                        api_content.append({"type": "text", "text": part["text"]})
                    elif "input_file" in part:
                        api_content.append({
                            "type": "input_file",
                            "file_id": part["input_file"]["file_id"]
                        } if "file_id" in part["input_file"] else {
                            "type": "input_file",
                            "filename": part["input_file"]["filename"],
                            "file_data": part["input_file"]["file_data"]
                        })
                    elif "input_image" in part:
                        api_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": part["input_image"]["image_url"],
                                "detail": part["input_image"].get("detail", "auto")
                            }
                        })
            else:
                raise ValueError(f"无效的消息内容格式: {content}")
            api_msg = {
                "role": role,
                "content": api_content
            }
            if "tool_calls" in msg:
                api_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    } for tc in msg["tool_calls"]
                ]
            if "tool_call_id" in msg:
                api_msg["tool_call_id"] = msg["tool_call_id"]
            logger.debug(f"构造消息: {json.dumps(api_msg, ensure_ascii=False)}")
            api_messages.append(api_msg)

        # 构造请求参数
        request_params = {
            "model": self.model,
            "messages": api_messages,
            "stream": stream
        }
        if max_output_tokens is not None:
            request_params["max_tokens"] = max_output_tokens
        if topp is not None:
            request_params["top_p"] = topp
        if temperature is not None:
            request_params["temperature"] = temperature
        if stop_sequences is not None:
            request_params["stop"] = stop_sequences
        if presence_penalty is not None:
            request_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            request_params["frequency_penalty"] = frequency_penalty
        if seed is not None:
            request_params["seed"] = seed
        if response_logprobs is not None:
            request_params["logprobs"] = response_logprobs
            if logprobs is not None:
                request_params["top_logprobs"] = logprobs
        if response_format:
            request_params["response_format"] = response_format

        if tools is not None:
            tool_definitions = []
            for name, func in tools.items():
                params = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
                if hasattr(func, "__code__"):
                    param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                    for param in param_names:
                        params["properties"][param] = {"type": "string"}
                        params["required"].append(param)
                else:
                    params["properties"] = {"arg": {"type": "string"}}
                    params["required"] = ["arg"]
                tool_definitions.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": getattr(func, "__doc__", f"调用 {name} 函数"),
                        "parameters": params
                    }
                })
            request_params["tools"] = tool_definitions

        if stream:
            assistant_content = ""
            tool_calls_buffer = []
            async for chunk in await self.client.chat.completions.create(**request_params):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        yield f"REASONING: {delta.reasoning_content}"
                    if delta.content:
                        yield delta.content
                        assistant_content += delta.content
                    elif delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call and tool_call.function:
                                tool_call_id = tool_call.id or f"call_{uuid.uuid4()}"
                                try:
                                    arguments = tool_call.function.arguments or "{}"
                                    json.loads(arguments)
                                    tool_calls_buffer.append({
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": arguments
                                        }
                                    })
                                    logger.info(f"工具调用: {tool_call.function.name}, 参数: {arguments}, ID: {tool_call_id}")
                                except json.JSONDecodeError:
                                    logger.error(f"工具调用 {tool_call.function.name} 的 arguments 无效: {arguments}")
                                    continue
                        if chunk.choices[0].finish_reason == "tool_calls" and tool_calls_buffer:
                            assistant_message = {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "Tool calls executed"}],
                                "tool_calls": tool_calls_buffer
                            }
                            api_messages.append(assistant_message)
                            messages.append(assistant_message)
                            tool_responses = await self._execute_tool(
                                [
                                    type('ToolCall', (), {
                                        'id': tc["id"],
                                        'function': type('Function', (), {
                                            'name': tc["function"]["name"],
                                            'arguments': tc["function"]["arguments"]
                                        })()
                                    })() for tc in tool_calls_buffer
                                ],
                                tools
                            )
                            for tool_response, tool_call_id in tool_responses:
                                tool_message = {
                                    "role": "tool",
                                    "content": tool_response["content"],
                                    "tool_call_id": tool_call_id
                                }
                                api_messages.append(tool_message)
                                messages.append(tool_message)
                            second_request_params = request_params.copy()
                            second_request_params["messages"] = api_messages
                            second_request_params["stream"] = True  # 保持流式
                            async for second_chunk in await self.client.chat.completions.create(**second_request_params):
                                if second_chunk.choices:
                                    second_delta = second_chunk.choices[0].delta
                                    if hasattr(second_delta, 'reasoning_content') and second_delta.reasoning_content:
                                        yield f"REASONING: {second_delta.reasoning_content}"
                                    if second_delta.content:
                                        yield second_delta.content
                                        assistant_content += second_delta.content
                                    if second_chunk.choices[0].finish_reason in ["stop", "length"]:
                                        if assistant_content:
                                            messages.append({
                                                "role": "assistant",
                                                "content": [{"type": "text", "text": assistant_content}]
                                            })
                                        assistant_content = ""
                            tool_calls_buffer = []
                    if chunk.choices[0].finish_reason in ["stop", "length"]:
                        if assistant_content:
                            messages.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": assistant_content}]
                            })
                        assistant_content = ""
        else:
            for attempt in range(retries):
                try:
                    response = await self.client.chat.completions.create(**request_params)
                    choice = response.choices[0]
                    message = choice.message
                    if message.tool_calls:
                        tool_calls = [
                            {
                                "id": tc.id or f"call_{uuid.uuid4()}",
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in message.tool_calls
                        ]
                        assistant_message = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Tool calls executed"}],
                            "tool_calls": tool_calls
                        }
                        api_messages.append(assistant_message)
                        messages.append(assistant_message)
                        tool_responses = await self._execute_tool(message.tool_calls, tools)
                        for tool_response, tool_call_id in tool_responses:
                            tool_message = {
                                "role": "tool",
                                "content": tool_response["content"],
                                "tool_call_id": tool_call_id
                            }
                            api_messages.append(tool_message)
                            messages.append(tool_message)
                        second_request_params = request_params.copy()
                        second_request_params["messages"] = api_messages
                        second_request_params["stream"] = False
                        response = await self.client.chat.completions.create(**second_request_params)
                        choice = response.choices[0]
                        message = choice.message
                        assistant_message = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": message.content or ""}]
                        }
                        messages.append(assistant_message)
                        if hasattr(message, 'reasoning_content') and message.reasoning_content:
                            yield f"REASONING: {message.reasoning_content}"
                        if message.content:
                            yield message.content
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": message.content or ""}]
                        }
                        if response_logprobs and choice.logprobs:
                            assistant_message["logprobs"] = choice.logprobs.content
                            messages.append(assistant_message)
                            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                                yield f"REASONING: {message.reasoning_content}"
                            yield f"{message.content or ''}\nLogprobs: {json.dumps(choice.logprobs.content, ensure_ascii=False)}"
                        else:
                            messages.append(assistant_message)
                            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                                yield f"REASONING: {message.reasoning_content}"
                            if message.content:
                                yield message.content
                    break
                except Exception as e:
                    logger.error(f"API 调用失败 (尝试 {attempt+1}/{retries}): {str(e)}")
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)

        self.model = original_model

    async def chat(
        self,
        messages: Union[str, List[Dict[str, any]]],
        stream: bool = False,
        tools: Optional[Dict[str, Callable]] = None,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        topp: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        response_format: Optional[Dict] = None,
        seed: Optional[int] = None,
        response_logprobs: Optional[bool] = None,
        logprobs: Optional[int] = None,
        retries: int = 3
    ) -> AsyncGenerator[str, None]:
        """发起聊天请求，支持多文件和多图片输入"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": messages}]}]
        if system_instruction:
            for i, message in enumerate(messages):
                if message.get("role") == "system":
                    messages[i] = {"role": "system", "content": [{"type": "text", "text": system_instruction}]}
                    break
            else:
                messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_instruction}]})

        async for part in self._chat_api(
            messages, stream, tools, max_output_tokens,
            system_instruction, topp, temperature,
            presence_penalty, frequency_penalty,
            stop_sequences, response_format,
            seed, response_logprobs, logprobs,
            retries
        ):
            yield part

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()