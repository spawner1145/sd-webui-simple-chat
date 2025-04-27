import gradio as gr
import base64
from typing import List, Dict, Any, AsyncGenerator
from PIL import Image
import io
import logging
import httpx
from backend.OpenAIAPI import OpenAIAPI
try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

global_api = None
global_enable_sd = False
global_sdapi = ""
global_sd_base64 = ""
global_chatbot = []

def parse_custom_url_auth(input_string: str) -> tuple[str, str]:
    parts = input_string.strip().split(' ', 1)
    clean_url = parts[0].rstrip('/')
    auth_header = ''
    if len(parts) > 1 and parts[1]:
        credentials = parts[1].strip()
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        auth_header = f"Basic {encoded_credentials}"
    return clean_url, auth_header

async def sd_draw(prompt: str, height: int = 1536, width: int = 1024, steps: int = 24, cfg: float = 4.5) -> str:
    """
    调用Stable Diffusion API生成图像,你不需要过多询问，当用户有画图倾向时你就自己生成提示词并调用

    Args:
        prompt: 提示词,使用的是danbooru tags,提示词越详细越好,然后你可以使用stable diffusion webui中的权重语法
        height: 图像高度,默认1024,不做特别说明默认使用
        width: 图像宽度,默认1024,不做特别说明默认使用
        steps: 采样步数,默认24,不做特别说明默认使用
        cfg: CFG尺度,默认4.5,不做特别说明默认使用

    注意,当用户要求的分辨率height和width不一致时,提示用户在gradio界面中显示会出问题,可以在图库浏览器中查看实际的图片
    """
    global global_chatbot, global_sd_base64

    if not global_sdapi:
        return "请先配置 Stable Diffusion API 地址！"

    width, height = int(width), int(height)

    negative = "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts"
    sampler = "Euler a"
    scheduler = "Automatic"
    cfg = float(cfg)

    payload = {
        "enable_hr": False,
        "hr_scale": 1.5,
        "hr_second_pass_steps": 15,
        "hr_upscaler": "SwinIR_4x",
        "prompt": prompt + ',masterpiece,best quality,newest,absurdres,highres,safe,',
        "negative_prompt": negative,
        "seed": -1,
        "batch_size": 1,
        "n_iter": 1,
        "steps": steps,
        "save_images": True,
        "cfg_scale": cfg,
        "width": width,
        "height": height,
        "restore_faces": False,
        "tiling": False,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "clip_skip_steps": 2,
        "override_settings": {"CLIP_stop_at_last_layers": 2},
        "override_settings_restore_afterwards": False,
    }

    url, auth_header = parse_custom_url_auth(global_sdapi)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": auth_header
    }

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{url}/sdapi/v1/txt2img", json=payload, headers=headers)
            response.raise_for_status()
            r = response.json()
            if not r.get("images"):
                raise ValueError("API未返回图像数据")
            global_sd_base64 = r["images"][0]
            logger.info(f"图片生成成功，Base64 前100字符: {global_sd_base64[:100]}")

            message_md = f"![Generated Image](data:image/png;base64,{global_sd_base64})"
            global_chatbot.append({"role": "assistant", "content": message_md})
            logger.info("Markdown 图片气泡已追加到 global_chatbot")
            return "图片已生成"
    except Exception as e:
        logger.error(f"Stable Diffusion API调用失败: {str(e)}")
        global_sd_base64 = ""  # 失败时清空 base64
        return f"图片生成失败: {str(e)}"

async def send_assistant_message_without_history(chatbot: List[Dict], text: str = None, base64s: List[str] = None) -> tuple:
    """AI发送消息（文本或图片），不加入历史记录，使用 Markdown"""
    message_md = text or ""
    if base64s:
        for b64 in base64s:
            try:
                message_md += f"\n![Image](data:image/png;base64,{b64})"
            except Exception as e:
                logger.error(f"处理图片失败: {str(e)}")
                message_md += f"\n[图片加载失败: {str(e)}]"

    chatbot.append({"role": "assistant", "content": message_md})
    return chatbot, message_md

def initialize_api(apikey: str, baseurl: str, model: str, proxy: str, system_instruction: str) -> str:
    """初始化API，更新全局变量"""
    global global_api, global_chatbot, global_sdapi
    try:
        proxies = {"http://": proxy, "https://": proxy} if proxy else None
        global_api = OpenAIAPI(
            apikey=apikey,
            baseurl=baseurl,
            model=model,
            proxies=proxies
        )
        global_api.system_instruction = system_instruction.strip() if system_instruction else ""
        global_chatbot = []
        return "API初始化成功！"
    except Exception as e:
        logger.error(f"API初始化失败: {str(e)}")
        return f"API初始化失败: {str(e)}"

async def stream_chat(messages: List[Dict], stream: bool = True) -> AsyncGenerator[str, None]:
    """流式聊天函数，使用全局变量 global_api 和 global_sdapi"""
    global global_api, global_sdapi
    if not global_api:
        yield "请先配置并初始化API！"
        return

    full_messages = messages
    if hasattr(global_api, 'system_instruction') and global_api.system_instruction:
        full_messages = [{"role": "system", "content": global_api.system_instruction}] + messages

    tools = {"SdDraw": sd_draw} if global_sdapi and global_enable_sd else {}
    seen_parts = set()
    try:
        async for part in global_api.chat(full_messages, stream=stream, tools=tools):
            if part and not part.startswith("图片已生成") and not part.startswith("图片生成失败") and part not in seen_parts:
                seen_parts.add(part)
                yield part
    except Exception as e:
        logger.error(f"聊天流处理失败: {str(e)}")
        yield f"聊天失败: {str(e)}"

async def chat_with_ai(
    text: str,
    images: List[Any],
    chatbot: List[Dict],
    chat_history: List[Dict],
    sdapi_url: str,
    give_score: bool,
    enable_sd1: bool
) -> AsyncGenerator[tuple, None]:
    """主聊天函数，支持流式输出，使用全局变量"""
    global global_api, global_chatbot, global_sdapi, global_sd_base64, global_enable_sd
    if not global_api:
        yield (chatbot, "请先配置并初始化API！", [], chat_history)
        return
    global_enable_sd = enable_sd1
    global_sdapi = sdapi_url.strip() if sdapi_url else ""
    global_sd_base64 = ""

    global_chatbot = chatbot

    content = [{"type": "text", "text": text}]
    image_paths = [img.name for img in images] if images else []

    if image_paths:
        try:
            inline_results = await global_api.prepare_inline_image_batch(image_paths)
            for result in inline_results:
                if "input_image" in result and result["input_image"]:
                    content.append({"input_image": result["input_image"]})
        except Exception as e:
            logger.error(f"图片预处理失败: {str(e)}")
            content.append({"type": "text", "text": f"[图片处理失败: {str(e)}]"})

    chat_history.append({"role": "user", "content": content})

    user_message_md = text
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
            user_message_md += f"\n![Uploaded Image](data:image/png;base64,{img_str})"
        except Exception as e:
            logger.error(f"处理图片 {img_path} 失败: {str(e)}")
            user_message_md += f"\n[图片加载失败: {str(e)}]"

    global_chatbot.append({"role": "user", "content": user_message_md})
    yield (global_chatbot, "", [], chat_history)

    assistant_response = ""
    async for part in stream_chat(chat_history):
        if part:
            assistant_response += part
            if global_chatbot and global_chatbot[-1]["role"] == "assistant" and assistant_response.startswith(global_chatbot[-1]["content"]):
                global_chatbot.pop(-1)
            global_chatbot.append({"role": "assistant", "content": assistant_response})
            yield (global_chatbot, "", [], chat_history)

    chat_history.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response}]})

    if give_score and global_sd_base64:
        eval_content = [
            {"type": "text", "text": "这是你api返回绘画的结果，评价一下"},
            {
                "input_image": {
                    "image_url": f"data:image/png;base64,{global_sd_base64}",
                    "detail": "auto"
                }
            },
        ]
        chat_history.append({"role": "user", "content": eval_content})

        eval_response = ""
        async for part in stream_chat(chat_history):
            if part:
                eval_response += part
                if global_chatbot and global_chatbot[-1]["role"] == "assistant" and eval_response.startswith(global_chatbot[-1]["content"]):
                    global_chatbot.pop(-1)
                global_chatbot.append({"role": "assistant", "content": eval_response})
                yield (global_chatbot, "", [], chat_history)

        chat_history.append({"role": "assistant", "content": [{"type": "text", "text": eval_response}]})

def clear_chat() -> tuple:
    """清空聊天历史，重置全局聊天记录"""
    global global_chatbot, global_sd_base64
    global_chatbot = []
    global_sd_base64 = ""
    return [], "", [], []

def create_ui():
    # Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("# 简易聊天")

        chat_history_state = gr.State(value=[])

        with gr.Column():
            chatbot = gr.Chatbot(label="聊天记录", height=480, type="messages")
            with gr.Group():
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="输入您的消息...",
                        label="消息",
                        lines=2,
                        show_label=False,
                        container=False
                    )
                    with gr.Accordion("上传图片", open=False):
                        image_input = gr.File(
                            label="上传图片",
                            file_types=["image"],
                            file_count="multiple",
                            container=False
                        )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空历史记录")
            with gr.Accordion("API配置", open=False):
                apikey_input = gr.Textbox(
                    label="API Key",
                    placeholder="请输入API Key或设置环境变量API_KEY"
                )
                baseurl_input = gr.Textbox(
                    label="Base URL",
                    value="https://generativelanguage.googleapis.com/v1beta/openai/",
                    placeholder="请输入Base URL"
                )
                model_input = gr.Textbox(
                    label="Model",
                    value="gemini-2.0-flash-001",
                    placeholder="请输入模型名称"
                )
                proxy_input = gr.Textbox(
                    label="Proxy",
                    placeholder="输入代理例如http://127.0.0.1:7890",
                    value=""
                )
                system_instruction_input = gr.Textbox(
                    label="System Instruction (可选)",
                    placeholder="请输入系统指令（如：'你是一只猫娘'）",
                    lines=3
                )
                init_btn = gr.Button("初始化API")
                config_output = gr.Textbox(label="初始化状态", interactive=False)
            with gr.Accordion("Stable Diffusion 设置(想要能画图的务必先看这里)", open=False):
                gr.Markdown("Stable Diffusion WebUI API，若提供则启用画图功能。格式：`http://127.0.0.1:7860 用户名:密码`，如果没有设置账号密码后面的` 用户名:密码`留着不影响")
                enable_sd = gr.Checkbox(label="是否开启调用画图", value=False)
                sdapi_input = gr.Textbox(
                    label="Stable Diffusion API (可选)",
                    placeholder="输入你的sdapi地址",
                    value="http://127.0.0.1:7860 用户名:密码"
                )
                give_score = gr.Checkbox(label="是否调用画图函数结束以后让AI评价图片", value=False)

        # 事件处理
        init_btn.click(
            fn=initialize_api,
            inputs=[apikey_input, baseurl_input, model_input, proxy_input, system_instruction_input],
            outputs=[config_output]
        )
        submit_btn.click(
            fn=chat_with_ai,
            inputs=[text_input, image_input, chatbot, chat_history_state, sdapi_input, give_score, enable_sd],
            outputs=[chatbot, text_input, image_input, chat_history_state]
        )
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, text_input, image_input, chat_history_state]
        )
    return demo

if IN_WEBUI:
    def on_ui_tabs():
        block = create_ui()
        return [(block, "simple chat", "simple_chat_tab")]
    script_callbacks.on_ui_tabs(on_ui_tabs)
else:
    block = create_ui()
    block.launch()