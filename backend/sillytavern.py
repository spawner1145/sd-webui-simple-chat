import re
from PIL import Image
import base64
import html
import os

async def use_folder_chara(file_name):
    try:
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            return silly_tavern_card(file_name, clear_html=True)
        else:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    return f.read()
            except:
                return f"不支持的文件类型: {file_name}"
    except FileNotFoundError:
        return f"文件不存在: {file_name}"
    except IOError as e:
        return f"读取文件失败: {str(e)}"
    except Exception as e:
        return f"处理角色卡失败: {str(e)}"

def clean_invalid_characters(s, clear_html=False):
    """
    清理字符串中的无效控制字符，并根据需要移除HTML标签及其内容，以及前面可能存在的'xxx:'或'xxx：'前缀。
    """
    cleaned = ''.join(c for c in s if ord(c) >= 32 or c in ('\t', '\n', '\r'))
    if clear_html:
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r'<[^>]+>.*?</[^>]+>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+?/>', '', cleaned)
        cleaned = re.sub(r'^.*?(?=:|：)', '', cleaned).lstrip(':： ').lstrip()
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n\s+', '\n', cleaned)

    return cleaned.strip()


def silly_tavern_card(image_path, clear_html=False):
    image = Image.open(image_path)
    # 打印基本信息
    # print("图片基本信息:")
    # print(f"格式: {image.format}")
    # print(f"大小: {image.size}")
    # print(f"模式: {image.mode}")

    # 打印所有图像信息
    # print("\n所有图像信息:")
    # for key, value in image.info.items():
    # print(f"{key}: {value}")

    # 尝试打印文本块
    try:
        print("\n文本块信息:")
        for k, v in image.text.items():
            print(f"{k}: {len(v)} 字符")
            # 如果文本很长，只打印前100个字符
            print(f"预览: {v[:100]}...")
            pass
    except AttributeError:
        return "错误，没有文本块信息"

    final = []

    # 尝试解码 base64
    try:
        for key, value in image.info.items():
            if isinstance(value, str) and 'chara' in key.lower():
                print(f"\n尝试解码 {key} 的 base64:")
                decoded = base64.b64decode(value)
                res = decoded.decode('utf-8', errors='ignore')
                final.append(res)

    except Exception as e:
        return (f"错误，解码失败: {e}")

    if final:
        s = "\n".join(final)
        return clean_invalid_characters(s, clear_html=False)
    else:
        return "错误，没有人设信息"
