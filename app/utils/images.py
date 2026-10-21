"""图像压缩与格式探测工具 (L0)"""

import base64
from io import BytesIO
from PIL import Image


def compress_image_to_base64(image_path: str, max_size=(1024, 1024), quality=80) -> str:
    """压缩图片并转换为 base64"""
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=quality)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error compressing image {image_path}: {e}")
        return ""


def guess_mime_type(data: bytes, default: str = "image/jpeg") -> str:
    """根据字节头判断图片的 MIME 类型"""
    if data.startswith(b"\x89PNG"):
        return "image/png"
    elif data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    elif data.startswith(b"GIF8"):
        return "image/gif"
    elif data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return default
