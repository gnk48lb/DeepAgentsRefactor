"""QQ 上传文件的纯函数工具：文件名清洗、下载地址校验、附件提取 (所在层: L0，只依赖标准库)

全部是纯函数 / 数据类，不做网络、不碰磁盘；除 validate_session_subdir（按约定抛 ValueError）外永不抛异常。
下载地址是带签名的凭证：这里的函数既不打印也不把它放进任何异常文本。
"""

import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit


@dataclass(frozen=True)
class IncomingFile:
    url: str                 # 带签名的下载地址：只在下载那一刻使用；绝不写日志 / 事件 / 回执
    filename: str = ""       # QQ 给的原始文件名，可能为空
    size: int = 0            # QQ 声明的大小（字节），0 = 未知
    content_type: str = ""


_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,10}")
_SESSION_SUBDIR_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
_FORBIDDEN_NAME_CHARS = frozenset('<>:"/\\?*|')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_MAX_URL_CHARS = 2048
_HOST_RE = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?")
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost")


def classify_content_type(content_type: Any) -> str:
    """QQ 附件的 content_type -> image / audio / video / file。无法识别的一律算 file（与 nonebot-adapter-qq 的分段规则一致）；
    voice* 归为 audio：语音不属于文件，这里宁可不下载。"""
    ct = content_type.strip().lower() if isinstance(content_type, str) else ""
    if ct.startswith("image"):
        return "image"
    if ct.startswith("audio") or ct.startswith("voice"):
        return "audio"
    if ct.startswith("video"):
        return "video"
    return "file"


def normalize_download_url(raw: Any) -> Optional[str]:
    """校验并规范化下载地址：统一成 https://、拒绝内网 / IP / 带 userinfo / 非默认端口。不合格返回 None。
    返回值与 None 都绝不写进日志。"""
    try:
        if not isinstance(raw, str):
            return None
        url = raw.strip()
        if not url or len(url) > _MAX_URL_CHARS:
            return None
        if any(ch.isspace() or ch == "\\" or unicodedata.category(ch).startswith("C") for ch in url):
            return None
        if url.startswith("//"):
            url = "https:" + url
        elif "://" in url:
            scheme, rest = url.split("://", 1)
            if scheme.lower() not in ("http", "https"):
                return None
            url = "https://" + rest
        else:
            url = "https://" + url
        parts = urlsplit(url)
        if parts.scheme != "https" or "@" in parts.netloc:
            return None
        port = parts.port
        if port is not None and port != 443:
            return None
        host = (parts.hostname or "").lower()
        if not host or "." not in host or ".." in host or not _HOST_RE.fullmatch(host):
            return None
        try:
            ipaddress.ip_address(host)
            return None
        except ValueError:
            pass
        last_label = host.rstrip(".").rsplit(".", 1)[-1]
        if last_label.isdigit() or host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
            return None
        return url
    except Exception:
        return None


def split_suffix(display_name: Any) -> tuple[str, str]:
    """把文件名拆成 (stem, suffix)：suffix 含点、小写，只认最后一个 .[A-Za-z0-9]{1,10} 片段；没有则 ""。"""
    name = display_name if isinstance(display_name, str) else ""
    idx = name.rfind(".")
    if idx > 0 and _SUFFIX_RE.fullmatch(name[idx:]):
        return name[:idx], name[idx:].lower()
    return name, ""


def sanitize_filename(raw: Any, *, max_stem: int = 60) -> str:
    """把 QQ 给的原始文件名清洗成可以安全展示、安全拼进路径的名字。永不返回空串，永不含路径分隔符或控制字符。"""
    try:
        name = unicodedata.normalize("NFC", raw) if isinstance(raw, str) else ""
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
        name = "".join(
            "_" if (ch in _FORBIDDEN_NAME_CHARS or unicodedata.category(ch).startswith("C")) else ch
            for ch in name
        )
        name = name.strip().rstrip(". ").lstrip(".").strip()
        stem, suffix = split_suffix(name)
        stem = stem.strip()[: max(1, int(max_stem))].rstrip(". ").lstrip(".")
        if not stem:
            stem = "file"
        if stem.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            stem = "_" + stem
        return stem + suffix
    except Exception:
        return "file"


def guess_suffix_from_bytes(head: Any) -> str:
    """只看前 16 字节的魔数猜后缀；不打开、不解压任何内容。认不出返回 ""。"""
    if not isinstance(head, (bytes, bytearray)):
        return ""
    head = bytes(head[:16])
    if head.startswith(b"%PDF-"):
        return ".pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if head.startswith(b"PK\x03\x04"):
        return ".zip"
    if head.startswith(b"\x1f\x8b"):
        return ".gz"
    return ""


def make_disk_name(display_name: str, sha8: str) -> str:
    """落盘 / 上传用的文件名：<stem>_<sha8><suffix>。stem 里除字母数字（含中日韩）、下划线、点、连字符之外的字符都换成 _，
    因为路径会被模型原样抄进工具参数和 shell 命令。"""
    stem, suffix = split_suffix(display_name)
    stem = re.sub(r"[^\w.\-]", "_", stem) or "file"
    return f"{stem}_{sha8}{suffix}"


def format_file_size(n: Any) -> str:
    """0 B / 512 B / 1.0 KB / 35.2 KB / 1.2 MB / 1.5 GB（1 KB = 1024 B）。负数 / 非数按 0。"""
    try:
        value = float(max(0, int(n)))
    except Exception:
        value = 0.0
    if value < 1024:
        return f"{int(value)} B"
    units = ("KB", "MB", "GB")
    for i, unit in enumerate(units):
        value /= 1024
        if round(value, 1) < 1024 or i == len(units) - 1:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} GB"


def validate_session_subdir(sid: Any) -> str:
    """会话 id 只能是 [A-Za-z0-9_-]{8,64}（uuid4().hex 天然满足），否则 ValueError；文案不含入参。"""
    if not isinstance(sid, str) or not _SESSION_SUBDIR_RE.fullmatch(sid):
        raise ValueError("会话目录名不合法")
    return sid


def receipt_name(display_name: str, *, safe: bool) -> str:
    """回执 / 清单里展示的文件名；safe 为真时把 "." 显示成 "[.]"，防止被 QQ 当域名拒收。"""
    return display_name.replace(".", "[.]") if safe else display_name


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def incoming_files_from_attachments(attachments: Any, segments: Any) -> list[IncomingFile]:
    """从 QQ 事件里提取待下载的文件。

    attachments 非空 -> 只用它（带文件名和大小）；为空 / None -> 回落到消息里的 file 段（只有 url）。
    按 url 去重保序；不在这里校验 URL 合法性，非法链接留到下载阶段报错，用户才能得到反馈。
    鸭子类型，不依赖 nonebot；任何异常都返回已经收集到的部分。
    """
    found: list[IncomingFile] = []
    seen: set[str] = set()
    try:
        had_attachments = False
        for att in (attachments or ()):          # 逐项迭代而不是先 list()：中途出错时仍能保留已收集到的部分
            had_attachments = True
            url = _field(att, "url")
            content_type = _field(att, "content_type")
            if not isinstance(url, str) or not url.strip():
                continue
            if classify_content_type(content_type) != "file" or url in seen:
                continue
            seen.add(url)
            filename = _field(att, "filename")
            try:
                size = max(0, int(_field(att, "size") or 0))
            except Exception:
                size = 0
            found.append(IncomingFile(
                url=url,
                filename=filename if isinstance(filename, str) else "",
                size=size,
                content_type=content_type if isinstance(content_type, str) else "",
            ))
        if had_attachments:
            return found
        for seg in (segments or ()):
            if _field(seg, "type") != "file":
                continue
            data = _field(seg, "data")
            url = data.get("url") if isinstance(data, dict) else None
            if not isinstance(url, str) or not url.strip() or url in seen:
                continue
            seen.add(url)
            found.append(IncomingFile(url=url))
    except Exception:
        pass
    return found
