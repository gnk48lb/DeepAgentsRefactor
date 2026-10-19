import json
import re
from typing import Optional, Tuple
import httpx
import config

_BASE = "https://restapi.amap.com"
_client: Optional[httpx.AsyncClient] = None
_KEY_RE = re.compile(r"(key=)[^&\s'\"`]+")


def redact(text: str) -> str:
    redacted = _KEY_RE.sub(r"\1***", text)
    key = getattr(config, "AMAP_API_KEY", None)
    if key and key in redacted:
        redacted = redacted.replace(key, "***")
    return redacted


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=_BASE,
            timeout=30.0,
            follow_redirects=True,
        )
    return _client


def set_client(client: Optional[httpx.AsyncClient]) -> None:
    global _client
    _client = client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def validate_config() -> None:
    key = getattr(config, "AMAP_API_KEY", None)
    if not key or str(key).startswith("${"):
        raise ValueError("AMAP_API_KEY 未配置或配置无效，请在 .env 中设置 AMAP_API_KEY。")


async def get(path: str, params: dict) -> Tuple[Optional[dict], Optional[str]]:
    validate_config()
    key = config.AMAP_API_KEY
    try:
        resp = await _get_client().get(path, params={**params, "key": key})
        resp.raise_for_status()
        return resp.json(), None
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return None, f"Request failed: {redact(str(e))}"
