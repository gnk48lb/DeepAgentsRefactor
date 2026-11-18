"""用户身份：平台 openid / CLI 名字 -> 内部 user_id (所在层: L0，只依赖标准库、app.config、app.utils.context)

内部 user_id 只有三类：
  - "owner"：命中 OWNER_USER_IDS 的所有平台 openid，以及本地 CLI 的默认身份（共享同一套会话与记忆）
  - 平台 openid 本身，格式 [A-Za-z0-9]{8,64}
  - "cli-<name>"：CLI 里 --member 的测试身份，name 为 [A-Za-z0-9_]{1,32}
平台来源的 id 产生不了 "owner"（长度不足 8）和 "cli-" 前缀（正则不含连字符），所以无法冒充。

注意一律用 fullmatch：re.match 配 `$` 会放过末尾换行。
"""

import re
from dataclasses import dataclass

from app import config

OWNER_USER_ID = "owner"

_PLATFORM_ID_RE = re.compile(r"[A-Za-z0-9]{8,64}")
_CLI_NAME_RE = re.compile(r"[A-Za-z0-9_]{1,32}")
_CLI_PREFIX = "cli-"


@dataclass(frozen=True)
class Identity:
    user_id: str
    is_owner: bool
    raw_id: str          # 平台原始 ID，仅用于显示


def owner_identity(raw_id: str = "cli") -> Identity:
    return Identity(user_id=OWNER_USER_ID, is_owner=True, raw_id=raw_id)


def cli_member_identity(name: str) -> Identity:
    """本地 CLI 的 member 测试身份（不开 QQ 就能体验 member）。"""
    if not isinstance(name, str) or not _CLI_NAME_RE.fullmatch(name):
        raise ValueError("--member 的名字只能由字母、数字、下划线组成，长度 1~32")
    return Identity(user_id=_CLI_PREFIX + name, is_owner=False, raw_id="cli")


def resolve_platform_user(raw_id: str) -> Identity:
    """QQ 官方机器人的 user_openid -> Identity。格式不合法抛 ValueError。"""
    if not isinstance(raw_id, str) or not _PLATFORM_ID_RE.fullmatch(raw_id):
        raise ValueError("平台用户 ID 格式不合法")
    if raw_id in config.OWNER_USER_IDS:        # 每次现读，不在 import 时拷贝
        return Identity(user_id=OWNER_USER_ID, is_owner=True, raw_id=raw_id)
    return Identity(user_id=raw_id, is_owner=False, raw_id=raw_id)


def validate_user_id(user_id: str) -> str:
    """只接受 owner、平台 openid、cli-... 三类，其余抛 ValueError；返回原值，方便就地使用。

    所有把 user_id 拼进过滤表达式、集合名、键名的地方都必须先过它。
    """
    if isinstance(user_id, str):
        if user_id == OWNER_USER_ID or _PLATFORM_ID_RE.fullmatch(user_id):
            return user_id
        if user_id.startswith(_CLI_PREFIX) and _CLI_NAME_RE.fullmatch(user_id[len(_CLI_PREFIX):]):
            return user_id
    raise ValueError(f"非法的 user_id: {user_id!r:.80}")


def invalid_owner_ids() -> list[str]:
    """OWNER_USER_IDS 里格式不合法的项（它们永远不会命中任何平台用户，多半是 .env 写错了）。"""
    return sorted(i for i in config.OWNER_USER_IDS if not _PLATFORM_ID_RE.fullmatch(i))
