"""手动管理 Modal 云端沙箱（member 的 CodeAgent）

没有启动时孤儿清理：同一台机器上可能同时跑着另一个进程（比如你在另一个终端开了 CLI），启动时杀“看起来没人管”的沙箱
会误杀它的沙箱；Modal 会按 idle_timeout / timeout 自己回收。进程被强杀（没走正常关闭流程）时，云端沙箱最多存活
MODAL_MAX_LIFETIME_SECONDS，可以用本脚本提前清理。只会碰带 svc=gnk48 标签的沙箱，且只显示对象 id 与哈希后的标签。

用法：
    uv run python scripts/manage_modal_sandboxes.py list
    uv run python scripts/manage_modal_sandboxes.py kill --all [--yes]
    uv run python scripts/manage_modal_sandboxes.py kill --user <用户哈希前缀> [--yes]

退出码：0 成功；1 失败或已取消；2 缺少令牌或没装 modal。令牌只在 .env 里，本脚本不读取、不打印令牌。
"""

from app import bootstrap
bootstrap.apply()

import argparse
import sys
from typing import Any

from app import config
from app.infra import modal_sandbox


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理 Modal 云端沙箱（只处理带 svc=gnk48 标签的）")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="列出沙箱：对象 id 与哈希后的用户 / 会话标签")
    kill = sub.add_parser("kill", help="终止沙箱（默认要交互确认）")
    target = kill.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="终止全部带 svc=gnk48 标签的沙箱")
    target.add_argument("--user", default="", help="只终止这个用户的沙箱：list 里显示的用户哈希（u）前缀")
    kill.add_argument("--yes", action="store_true", help="跳过交互确认")
    return parser


def _describe(sb: Any) -> dict:
    try:
        tags = dict(sb.get_tags())
    except Exception:
        tags = {}
    return {"id": str(sb.object_id), "u": tags.get("u", "?"), "s": tags.get("s", "?"), "role": tags.get("role", "?")}


def _list_sandboxes(modal: Any) -> list:
    """返回 [(沙箱对象, 描述)]；App 还不存在就当作没有沙箱。"""
    try:
        app = modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=False)
    except Exception as e:
        cls = getattr(getattr(modal, "exception", None), "NotFoundError", None)
        if cls is not None and isinstance(e, cls):
            return []
        raise
    return [(sb, _describe(sb)) for sb in modal.Sandbox.list(app_id=app.app_id, tags={"svc": "gnk48"})]


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if not config.MODAL_TOKENS_PRESENT:
        print("❌ 没有配置 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET：请把令牌写进 .env 后重试。")
        return 2
    try:
        modal = modal_sandbox.load_modal()
    except Exception as e:
        print(f"❌ 无法导入 modal 包（请先 uv sync）: {modal_sandbox.redact(f'{type(e).__name__}: {e}')[:200]}")
        return 2

    try:
        found = _list_sandboxes(modal)
    except Exception as e:
        print(f"❌ 无法列出沙箱: {modal_sandbox.explain(e)}")
        return 1

    if args.command == "list":
        if not found:
            print("没有带 svc=gnk48 标签的沙箱。")
            return 0
        print(f"共 {len(found)} 个沙箱：")
        for _, d in found:
            print(f"  {d['id']}  用户 u={d['u']}  会话 s={d['s']}  角色 {d['role']}")
        return 0

    if args.all:
        selected = found
        scope = "全部"
    else:
        prefix = args.user.strip()
        if not prefix:
            print("❌ --user 不能为空：请给 list 里显示的用户哈希（u）前缀。")
            return 1
        selected = [(sb, d) for sb, d in found if d["u"].startswith(prefix)]
        scope = f"用户哈希以 {prefix} 开头的"
    if not selected:
        print(f"没有匹配的沙箱（{scope}），什么都没有终止。")
        return 0

    print(f"即将终止{scope} {len(selected)} 个沙箱（里面的 /work 文件会丢失）：")
    for _, d in selected:
        print(f"  {d['id']}  u={d['u']}  s={d['s']}")
    if not args.yes:
        try:
            answer = input("确认终止？输入 yes 继续: ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "yes":
            print("已取消。")
            return 1

    killed = failed = 0
    for sb, d in selected:
        try:
            sb.terminate(wait=False)
            killed += 1
        except Exception as e:
            failed += 1
            print(f"⚠️ 终止 {d['id']} 失败: {modal_sandbox.explain(e, limit=200)}")
    print(f"已终止 {killed} 个" + (f"，{failed} 个失败" if failed else "") + "。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
