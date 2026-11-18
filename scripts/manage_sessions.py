"""会话管理运维脚本

查看 / 删除会话。彻底删除（事件日志 + 挂起态 + checkpoint 线程）只走这里；
聊天里的 /reset 只清上下文，不删事件日志。
运维脚本不限用户：list 默认列出所有用户的会话（--user 过滤），show / delete 按 id 前缀跨用户解析。

用法：
    uv run python scripts/manage_sessions.py list [--user owner|<user_id前缀>]
    uv run python scripts/manage_sessions.py show <id前缀> [--from N --limit N --types a,b]
    uv run python scripts/manage_sessions.py delete <id前缀> [--yes]
"""

from app import bootstrap
bootstrap.apply()

import argparse
import sys
import time

from langgraph.checkpoint.sqlite import SqliteSaver

from app import config
from app.infra import session_store


def format_time(ts: float) -> str:
    if not ts or ts <= 0:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)


def _resolve(store, ref: str):
    """运维脚本跨用户解析会话 id / 前缀（聊天链路里的解析永远限定在当前用户自己的会话内）。"""
    try:
        info = store.resolve_session_any(ref)
    except ValueError as e:
        print(f"❌ {e}")
        return None
    if info is None:
        print(f"❌ 没有找到会话: {ref}")
    return info


def _resolve_user(store, ref: str):
    """--user 的取值：owner、完整 user_id 或唯一前缀。返回 user_id；找不到返回 None。"""
    if ref.strip().lower() == "owner":
        return "owner"
    try:
        user = store.resolve_user(ref)
    except ValueError as e:
        print(f"❌ {e}")
        return None
    if user is None:
        print(f"❌ 没有找到用户: {ref}")
        return None
    return user.user_id


def _user_label(store, user_id: str) -> str:
    """显示用：owner 原样；其他用户给 id 前 8 位 + 备注。"""
    if user_id == "owner":
        return "owner"
    user = store.get_user(user_id)
    label = f"（{user.label}）" if user is not None and user.label else ""
    return f"{user_id[:8]}{label}"


def list_sessions(store, limit: int, user_ref: str = ""):
    user_id = None
    if user_ref:
        user_id = _resolve_user(store, user_ref)
        if user_id is None:
            return 1
    infos = store.list_sessions_any(limit=limit, user_id=user_id)
    total = store.count_sessions(user_id=user_id)
    # 每个出现的用户各取一次活动会话（这些用户都至少有一个会话，不会触发新建）
    active = {uid: store.get_active_session_id(user_id=uid) for uid in {i.user_id for i in infos}}
    scope = f"，用户 {_user_label(store, user_id)}" if user_id else "，全部用户"
    print(f"=== 会话列表（库: {store.path}{scope}，共 {total} 个，显示最近 {len(infos)} 个）===")
    for info in infos:
        mark = " [活动]" if info.id == active.get(info.user_id) else ""
        print(
            f"{info.id}{mark}\n"
            f"  用户: {_user_label(store, info.user_id)} | 标题: {info.title or '（无标题）'} | 轮数: {info.turns} | 事件: {info.events} | "
            f"最后活跃: {format_time(info.updated_at)} | 创建: {format_time(info.created_at)}"
        )
    return 0


def show_session(store, ref: str, start_seq: int, limit: int, types: str):
    info = _resolve(store, ref)
    if info is None:
        return 1
    type_list = [t.strip() for t in types.split(",") if t.strip()] if types else None
    events = store.get_events(info.id, start_seq=start_seq, limit=limit, types=type_list)
    print(f"=== 会话 {info.id}「{info.title or '（无标题）'}」共 {info.events} 条事件，处理游标 {info.processed_seq} ===")
    ts = store.load_turn_state(info.id)
    if ts is not None:
        print(f"[挂起态] 原话: {ts.user_text!r} | 暂存图片: {len(ts.outbox)} 张")
    for ev in events:
        body = " ".join(str(v) for v in ev.payload.values() if isinstance(v, (str, int, float)))
        print(f"#{ev.seq} {format_time(ev.ts)} {ev.type}: {body[:300]}")
    if not events:
        print("（这个范围内没有事件）")
    return 0


def delete_session(store, ref: str, yes: bool):
    info = _resolve(store, ref)
    if info is None:
        return 1
    print(f"即将彻底删除会话 {info.id}「{info.title or '（无标题）'}」：{info.events} 条事件 + 挂起态 + checkpoint 线程，不可恢复。")
    if not yes:
        try:
            answer = input("确认删除？输入 yes 继续: ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "yes":
            print("已取消。")
            return 1
    was_active = info.id == store.get_active_session_id(user_id=info.user_id)
    store.delete_session(info.id)
    # checkpoint 线程：用官方同步 SqliteSaver 在 checkpoints.db 上删除（自己开连接，用完关闭）
    try:
        with SqliteSaver.from_conn_string(config.CHECKPOINT_DB_PATH) as saver:
            saver.delete_thread(info.id)
    except Exception as e:
        print(f"⚠️ 事件与会话已删除，但清理 checkpoint 线程失败（可稍后重试）: {e}")
        return 1
    print(f"✅ 已删除会话 {info.id[:8]}。" + ("它是活动会话，下一条消息会自动建新的默认会话。" if was_active else ""))
    return 0


def main():
    parser = argparse.ArgumentParser(description="管理会话与事件日志（sessions.db / checkpoints.db）")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="列出会话")
    p_list.add_argument("--limit", type=int, default=50, help="最多显示多少个（默认 50）")
    p_list.add_argument("--user", default="", help="只看某个用户：owner、user_id 或唯一前缀（默认全部用户）")

    p_show = sub.add_parser("show", help="查看某个会话的事件")
    p_show.add_argument("id", help="会话 id 或唯一前缀")
    p_show.add_argument("--from", dest="start_seq", type=int, default=1, help="起始事件序号（默认 1）")
    p_show.add_argument("--limit", type=int, default=50, help="最多显示多少条（默认 50）")
    p_show.add_argument("--types", default="", help="事件类型过滤，逗号分隔")

    p_del = sub.add_parser("delete", help="彻底删除会话（事件日志 + 挂起态 + checkpoint 线程）")
    p_del.add_argument("id", help="会话 id 或唯一前缀")
    p_del.add_argument("--yes", action="store_true", help="跳过交互确认")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    store = session_store.get_store()
    try:
        if args.command == "list":
            return list_sessions(store, args.limit, args.user)
        if args.command == "show":
            return show_session(store, args.id, args.start_seq, args.limit, args.types)
        return delete_session(store, args.id, args.yes)
    finally:
        session_store.close_store()


if __name__ == "__main__":
    sys.exit(main())
