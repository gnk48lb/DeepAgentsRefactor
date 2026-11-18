"""长期记忆管理运维脚本

支持查看、软删除、硬删除与恢复长期记忆，统一通过 app.memory.store 接口进行操作。
每个用户一个独立的记忆集合：用 --user 选择操作谁的记忆（不写就是 owner 自己的）。

用法：
    uv run python scripts/manage_memory.py [--user owner|<user_id或唯一前缀>] list [--all]
    uv run python scripts/manage_memory.py [--user ...] delete <id...> [--hard]
    uv run python scripts/manage_memory.py [--user ...] restore <id...>
"""

from app import bootstrap
bootstrap.apply()

import argparse
import sys
import time
from app.infra import session_store
from app.memory import store
from app.utils.identity import OWNER_USER_ID, validate_user_id


def format_time(ts: int) -> str:
    """格式化 unix 秒时间戳。"""
    if not ts or ts <= 0:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)


def resolve_user_id(ref) -> str:
    """--user 的取值：不写 / owner -> owner；其余先按 users 表的 user_id 完整值或唯一前缀解析，
    解析不到再把它当作完整的 user_id（要求格式合法）。找不到或有歧义时直接退出。"""
    if not ref or ref.strip().lower() == "owner":
        return OWNER_USER_ID
    ref = ref.strip()
    try:
        info = session_store.get_store().resolve_user(ref)
    except ValueError as e:
        raise SystemExit(f"❌ {e}")
    finally:
        session_store.close_store()
    if info is not None:
        return info.user_id
    try:
        return validate_user_id(ref)
    except ValueError:
        raise SystemExit(f"❌ 没有找到用户: {ref}")


def list_memories(user_id: str, include_all: bool = False):
    records = store.list_memories(include_inactive=include_all, user_id=user_id)
    coll = store.collection_name(user_id)
    status_label = "全量（含 superseded / deleted）" if include_all else "仅有效（active）"
    print(f"=== 长期记忆清单 (用户: {user_id[:8] if user_id != OWNER_USER_ID else OWNER_USER_ID}, 集合: {coll}, 范围: {status_label}, 共 {len(records)} 条) ===")
    if not records:
        print("未找到符合条件的记忆记录。")
        return

    for item in records:
        rec_id = item.get("id")
        status = item.get("status", "unknown")
        kind = item.get("kind", "fact")
        created = format_time(item.get("created_at", 0))
        updated = format_time(item.get("updated_at", 0))
        supersedes = item.get("supersedes", -1)
        text = item.get("text", "")
        sup_str = str(supersedes) if supersedes != -1 else "-"
        print(
            f"ID: {rec_id} | Status: {status} | Kind: {kind} | "
            f"Created: {created} | Updated: {updated} | Supersedes: {sup_str}\n"
            f"  Text: {text}"
        )


def delete_memories(user_id: str, ids: list[int], hard: bool = False):
    if not ids:
        print("未指定要删除的 ID。")
        return

    if hard:
        purged = store.purge_memories(ids, user_id=user_id)
        print(f"⚠️ [物理清除] 成功物理删除 {purged} 条记录: {ids}")
    else:
        success_count = 0
        for mid in ids:
            if store.retire_memory(mid, evidence="scripts.manage_memory 显式删除", user_id=user_id):
                success_count += 1
        print(f"✅ [软删除完成] 成功标记删除 {success_count}/{len(ids)} 条记录: {ids}")


def restore_memories(user_id: str, ids: list[int]):
    if not ids:
        print("未指定要恢复的 ID。")
        return

    success_count = 0
    for mid in ids:
        if store.restore_memory(mid, user_id=user_id):
            success_count += 1
    print(f"✅ [恢复完成] 成功恢复 {success_count}/{len(ids)} 条记录为 active: {ids}")


def main():
    parser = argparse.ArgumentParser(description="管理 Milvus 长期记忆集合 (Schema v2，每个用户一个集合)")
    parser.add_argument(
        "--user",
        default=None,
        help="操作哪个用户的记忆：owner（不写就是它）、user_id 的完整值或唯一前缀",
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="查看记忆记录")
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="包含已废弃 (superseded) 和已删除 (deleted) 的全量记忆",
    )

    # delete
    del_parser = subparsers.add_parser("delete", help="删除记忆记录")
    del_parser.add_argument("ids", nargs="+", type=int, help="要删除的记录 ID 列表")
    del_parser.add_argument(
        "--hard",
        action="store_true",
        help="物理清除（彻底抹除数据，不可恢复；默认仅软删除置为 deleted）",
    )

    # restore
    res_parser = subparsers.add_parser("restore", help="恢复已删除的记录为 active")
    res_parser.add_argument("ids", nargs="+", type=int, help="要恢复的记录 ID 列表")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    user_id = resolve_user_id(args.user)
    if args.command == "list":
        list_memories(user_id, include_all=args.all)
    elif args.command == "delete":
        delete_memories(user_id, args.ids, hard=args.hard)
    elif args.command == "restore":
        restore_memories(user_id, args.ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
