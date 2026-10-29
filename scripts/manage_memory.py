"""长期记忆管理运维脚本

支持查看、软删除、硬删除与恢复长期记忆，统一通过 app.memory.store 接口进行操作。
"""

from app import bootstrap
bootstrap.apply()

import argparse
import time
from app.memory import store


def format_time(ts: int) -> str:
    """格式化 unix 秒时间戳。"""
    if not ts or ts <= 0:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)


def list_memories(include_all: bool = False):
    records = store.list_memories(include_inactive=include_all)
    coll = store.collection_name()
    status_label = "全量（含 superseded / deleted）" if include_all else "仅有效（active）"
    print(f"=== 长期记忆清单 (集合: {coll}, 范围: {status_label}, 共 {len(records)} 条) ===")
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


def delete_memories(ids: list[int], hard: bool = False):
    if not ids:
        print("未指定要删除的 ID。")
        return

    if hard:
        purged = store.purge_memories(ids)
        print(f"⚠️ [物理清除] 成功物理删除 {purged} 条记录: {ids}")
    else:
        success_count = 0
        for mid in ids:
            if store.retire_memory(mid, evidence="scripts.manage_memory 显式删除"):
                success_count += 1
        print(f"✅ [软删除完成] 成功标记删除 {success_count}/{len(ids)} 条记录: {ids}")


def restore_memories(ids: list[int]):
    if not ids:
        print("未指定要恢复的 ID。")
        return

    success_count = 0
    for mid in ids:
        if store.restore_memory(mid):
            success_count += 1
    print(f"✅ [恢复完成] 成功恢复 {success_count}/{len(ids)} 条记录为 active: {ids}")


def main():
    parser = argparse.ArgumentParser(description="管理 Milvus 长期记忆集合 (Schema v2)")
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

    if args.command == "list":
        list_memories(include_all=args.all)
    elif args.command == "delete":
        delete_memories(args.ids, hard=args.hard)
    elif args.command == "restore":
        restore_memories(args.ids)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
