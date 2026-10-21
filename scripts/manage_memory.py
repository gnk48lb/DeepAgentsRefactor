import sys
import argparse
from pathlib import Path

# Add project root to sys.path


# 设置标准输出编码为 utf-8



from app import config
from app import database

def list_memories():
    c = database.get_client()
    coll_name = config.MEMORY_COLLECTION_NAME
    if not c.has_collection(coll_name):
        print(f"集合 '{coll_name}' 不存在或尚未初始化。")
        return

    try:
        c.load_collection(collection_name=coll_name)
    except Exception:
        pass

    try:
        # 使用通用查询获取所有记录
        results = c.query(
            collection_name=coll_name,
            filter="id >= 0",
            output_fields=["id", "text"]
        )
        if not results:
            print("记忆集合为空，无任何记录。")
            return

        print(f"=== 记忆清单 (集合: {coll_name}, 共 {len(results)} 条) ===")
        for item in results:
            rec_id = item.get("id")
            rec_text = item.get("text", "")
            print(f"ID: {rec_id} | Text: {rec_text}")
    except Exception as e:
        print(f"❌ 查询记忆失败: {e}")

def delete_memories(ids):
    if not ids:
        print("未指定要删除的 ID。")
        return
    c = database.get_client()
    coll_name = config.MEMORY_COLLECTION_NAME
    if not c.has_collection(coll_name):
        print(f"集合 '{coll_name}' 不存在。")
        return

    try:
        int_ids = [int(i) for i in ids]
        c.delete(collection_name=coll_name, pks=int_ids)
        try:
            c.flush(coll_name)
        except Exception:
            pass
        print(f"✅ 已删除 ID 列表: {int_ids}")
    except Exception as e:
        print(f"❌ 删除记忆失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="管理 Milvus 记忆集合")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="打印记忆集合里所有记录的 id 和 text")

    del_parser = subparsers.add_parser("delete", help="按 id 删除记录")
    del_parser.add_argument("ids", nargs="+", help="要删除的记录 ID 列表")

    args = parser.parse_args()

    if args.command == "list":
        list_memories()
    elif args.command == "delete":
        delete_memories(args.ids)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
