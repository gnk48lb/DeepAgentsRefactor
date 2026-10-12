import sys, os
sys.path.insert(0, os.path.abspath("."))
from app import database
from app.tools import rag

# 初始化数据库并加载集合
database.run_global_database_init()

print("=== 1. DIRECT RAG CALL: 火神战姬 Q技能 ===")
res_rag = rag.invoke({"query": "火神战姬 Q技能"})
print("Type of rag result:", type(res_rag))
if isinstance(res_rag, list):
    for i, part in enumerate(res_rag):
        if isinstance(part, dict) and part.get("type") == "text":
            print(f"--- Part {i} [text] ---")
            print(part.get("text"))
        elif isinstance(part, dict) and part.get("type") == "image_url":
            b64_url = part.get("image_url", {}).get("url", "")
            print(f"--- Part {i} [image_url] --- (base64 length: {len(b64_url)})")
else:
    print("Raw rag result:\n", res_rag)
