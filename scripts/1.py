import os
import numpy as np
from pymilvus import MilvusClient

DB_FILE = "windows_milvus_test.db"
COLLECTION_NAME = "hello_milvus"

if os.path.exists(DB_FILE):
    import shutil
    shutil.rmtree(DB_FILE, ignore_errors=True)

print("--- [1] 初始化 Milvus Lite ---")
client = MilvusClient(uri=DB_FILE)

try:
    print("--- [2] 一键创建集合（全自动索引） ---")
    # 这种快捷创建方式会自动创建 ID、Vector，并在底层自动处理好 Windows 的索引文件锁
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=1024,
        metric_type="IP"
    )
    
    print("--- [3] 插入测试数据 ---")
    dense_vectors = np.random.randn(2, 1024).tolist()
    data = [
        {"id": 1, "vector": dense_vectors[0], "text": "第一条测试"},
        {"id": 2, "vector": dense_vectors[1], "text": "第二条测试"}
    ]
    client.insert(collection_name=COLLECTION_NAME, data=data)
    
    print("--- [4] 执行检索 ---")
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_vectors[0]],
        limit=1,
        output_fields=["text"]
    )
    print("🔍 结果:", res[0][0]['entity']['text'])
    print("🎉 Windows 原生完美跑通！")

except Exception as e:
    print(f"❌ 失败: {e}")
finally:
    client.close()