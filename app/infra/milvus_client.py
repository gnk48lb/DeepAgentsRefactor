"""Milvus 客户端单例管理 (所在层: L1)"""

import os
from pymilvus import MilvusClient
from app import config

client = None


def get_client() -> MilvusClient:
    global client
    if client is None:
        # Ensure the parent directory of config.MILVUS_DB_PATH exists before initialization
        if not config.MILVUS_DB_PATH.startswith("http://") and not config.MILVUS_DB_PATH.startswith("https://"):
            db_dir = os.path.dirname(config.MILVUS_DB_PATH)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        # Use local Milvus DB Path from config, and disable keep_alive to prevent gRPC too_many_pings on Windows
        client = MilvusClient(uri=config.MILVUS_DB_PATH, keep_alive=False)
    return client
