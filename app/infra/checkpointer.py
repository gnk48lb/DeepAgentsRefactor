"""LangGraph checkpoint 的 SQLite 持久化辅助 (所在层: L1)

checkpointer 用官方 AsyncSqliteSaver（不手写 BaseCheckpointSaver：HITL 恢复依赖 pending-writes 语义）。
本模块只做三件事：打开 / 关闭 saver，以及清理“子专家嵌套命名空间”的 checkpoint。

严格 msgpack：显式传入 allowed_msgpack_modules=None 的序列化器（等价于 LANGGRAPH_STRICT_MSGPACK=true，
但不依赖环境变量在 langgraph 被导入之前设置）。本项目的状态类型（消息、deepagents 的 files / todos、
Interrupt、Command）都在内置安全清单里或是纯 dict，重启恢复测试（smoke_chat_service H）覆盖了这一点。
"""

import os
from typing import Optional

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app import config

# Phase 0 对 langgraph-checkpoint-sqlite 3.1.x 源码核对过的表：两张表都带 (thread_id, checkpoint_ns)
_CHECKPOINT_TABLES = ("checkpoints", "writes")


async def open_checkpointer(path: Optional[str] = None) -> AsyncSqliteSaver:
    """打开 checkpoints.db。必须在运行中的事件循环里调用：saver 绑定创建它时的事件循环。"""
    path = path or config.CHECKPOINT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = await aiosqlite.connect(path)
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
        saver = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        await saver.setup()
    except Exception:
        await conn.close()
        raise
    return saver


async def close_checkpointer(saver: Optional[AsyncSqliteSaver]) -> None:
    """幂等关闭。不关 aiosqlite 连接会让进程退出时挂住。"""
    if saver is None:
        return
    conn = getattr(saver, "conn", None)
    if conn is None:
        return
    try:
        await conn.close()
    except ValueError:
        pass  # 已关闭
    except Exception as e:
        print(f"⚠️ 警告：checkpointer 关闭失败 (非致命): {e}")


async def prune_nested_checkpoints(saver: AsyncSqliteSaver, thread_id: str) -> int:
    """删除该 thread 下 checkpoint_ns != '' 的 checkpoint 与 writes 行，返回删除行数。

    为什么只动嵌套命名空间：DesktopAgent 截图的 base64 会让子专家 checkpoint 迅速膨胀，
    而一轮正常结束后这些嵌套状态不会再被读取。根命名空间保留全部历史：不要清理它，
    也不要调用 saver.aprune()（官方文档提醒，朴素 keep_latest 在用了 DeltaChannel 的图上会让状态静默重建为空）。
    任何异常只打印警告并返回 0，绝不影响对话。
    """
    try:
        deleted = 0
        async with saver.lock:
            conn = saver.conn
            for table in _CHECKPOINT_TABLES:
                cur = await conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ? AND checkpoint_ns != ''", (str(thread_id),)
                )
                deleted += max(cur.rowcount, 0)
                await cur.close()
            await conn.commit()
        return deleted
    except Exception as e:
        print(f"⚠️ 警告：嵌套 checkpoint 清理失败 (非致命): {e}")
        return 0
