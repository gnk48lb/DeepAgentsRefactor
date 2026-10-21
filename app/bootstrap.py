"""系统启动自适应补丁与环境初始化模块 (L0)"""

import logging
import os
import sys

_applied = False


def apply():
    """执行环境自适应配置与补丁（幂等调用）"""
    global _applied
    if _applied:
        return
    _applied = True

    # 1. UTF-8 输出重配置保护
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 2. 压低底层 C/C++ 库与向量引擎的日志噪音
    logging.getLogger("milvus_lite").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)

    # 3. Windows 平台下的 os.rename WinError 183 兼容补丁
    if sys.platform == "win32":
        if not getattr(os.rename, "_is_safe_rename", False):
            _orig_rename = os.rename

            def _safe_rename(src, dst):
                try:
                    _orig_rename(src, dst)
                except (FileExistsError, OSError) as e:
                    try:
                        os.replace(src, dst)
                    except Exception:
                        raise e

            _safe_rename._is_safe_rename = True
            os.rename = _safe_rename
