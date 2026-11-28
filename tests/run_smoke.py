"""一键运行全部离线冒烟测试（不含需要子进程/网络的 e2e 与 live 测试）。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 子进程环境统一压成 desktop：本机 .env 里可能是 DEPLOY_MODE=cloud，load_dotenv 不覆盖已有的环境变量。
# 想测 cloud 的断言都在 tests/smoke_cloud_mode.py 里，它自己临时切换 config.DEPLOY_MODE。
ENV = {**os.environ, "DEPLOY_MODE": "desktop"}
SMOKES = [
    "tests/smoke_architecture.py",
    "tests/smoke_file_agent.py",
    "tests/smoke_budget.py",
    "tests/smoke_skills.py",
    "tests/smoke_amap_contract.py",
    "tests/smoke_amap_http.py",
    "tests/smoke_chat_service.py",
    "tests/smoke_session.py",
    "tests/smoke_multiuser.py",
    "tests/smoke_qq_files.py",
    "tests/smoke_scheduler.py",
    "tests/smoke_scheduled_agent.py",
    "tests/smoke_knowledge.py",
    "tests/smoke_memory.py",
    "tests/smoke_orchestrator.py",
    "tests/smoke_cloud_mode.py",
    "tests/smoke_models.py",
    "tests/smoke_units.py",
    "tests/smoke_sandbox.py",
    "tests/smoke_modal.py",
    "tests/smoke_host_shell.py",
]

failed = []
for rel in SMOKES:
    print(f"\n=== {rel} ===", flush=True)
    if subprocess.call([sys.executable, str(ROOT / rel)], cwd=ROOT, env=ENV) != 0:
        failed.append(rel)
print("\n" + ("全部通过" if not failed else f"失败: {failed}"))
sys.exit(1 if failed else 0)
