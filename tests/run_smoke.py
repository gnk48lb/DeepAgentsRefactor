"""一键运行全部离线冒烟测试（不含需要子进程/网络的 e2e 与 live 测试）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKES = [
    "tests/smoke_architecture.py",
    "tests/smoke_file_agent.py",
    "tests/smoke_budget.py",
    "tests/smoke_skills.py",
    "tests/smoke_amap_contract.py",
    "tests/smoke_amap_http.py",
    "tests/smoke_chat_service.py",
    "tests/smoke_knowledge.py",
    "tests/smoke_memory.py",
    "tests/smoke_orchestrator.py",
    "tests/smoke_units.py",
]

failed = []
for rel in SMOKES:
    print(f"\n=== {rel} ===", flush=True)
    if subprocess.call([sys.executable, str(ROOT / rel)], cwd=ROOT) != 0:
        failed.append(rel)
print("\n" + ("全部通过" if not failed else f"失败: {failed}"))
sys.exit(1 if failed else 0)
