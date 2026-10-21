"""技能后端装配与路由管理 (所在层: L4)

deepagents Skills 的 backend 装配：把磁盘上的 skills/ 目录暴露成虚拟路径 /skills/。
故意不 import app.models / app.deep_agent，方便离线烟测（不需要任何 API Key）。
"""
import os

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from app import config

PROJECT_ROOT = config.PROJECT_ROOT
SKILLS_DIR = os.path.join(PROJECT_ROOT, "skills")   # 磁盘上的真实目录
SKILLS_ROUTE = "/skills/"                           # 给 SkillsMiddleware / read_file 看的虚拟路径


def build_skills_backend() -> CompositeBackend:
    """默认仍是内存 state 虚拟文件系统（与此前行为一致），仅把 /skills/ 路由到真实 skills/ 目录。"""
    os.makedirs(SKILLS_DIR, exist_ok=True)
    return CompositeBackend(
        default=StateBackend(),
        routes={SKILLS_ROUTE: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
    )
