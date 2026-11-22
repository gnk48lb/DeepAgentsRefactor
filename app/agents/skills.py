"""技能后端装配与路由管理 (所在层: L4)

deepagents Skills 的 backend 装配：把磁盘上的 skills/ 目录暴露成虚拟路径 /skills/。
故意不 import app.infra.models / app.agents.orchestrator，方便离线烟测（不需要任何 API Key）。
"""
import os

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import DeleteResult, EditResult, FileUploadResponse, WriteResult
from app import config

PROJECT_ROOT = config.PROJECT_ROOT
SKILLS_DIR = os.path.join(PROJECT_ROOT, "skills")   # 磁盘上的真实目录（owner 用）
SKILLS_MEMBER_DIR = os.path.join(PROJECT_ROOT, "skills_member")   # member 用的技能目录：与 owner 的物理分离，且只读
SKILLS_ROUTE = "/skills/"                           # 给 SkillsMiddleware / read_file 看的虚拟路径

_READONLY_ERROR = "错误：/skills/ 是只读目录，不能写入、修改或删除。"


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """只读的真实目录后端：写 / 改 / 删 / 上传一律不碰磁盘、直接返回带 error 的结果；读类方法保持父类行为。

    FilesystemBackend 本身没有任何只读保护，而 member 的 CodeAgent 有 write_file / edit_file：
    如果 /skills/ 路由到一个可写的真实目录，member 就能改写 owner 的 Agent 之后会读的技能文件。
    """

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=_READONLY_ERROR)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=_READONLY_ERROR)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error=_READONLY_ERROR)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error=_READONLY_ERROR)

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=_READONLY_ERROR)

    async def adelete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=_READONLY_ERROR)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error=_READONLY_ERROR) for path, _ in files]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error=_READONLY_ERROR) for path, _ in files]


def build_member_skills_backend() -> CompositeBackend:
    """member 主管用的技能后端：/skills/ 路由到只读的 skills_member/，其余仍是内存 state 虚拟文件系统。"""
    return CompositeBackend(
        default=StateBackend(),
        routes={SKILLS_ROUTE: ReadOnlyFilesystemBackend(root_dir=SKILLS_MEMBER_DIR, virtual_mode=True)},
    )


def build_skills_backend() -> CompositeBackend:
    """默认仍是内存 state 虚拟文件系统（与此前行为一致），仅把 /skills/ 路由到真实 skills/ 目录。"""
    os.makedirs(SKILLS_DIR, exist_ok=True)
    return CompositeBackend(
        default=StateBackend(),
        routes={SKILLS_ROUTE: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)},
    )
