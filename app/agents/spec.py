"""专家智能体规范数据结构定义 (所在层: L4)"""

from dataclasses import dataclass
from typing import Callable, Sequence

from app.agents.budget import build_budget_middleware, tool_limit_for


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    description: str
    system_prompt: str
    tools_factory: Callable[[], Sequence]   # 延迟取工具
    model_key: str                          # app.infra.models 上的属性名
    skills: tuple[str, ...] = ()

    def to_subagent(self, models) -> dict:  # 产出 deepagents 的 SubAgent dict
        d = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools_factory()),
            "model": getattr(models, self.model_key),
            "middleware": build_budget_middleware(tool_limit_for(self.name)),
        }
        if self.skills:
            d["skills"] = list(self.skills)
        return d
