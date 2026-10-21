"""
tests/smoke_architecture.py

架构契约与分层离线冒烟测试：
- A 导入可解析性与禁令 (无裸 import config, app/ 无相对导入, 无 from app.config import X, mcp_servers 无 app.* 导入)
- B 分层与单向依赖校验 (L0..L5 单向依赖, 同层互不 import, tools 互不 import, experts 互不 import, app/ 不得 import app.bootstrap)
- C __init__.py 卫生 (除 app/tools/amap/__init__.py 外均无 Import 节点)
- D 提示词单例与角色识别 (_identify_agent 对主 Agent、记忆提取器与各专家 Prompt 准确识别, WORKER_BASE_PROMPT 唯一赋值)
- E 专家规格表 (ExpertSpec 结构、属性名、工具顺序与 _DECLARATIVE_SPECS 顺序)
- F 轻量导入 (子进程中导入轻量模块时不触发 models / milvus_client / knowledge / memory)
- G 无 key 导入 (子进程中清除全部 API key 后可安全导入 app 全模块，访问模型抛 ValueError，BM25 缓存单例正常)
- H MCP 配置文件 (mcp_servers.json 中的脚本路径真实存在)
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]

LAYER_MAP = {
    "config": 0,
    "utils": 0,
    "runtime": 1,
    "infra": 1,
    "knowledge": 2,
    "memory": 2,
    "tools": 3,
    "agents": 4,
    "services": 5,
}


def _collect_py_files(dirs: List[str]) -> List[Path]:
    files: List[Path] = []
    for d in dirs:
        p = ROOT / d
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            for f in p.rglob("*.py"):
                files.append(f)
    return files


def test_a_imports_and_bans():
    """A 导入可解析性与语法禁令。"""
    targets = ["app", "plugins", "scripts", "tests", "bot.py", "main.py", "mcp_servers"]
    files = _collect_py_files(targets)

    # 预加载模块顶层名字缓存，用于快速判断 from M import N 是否存在
    top_level_names: dict[str, set[str]] = {}
    has_getattr: set[str] = set()

    for f in (ROOT / "app").rglob("*.py"):
        rel = f.relative_to(ROOT)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            mod_name = ".".join(parts[:-1])
        else:
            parts[-1] = parts[-1][:-3]
            mod_name = ".".join(parts)

        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), str(f))
        except Exception:
            continue

        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__":
                has_getattr.add(mod_name)

        top_level_names[mod_name] = names

    for f in files:
        rel_str = str(f.relative_to(ROOT)).replace("\\", "/")
        is_app = rel_str.startswith("app/")
        is_mcp = rel_str.startswith("mcp_servers/")
        content = f.read_text(encoding="utf-8")
        tree = ast.parse(content, str(f))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    assert name != "config", f"{rel_str}:{node.lineno} 禁止裸 import config"
                    if is_mcp:
                        assert not name.startswith("app"), f"{rel_str}:{node.lineno} mcp_servers 禁止导入 app.*"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not (is_app and node.level > 0), f"{rel_str}:{node.lineno} app/ 内部禁止相对导入"
                assert mod != "app.config", f"{rel_str}:{node.lineno} 禁止使用 from app.config import ..."
                if is_mcp:
                    assert not mod.startswith("app"), f"{rel_str}:{node.lineno} mcp_servers 禁止导入 app.*"

                if mod.startswith("app"):
                    # 检查模块是否存在
                    mod_path = ROOT / (mod.replace(".", "/") + ".py")
                    pkg_path = ROOT / mod.replace(".", "/") / "__init__.py"
                    assert mod_path.exists() or pkg_path.exists(), f"{rel_str}:{node.lineno} 模块 {mod} 不存在"

                    # 检查导入的名字是否存在
                    if mod not in has_getattr and mod in top_level_names:
                        for alias in node.names:
                            target_name = alias.name
                            if target_name == "*":
                                continue
                            sub_mod_path = ROOT / (mod.replace(".", "/") + f"/{target_name}.py")
                            sub_pkg_path = ROOT / mod.replace(".", "/") / target_name / "__init__.py"
                            is_submodule = sub_mod_path.exists() or sub_pkg_path.exists()
                            is_exported = target_name in top_level_names[mod]
                            assert is_submodule or is_exported, (
                                f"{rel_str}:{node.lineno} 无法从 {mod} 导入 {target_name}"
                            )


def test_b_layering_and_deps():
    """B 分层架构与单向依赖校验。"""
    for f in (ROOT / "app").rglob("*.py"):
        rel = f.relative_to(ROOT)
        rel_str = str(rel).replace("\\", "/")
        parts = rel_str.split("/")
        pkg = parts[1]
        if pkg.endswith(".py"):
            pkg = pkg[:-3]
        file_layer = LAYER_MAP.get(pkg)
        if file_layer is None:
            continue

        tree = ast.parse(f.read_text(encoding="utf-8"), str(f))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    targets.append(node.module)

            for t in targets:
                if not t.startswith("app"):
                    continue
                assert t != "app.bootstrap", f"{rel_str}:{node.lineno} app/ 内部禁止导入 app.bootstrap"
                if t in ("app", "app.config"):
                    continue
                t_parts = t.split(".")
                if len(t_parts) < 2:
                    continue
                t_pkg = t_parts[1]
                t_layer = LAYER_MAP.get(t_pkg)
                if t_layer is None:
                    continue

                # 向上依赖拦截
                assert t_layer <= file_layer, (
                    f"{rel_str}:{node.lineno} 分层违规：L{file_layer}({pkg}) 向上导入了 L{t_layer}({t})"
                )

                # 同层互不 import
                if t_layer == file_layer:
                    assert t_pkg == pkg, (
                        f"{rel_str}:{node.lineno} 同层越界：{pkg} 导入了同层平级子包 {t_pkg} ({t})"
                    )
                    # tools/*_tools.py 互不 import
                    if pkg == "tools" and f.stem.endswith("_tools"):
                        if len(t_parts) >= 3 and t_parts[2].endswith("_tools"):
                            assert t_parts[2] == f.stem, (
                                f"{rel_str}:{node.lineno} 工具同级违规：{f.name} 导入了平级工具模块 {t_parts[2]}"
                            )
                    # agents/experts/* 互不 import
                    if pkg == "agents" and "experts" in parts:
                        if len(t_parts) >= 4 and t_parts[2] == "experts":
                            assert t_parts[3] == f.stem, (
                                f"{rel_str}:{node.lineno} 专家同级违规：{f.name} 导入了平级专家模块 {t_parts[3]}"
                            )


def test_c_empty_inits():
    """C __init__.py 文件为空（除 docstring 外无 Import 节点）。"""
    for f in (ROOT / "app").rglob("__init__.py"):
        rel_str = str(f.relative_to(ROOT)).replace("\\", "/")
        if rel_str == "app/tools/amap/__init__.py":
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), str(f))
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not imports, f"{rel_str} 含有 Import 语句，子包 __init__.py 必须为空！"


def test_d_prompt_uniqueness_and_identity():
    """D 提示词单例与角色识别验证。"""
    # 1. 验证 WORKER_BASE_PROMPT 在全仓只有一处赋值
    assigns = []
    for d in ["app", "plugins", "scripts", "mcp_servers"]:
        for f in (ROOT / d).rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"), str(f))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "WORKER_BASE_PROMPT":
                            assigns.append(str(f.relative_to(ROOT)).replace("\\", "/"))
    assert assigns == ["app/agents/prompts.py"], f"WORKER_BASE_PROMPT 在全仓必须唯一定义: {assigns}"

    # 2. 验证 _identify_agent 准确性
    from langchain_core.messages import SystemMessage
    from app.runtime.cli_logger import _identify_agent
    from app.agents.prompts import MAIN_AGENT_PROMPT
    from app.memory.extractor import MEMORY_EXTRACTOR_PROMPT
    from app.agents.experts import (
        knowledge_agent,
        media_agent,
        map_agent,
        code_agent,
        sql_agent,
        browser_agent,
        desktop_agent,
        file_agent,
    )

    def ident(p: str) -> str:
        return _identify_agent([[SystemMessage(content=p)]], None, None)

    assert ident(MAIN_AGENT_PROMPT) == "MainAgent (Supervisor)"
    assert ident(MEMORY_EXTRACTOR_PROMPT) == "MemoryExtractor"
    assert ident(knowledge_agent.PROMPT) == "KnowledgeAgent"
    assert ident(media_agent.PROMPT) == "MediaAgent"
    assert ident(map_agent.PROMPT) == "MapAgent"
    assert ident(code_agent.PROMPT) == "CodeAgent"
    assert ident(sql_agent.PROMPT) == "SQLAgent"
    assert ident(browser_agent.PROMPT) == "BrowserAgent"
    assert ident(desktop_agent.PROMPT) == "DesktopAgent"
    assert ident(file_agent.PROMPT + file_agent.PROMPT_MARKITDOWN) == "FileAgent"


def test_e_expert_spec_table():
    """E 专家规格表契约验证。"""
    import json
    from app.agents.experts import (
        knowledge_agent,
        media_agent,
        map_agent,
        code_agent,
        sql_agent,
        browser_agent,
    )
    from app.agents import supervisor
    from app.agents.skills import SKILLS_ROUTE

    expected_specs = [
        (knowledge_agent.SPEC, "KnowledgeAgent", "vlm", ["rag", "web_search", "get_current_time"], ()),
        (media_agent.SPEC, "MediaAgent", "worker_llm", ["av_graph_rag"], ()),
        (code_agent.SPEC, "CodeAgent", "worker_llm", ["execute_python_code"], (SKILLS_ROUTE,)),
        (sql_agent.SPEC, "SQLAgent", "worker_llm", ["execute_sql"], ()),
        (
            browser_agent.SPEC,
            "BrowserAgent",
            "worker_llm",
            ["read_webpage_content", "get_bilibili_profile", "execute_complex_browser_action"],
            (),
        ),
    ]

    for spec, name, model_key, tool_names, skills in expected_specs:
        assert spec.name == name, f"Spec name mismatch: {spec.name} != {name}"
        assert spec.model_key == model_key, f"Spec model_key mismatch for {name}: {spec.model_key} != {model_key}"
        assert [t.name for t in spec.tools_factory()] == tool_names, (
            f"Tools mismatch for {name}: {[t.name for t in spec.tools_factory()]} != {tool_names}"
        )
        assert spec.skills == skills, f"Skills mismatch for {name}: {spec.skills} != {skills}"

    # MapAgent 校验：工具名顺序需与 tools_snapshot.json 一致
    assert map_agent.SPEC.name == "MapAgent"
    assert map_agent.SPEC.model_key == "worker_llm"
    assert map_agent.SPEC.skills == ()
    snap_data = json.loads((ROOT / "app/tools/amap/tools_snapshot.json").read_text(encoding="utf-8"))
    assert [t.name for t in map_agent.SPEC.tools_factory()] == [item["name"] for item in snap_data]

    # supervisor._DECLARATIVE_SPECS 顺序验证
    expected_order = ["KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent", "SQLAgent", "BrowserAgent"]
    actual_order = [s.name for s in supervisor._DECLARATIVE_SPECS]
    assert actual_order == expected_order, f"Declarative specs order mismatch: {actual_order} != {expected_order}"


def test_f_lightweight_imports_subprocess():
    """F 轻量模块导入隔离性（子进程运行）。"""
    code = """
import sys
import app.config
import app.utils.text
import app.utils.images
import app.runtime.hitl
import app.runtime.outbox
import app.agents.spec
import app.agents.prompts
import app.agents.skills
import app.agents.experts.file_agent
import app.tools.markitdown_tool
import app.tools.amap

forbidden = ["app.infra.models", "app.infra.milvus_client", "app.knowledge", "app.memory"]
loaded = [m for m in forbidden if any(name == m or name.startswith(m + ".") for name in sys.modules)]
assert not loaded, f"轻量导入泄漏了受限模块: {loaded}"
"""
    env = dict(os.environ)
    res = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"轻量导入测试失败:\n{res.stdout}\n{res.stderr}"


def test_g_empty_key_imports_subprocess():
    """G 无 API Key 环境安全导入与模型惰性化（子进程运行）。"""
    code = """
import os, sys, pkgutil, importlib

keys = [
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "SILICON_FLOW_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN", "AMAP_API_KEY", "TAVILY_API_KEY",
    "CLOUDFLARE_API_TOKEN"
]
for k in keys:
    os.environ[k] = ""

import app
for info in pkgutil.walk_packages(app.__path__, "app."):
    importlib.import_module(info.name)

from app.infra import models

try:
    _ = models.worker_llm
    raise AssertionError("models.worker_llm 应在缺少 key 时抛出 ValueError")
except ValueError:
    pass

try:
    models.validate_required_keys()
    raise AssertionError("models.validate_required_keys 应在缺少 key 时抛出 ValueError")
except ValueError:
    pass

assert models.bm25_ef is models.bm25_ef, "bm25_ef 必须为单例缓存"
assert models.bm25_ef is not models.av_bm25_ef, "bm25_ef 与 av_bm25_ef 不应为同一实例"
"""
    env = dict(os.environ)
    res = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"无 Key 导入测试失败:\n{res.stdout}\n{res.stderr}"


def test_h_mcp_config():
    """H mcp_servers.json 配置文件参数有效性。"""
    config_file = ROOT / "mcp_servers.json"
    assert config_file.exists(), "mcp_servers.json 配置文件不存在"
    data = json.loads(config_file.read_text(encoding="utf-8"))
    for server_name, server_cfg in data.items():
        args = server_cfg.get("args", [])
        for arg in args:
            if isinstance(arg, str) and arg.endswith(".py"):
                p = ROOT / arg
                assert p.exists(), f"MCP 服务器 {server_name} 依赖的脚本 {arg} 不存在！"


SCENARIOS = [
    ("A 导入可解析性与语法禁令", test_a_imports_and_bans),
    ("B 分层架构与单向依赖校验", test_b_layering_and_deps),
    ("C __init__.py 卫生", test_c_empty_inits),
    ("D 提示词单例与角色识别", test_d_prompt_uniqueness_and_identity),
    ("E 专家规格表契约验证", test_e_expert_spec_table),
    ("F 轻量模块导入隔离性", test_f_lightweight_imports_subprocess),
    ("G 无 key 导入与模型惰性化", test_g_empty_key_imports_subprocess),
    ("H MCP 配置文件校验", test_h_mcp_config),
]


def main():
    print("=" * 70)
    print("开始运行系统架构契约与分层离线冒烟测试...")
    print("=" * 70)
    passed = 0
    failed = 0

    for name, test_func in SCENARIOS:
        try:
            test_func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
