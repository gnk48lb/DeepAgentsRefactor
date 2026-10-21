"""
tests/smoke_skills.py

Skills 离线冒烟测试：验证 skills/*/SKILL.md 规范与 app/agents/skills.py backend 装配。
不依赖外部服务、不需要任何 API Key。
"""
import glob
import os
import re
import sys
from pathlib import Path
import yaml

# 保证标准输出统一 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent



# 仅导入 app.agents.skills，严禁导入 app.infra.models / app.agents.orchestrator
from app.agents.skills import build_skills_backend, SKILLS_ROUTE, SKILLS_DIR


def parse_frontmatter(text: str) -> dict:
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.DOTALL)
    if not m:
        raise ValueError("缺少 YAML frontmatter（文件必须以 --- 开头，并以单独一行 --- 结束）")
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        raise ValueError("frontmatter 不是 YAML 映射")
    return data


def run_checks() -> bool:
    print("=" * 60)
    print("开始运行 Skills 离线冒烟测试...")
    print("=" * 60)
    all_passed = True

    # 1. 遍历 skills/*/SKILL.md 检查规范
    pattern = os.path.join(SKILLS_DIR, "*", "SKILL.md")
    skill_files = glob.glob(pattern)

    if not skill_files:
        print("❌ 未发现任何 skills/*/SKILL.md 文件")
        return False

    name_regex = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

    for sf in skill_files:
        skill_dir_name = os.path.basename(os.path.dirname(sf))
        rel_path = os.path.relpath(sf, str(_PROJECT_ROOT))
        print(f"\n[检查技能文件] {rel_path}:")

        with open(sf, "rb") as f:
            raw_bytes = f.read()

        # 检查 BOM
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            print("  ❌ 错误: 文件以 UTF-8 BOM 开头")
            all_passed = False
            continue
        print("  ✅ 编码检查: 无 UTF-8 BOM")

        text = raw_bytes.decode("utf-8")

        # 行数检查 <= 100
        line_count = len(text.splitlines())
        if line_count > 100:
            print(f"  ❌ 错误: 文件行数 ({line_count}) > 100")
            all_passed = False
        else:
            print(f"  ✅ 行数检查: {line_count} 行 (≤ 100)")

        # 解析 frontmatter
        try:
            fm = parse_frontmatter(text)
            print("  ✅ Frontmatter 解析通过")
        except Exception as e:
            print(f"  ❌ 错误: Frontmatter 解析失败: {e}")
            all_passed = False
            continue

        # name 检查
        name = fm.get("name")
        if not name or name != skill_dir_name:
            print(f"  ❌ 错误: name '{name}' 与目录名 '{skill_dir_name}' 不一致")
            all_passed = False
        elif not name_regex.match(name):
            print(f"  ❌ 错误: name '{name}' 不符合规范 '^[a-z0-9]+(-[a-z0-9]+)*$'")
            all_passed = False
        else:
            print(f"  ✅ 名称检查: '{name}' 格式正确且与目录名一致")

        # description 检查
        desc = fm.get("description", "")
        if not desc or not isinstance(desc, str) or len(desc.strip()) == 0:
            print("  ❌ 错误: description 为空")
            all_passed = False
        elif len(desc) > 1024:
            print(f"  ❌ 错误: description 长度 ({len(desc)}) > 1024 字符")
            all_passed = False
        else:
            print(f"  ✅ 描述检查: 非空且长度 {len(desc)} ≤ 1024")

        # 检查关键字 /workspace/outputs/ 与 plt.show
        if "/workspace/outputs/" not in text:
            print("  ❌ 错误: SKILL.md 未包含 '/workspace/outputs/'")
            all_passed = False
        else:
            print("  ✅ 关键字检查: 包含 '/workspace/outputs/'")

        if "plt.show" not in text:
            print("  ❌ 错误: SKILL.md 未包含 'plt.show'")
            all_passed = False
        else:
            print("  ✅ 关键字检查: 包含 'plt.show'")

    # 2. backend 装配与虚拟路径检查
    print(f"\n[检查 Backend 装配与路由]:")
    backend = build_skills_backend()

    # 检查 ls 列出 /skills/
    try:
        ls_res = backend.ls(SKILLS_ROUTE)
        entries = ls_res.entries if hasattr(ls_res, "entries") else ls_res
        paths = [e["path"] for e in (entries or [])]
        expected_dir = f"{SKILLS_ROUTE}matplotlib-chart/"
        if expected_dir in paths or any("matplotlib-chart" in p for p in paths):
            print(f"  ✅ backend.ls('{SKILLS_ROUTE}') 成功列出技能目录: {paths}")
        else:
            print(f"  ❌ 错误: backend.ls('{SKILLS_ROUTE}') 未能找到 '{expected_dir}', 得到: {paths}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 错误: backend.ls 失败: {e}")
        all_passed = False

    # 检查读取 /skills/matplotlib-chart/SKILL.md
    target_skill_path = f"{SKILLS_ROUTE}matplotlib-chart/SKILL.md"
    try:
        read_res = backend.read(target_skill_path)
        content = ""
        if hasattr(read_res, "file_data") and isinstance(read_res.file_data, dict):
            content = read_res.file_data.get("content", "")
        elif isinstance(read_res, str):
            content = read_res

        if content and "/workspace/outputs/" in content and "matplotlib-chart" in content:
            print(f"  ✅ backend.read('{target_skill_path}') 成功读取内容 ({len(content)} 字符)")
        else:
            print(f"  ❌ 错误: backend.read 内容不符合预期: {read_res}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 错误: backend.read 失败: {e}")
        all_passed = False

    # 检查 download_files (SkillsMiddleware 使用的加载方法)
    try:
        dl_res = backend.download_files([target_skill_path])
        if dl_res and len(dl_res) > 0 and dl_res[0].content and b"/workspace/outputs/" in dl_res[0].content:
            print(f"  ✅ backend.download_files(['{target_skill_path}']) 成功下载 ({len(dl_res[0].content)} 字节)")
        else:
            print(f"  ❌ 错误: backend.download_files 返回异常: {dl_res}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 错误: backend.download_files 失败: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("测试结果: 全部通过 (PASS)")
        print("=" * 60)
        return True
    else:
        print("测试结果: 存在失败项 (FAIL)")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_checks()
    if not success:
        sys.exit(1)
