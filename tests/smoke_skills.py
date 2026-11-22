"""
tests/smoke_skills.py

Skills 离线冒烟测试：验证 skills/*/SKILL.md 规范与 app/agents/skills.py backend 装配。
不依赖外部服务、不需要任何 API Key。
"""
import asyncio
import glob
import hashlib
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
from app.agents.skills import (
    SKILLS_DIR,
    SKILLS_MEMBER_DIR,
    SKILLS_ROUTE,
    ReadOnlyFilesystemBackend,
    build_member_skills_backend,
    build_skills_backend,
)

# member 的技能是给云端沙箱写的：不得出现本机沙箱 / 宿主机 / 其他专家相关的词
_MEMBER_FORBIDDEN_WORDS = ("Docker", "/workspace", "workspace", "宿主机", "FileAgent", "DesktopAgent", "BrowserAgent")


def parse_frontmatter(text: str) -> dict:
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.DOTALL)
    if not m:
        raise ValueError("缺少 YAML frontmatter（文件必须以 --- 开头，并以单独一行 --- 结束）")
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        raise ValueError("frontmatter 不是 YAML 映射")
    return data


def _tree_hashes(root: str) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).rglob("*")) if p.is_file()}


def run_member_checks() -> bool:
    """member 的技能：skills_member/ 的规范、只读后端、路径穿越。skills/（owner 用）一个字都不动。"""
    print("\n[member 技能 skills_member/ 与只读后端]:")
    ok = True

    def check(condition: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        if condition:
            print(f"  ✅ {label}")
        else:
            ok = False
            print(f"  ❌ {label}" + (f": {detail}" if detail else ""))

    name_regex = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
    files = sorted(glob.glob(os.path.join(SKILLS_MEMBER_DIR, "*", "SKILL.md")))
    check({os.path.basename(os.path.dirname(f)) for f in files} == {"matplotlib-chart", "data-analysis"}, "skills_member/ 恰好有 matplotlib-chart 与 data-analysis", str(files))
    for sf in files:
        skill = os.path.basename(os.path.dirname(sf))
        raw = open(sf, "rb").read()
        text = raw.decode("utf-8")
        check(not raw.startswith(b"\xef\xbb\xbf"), f"{skill}: 无 UTF-8 BOM")
        check(len(text.splitlines()) <= 100, f"{skill}: 行数 ≤ 100", str(len(text.splitlines())))
        try:
            fm = parse_frontmatter(text)
        except Exception as e:
            check(False, f"{skill}: frontmatter 解析", str(e))
            continue
        check(fm.get("name") == skill and bool(name_regex.match(str(fm.get("name")))), f"{skill}: name 与目录名一致且格式正确")
        desc = fm.get("description")
        check(isinstance(desc, str) and 0 < len(desc.strip()) <= 1024, f"{skill}: description 非空且 ≤ 1024")
        bad = [w for w in _MEMBER_FORBIDDEN_WORDS if w in text]
        check(not bad, f"{skill}: 不含 Docker / workspace / 宿主机 / 其他专家名", str(bad))
        if skill == "matplotlib-chart":
            check("/work/outputs/" in text and "plt.show" in text, f"{skill}: 含 /work/outputs/ 与 plt.show 禁令")
        else:
            check("/work" in text and "仅当 CodeAgent 的运行环境说明写明沙箱可以联网" in text, f"{skill}: 围绕 /work，且联网相关句子是有条件的")

    # 后端装配：member 的 /skills/ 是只读后端、根目录是 skills_member/；owner 的不变
    member_backend = build_member_skills_backend()
    route = member_backend.routes[SKILLS_ROUTE]
    check(isinstance(route, ReadOnlyFilesystemBackend) and os.path.realpath(str(route.cwd)) == os.path.realpath(SKILLS_MEMBER_DIR),
          "build_member_skills_backend: /skills/ 是只读后端且指向 skills_member/")
    owner_route = build_skills_backend().routes[SKILLS_ROUTE]
    check(not isinstance(owner_route, ReadOnlyFilesystemBackend) and os.path.realpath(str(owner_route.cwd)) == os.path.realpath(SKILLS_DIR),
          "build_skills_backend（owner）不变：可写后端、指向 skills/")

    # ReadOnlyFilesystemBackend：全部写类方法都返回带 error 的结果，磁盘上没有任何变化
    snap_member, snap_owner = _tree_hashes(SKILLS_MEMBER_DIR), _tree_hashes(SKILLS_DIR)
    ro = ReadOnlyFilesystemBackend(root_dir=SKILLS_MEMBER_DIR, virtual_mode=True)
    target = "/matplotlib-chart/SKILL.md"
    results = {
        "write": ro.write(target, "pwned"), "awrite": asyncio.run(ro.awrite(target, "pwned")),
        "edit": ro.edit(target, "matplotlib", "pwned"), "aedit": asyncio.run(ro.aedit(target, "matplotlib", "pwned")),
        "delete": ro.delete(target), "adelete": asyncio.run(ro.adelete(target)),
    }
    for name, res in results.items():
        check(bool(res.error) and "只读" in res.error, f"ReadOnly.{name} 返回只读错误", repr(res))
    for name, res in (("upload_files", ro.upload_files([(target, b"pwned"), ("/new.md", b"x")])),
                      ("aupload_files", asyncio.run(ro.aupload_files([(target, b"pwned")])))):
        check(res and all(r.error and "只读" in r.error for r in res), f"ReadOnly.{name} 每一项都返回只读错误", repr(res))
    check(_tree_hashes(SKILLS_MEMBER_DIR) == snap_member and _tree_hashes(SKILLS_DIR) == snap_owner, "写类方法之后磁盘上没有任何变化")
    check(not (Path(SKILLS_MEMBER_DIR) / "new.md").exists(), "没有新建文件")

    # 读类方法仍可用
    listing = ro.ls("/")
    entries = listing.entries if hasattr(listing, "entries") else listing
    check(any("matplotlib-chart" in str(e) for e in (entries or [])), "ReadOnly.ls 仍可用", repr(listing))
    read = ro.read(target)
    content = getattr(getattr(read, "file_data", None), "get", lambda *_: "")("content", "") if isinstance(getattr(read, "file_data", None), dict) else str(read)
    check("/work/outputs/" in content, "ReadOnly.read 仍可用", repr(read)[:120])
    check(ro.grep("plt.show", "/") is not None and ro.glob("*/SKILL.md", "/") is not None, "ReadOnly.grep / glob 仍可用")
    dl = ro.download_files([target])
    check(dl and dl[0].content and b"/work/outputs/" in dl[0].content, "ReadOnly.download_files 仍可用")

    # 经 CompositeBackend 做路径穿越：读 / 写 / 列目录都拿不到 skills_member/ 之外的东西
    marker_root = Path(SKILLS_MEMBER_DIR).parent
    before_root = sorted(p.name for p in marker_root.iterdir())
    def attempt(fn):
        """返回 (结果文本, 是否被拦截)：路径穿越要么抛 ValueError（Path traversal not allowed），要么返回带 error 的结果。"""
        try:
            res = fn()
        except ValueError as e:
            return f"ValueError: {e}", True
        return repr(res), bool(getattr(res, "error", None))

    for path in ("/skills/../.env", "/skills/a/../../README.md", "/skills/~", "/skills/../pyproject.toml"):
        shown, blocked = attempt(lambda: member_backend.read(path))
        leaked = [w for w in ("MODAL_TOKEN", "API_KEY", "[project]", "MYSQL_PASSWORD") if w in shown]
        check(not leaked, f"读 {path} 拿不到 skills_member/ 之外的内容", shown[:120])
        shown, blocked = attempt(lambda: member_backend.write(path, "pwned"))
        check(blocked, f"写 {path} 被拒绝", shown[:120])
    shown, _ = attempt(lambda: member_backend.ls("/skills/.."))
    check(not any(w in shown for w in ("pyproject.toml", "main.py", ".env", "README.md")), "列 /skills/.. 看不到项目根目录", shown[:160])
    check(sorted(p.name for p in marker_root.iterdir()) == before_root, "穿越尝试没有在项目根目录里留下任何文件")
    check(_tree_hashes(SKILLS_MEMBER_DIR) == snap_member and _tree_hashes(SKILLS_DIR) == snap_owner, "穿越尝试之后两个技能目录都没变")
    return ok


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

        # 按技能分别断言关键字
        if skill_dir_name == "matplotlib-chart":
            if "/work/outputs/" not in text:
                print("  ❌ 错误: matplotlib-chart SKILL.md 未包含 '/work/outputs/'")
                all_passed = False
            else:
                print("  ✅ 关键字检查: 包含 '/work/outputs/'")

            if "plt.show" not in text:
                print("  ❌ 错误: matplotlib-chart SKILL.md 未包含 'plt.show'")
                all_passed = False
            else:
                print("  ✅ 关键字检查: 包含 'plt.show'")
        elif skill_dir_name == "data-analysis":
            if "/workspace" not in text:
                print("  ❌ 错误: data-analysis SKILL.md 未包含 '/workspace'")
                all_passed = False
            else:
                print("  ✅ 关键字检查: 包含 '/workspace'")

            if "/work" not in text:
                print("  ❌ 错误: data-analysis SKILL.md 未包含 '/work'")
                all_passed = False
            else:
                print("  ✅ 关键字检查: 包含 '/work'")

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

        if content and "/work/outputs/" in content and "matplotlib-chart" in content:
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
        if dl_res and len(dl_res) > 0 and dl_res[0].content and b"/work/outputs/" in dl_res[0].content:
            print(f"  ✅ backend.download_files(['{target_skill_path}']) 成功下载 ({len(dl_res[0].content)} 字节)")
        else:
            print(f"  ❌ 错误: backend.download_files 返回异常: {dl_res}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ 错误: backend.download_files 失败: {e}")
        all_passed = False

    if not run_member_checks():
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
