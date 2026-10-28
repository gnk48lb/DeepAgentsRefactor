"""
tests/smoke_units.py

纯函数 / 小模块的离线单元冒烟测试(不需要任何 key / 网络 / Docker):
  A config._resolve_path:URI 原样、绝对路径原样、相对路径相对项目根
  B bootstrap.apply:Windows 下 os.rename 遇到 FileExistsError 时回退到 os.replace(在任意平台模拟)
  C utils.images.guess_mime_type
  D DesktopAgent 截图历史处理:只有最新一张合法截图保留图片,历史截图替换为占位;失败文本原样保留
  E DesktopAgent 未应答 tool_calls 的判定
  F SQL 加固与超时只读
  G rag 同源图片描述只展开一次
"""
import base64
import os
import sys
import tempfile
import traceback
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import bootstrap, config
from app.agents.experts import desktop_agent as da
from app.tools import sql_tools
from app.tools.knowledge_tools import _build_context
from app.utils.images import guess_mime_type


def test_a_resolve_path():
    r = config._resolve_path
    assert r("") == ""
    assert r("http://localhost:19530") == "http://localhost:19530"
    abs_p = os.path.abspath(os.path.join(os.sep, "tmp", "x.db"))
    assert r(abs_p) == abs_p
    rel = r("storage/x.db")
    assert os.path.isabs(rel)
    assert rel == os.path.normpath(os.path.join(config.PROJECT_ROOT, "storage", "x.db")), rel


def test_b_bootstrap_rename_fallback():
    real_platform, real_rename, real_applied = sys.platform, os.rename, bootstrap._applied

    def fake_rename(src, dst):        # 模拟 Windows:目标已存在时 os.rename 抛 FileExistsError
        raise FileExistsError(17, "exists", dst)

    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "a.txt"), os.path.join(td, "b.txt")
        open(a, "w").write("new")
        open(b, "w").write("old")
        try:
            sys.platform = "win32"
            os.rename = fake_rename
            bootstrap._applied = False
            bootstrap.apply()
            assert getattr(os.rename, "_is_safe_rename", False), "apply() 应安装 os.rename 兜底补丁"
            os.rename(a, b)           # 补丁应回退到 os.replace
            assert open(b).read() == "new" and not os.path.exists(a)
            bootstrap.apply()         # 幂等:不应再包一层
        finally:
            sys.platform, os.rename, bootstrap._applied = real_platform, real_rename, real_applied


def test_c_guess_mime():
    assert guess_mime_type(b"\x89PNG\r\n\x1a\n....") == "image/png"
    assert guess_mime_type(b"\xff\xd8\xff....") == "image/jpeg"
    assert guess_mime_type(b"GIF89a....") == "image/gif"
    assert guess_mime_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert guess_mime_type(b"unknown") == "image/jpeg"
    assert guess_mime_type(b"unknown", default="image/png") == "image/png"


def _shot_call(i):
    return AIMessage(content="", tool_calls=[{"name": "take_grid_screenshot", "args": {}, "id": f"c{i}"}])


def test_d_desktop_screenshot_history():
    b64 = base64.b64encode(b"x" * 300).decode()
    msgs = [_shot_call(1), ToolMessage(content=b64, tool_call_id="c1", name="take_grid_screenshot"),
            _shot_call(2), ToolMessage(content=b64, tool_call_id="c2", name="take_grid_screenshot")]
    out = da._process_desktop_messages_for_llm(msgs)
    tool_msgs = [m for m in out if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content == da.HISTORICAL_SCREENSHOT_PLACEHOLDER, "历史截图应替换为占位文案"
    assert tool_msgs[1].content == da.LATEST_SCREENSHOT_TOOL_MESSAGE
    humans = [m for m in out if isinstance(m, HumanMessage)]
    assert len(humans) == 1 and humans[0].content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # 最新一张是失败文本(非合法 base64):原样保留,且不追加带图消息
    bad = [_shot_call(1), ToolMessage(content="截图失败: 权限不足", tool_call_id="c1", name="take_grid_screenshot")]
    out2 = da._process_desktop_messages_for_llm(bad)
    assert out2[1].content == "截图失败: 权限不足" and not [m for m in out2 if isinstance(m, HumanMessage)]


def test_e_unanswered_tool_calls():
    ai = AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"}, {"name": "b", "args": {}, "id": "2"}])
    msgs = [ai, ToolMessage(content="ok", tool_call_id="1", name="a")]
    pending = da._unanswered_tool_calls(msgs)
    assert [tc["id"] for tc in pending] == ["2"], pending
    assert da._unanswered_tool_calls([ai, ToolMessage(content="ok", tool_call_id="1", name="a"),
                                      ToolMessage(content="ok", tool_call_id="2", name="b")]) == []


def test_f_sql_hardening():
    prep = sql_tools._prepare_select_sql
    assert prep("SELECT * FROM works;") == "SELECT * FROM works\nLIMIT 50"
    for bad in ["DELETE FROM works", "SELECT 1; DROP TABLE works",
                "SELECT * FROM works INTO OUTFILE '/tmp/x'", "select load_file('/etc/passwd')",
                "SELECT 1 INTO   dumpfile '/tmp/y'"]:
        try:
            prep(bad)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝: {bad}")
    captured = {}

    class _Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql): captured["sql"] = sql
        def fetchall(self): return [{"a": 1}]

    class _Conn:
        def cursor(self): return _Cur()
        def close(self): captured["closed"] = True

    real = sql_tools.pymysql.connect
    sql_tools.pymysql.connect = lambda **kw: (captured.update(kw), _Conn())[1]
    try:
        out = sql_tools.execute_sql.invoke({"sql_query": "SELECT 1"})
    finally:
        sql_tools.pymysql.connect = real
    assert out == "[{'a': 1}]", out
    assert captured["connect_timeout"] == 10 and captured["read_timeout"] == 30, captured
    assert "READ ONLY" in captured["init_command"].upper(), captured
    assert captured["sql"].endswith("LIMIT 50") and captured["closed"]




def test_g_rag_image_description_dedup():
    orig_data_dir = config.DATA_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            config.DATA_DIR = td
            unique_tag = "UNIQUE_TAG_DEDUP_98765_ABCD"
            summary_content = ("这是图片摘要测试文本。" * 150) + unique_tag + ("这是图片摘要后续文本。" * 150)
            summary_file = Path(td) / "summary.md"
            summary_file.write_text(summary_content, encoding="utf-8")

            # 构造 3 条 source_path 与 image_path 相同、content_type="image_description" 的假命中
            hit1 = (0.95, {"entity": {"text": "chunk1", "source_path": "summary.md", "chunk_id": "c1", "content_type": "image_description", "image_path": "img.png"}})
            hit2 = (0.90, {"entity": {"text": "chunk2", "source_path": "summary.md", "chunk_id": "c2", "content_type": "image_description", "image_path": "img.png"}})
            hit3 = (0.85, {"entity": {"text": "chunk3", "source_path": "summary.md", "chunk_id": "c3", "content_type": "image_description", "image_path": "img.png"}})
            # 混入 1 条 content_type="text" 的命中
            text_hit = (0.80, {"entity": {"text": "普通文本知识片段", "source_path": "doc.txt", "chunk_id": "c4", "content_type": "text", "image_path": ""}})

            best_hits = [hit1, hit2, hit3, text_hit]
            context_docs, image_paths = _build_context(best_hits)
            context = "\n".join(context_docs)

            # 断言拼出的 context 中该唯一标记串只出现 1 次、image_paths 长度为 1
            assert context.count(unique_tag) == 1, f"唯一标记串出现次数不为 1: {context.count(unique_tag)}"
            assert len(image_paths) == 1, f"image_paths 长度不为 1: {len(image_paths)}"
            assert image_paths == ["img.png"]
            # 断言 content_type='text' 仍被保留
            assert any("普通文本知识片段" in doc for doc in context_docs), "普通文本知识命中未被保留"
            assert len(context_docs) == 2, f"context_docs 预期包含 2 条，实际为: {len(context_docs)}"
    finally:
        config.DATA_DIR = orig_data_dir


def test_h_browser_close_idempotent():
    import asyncio
    from app.infra import browser_service

    # Case 1: _browser is None
    browser_service._browser = None
    asyncio.run(browser_service.close_browser())
    assert browser_service._browser is None

    # Case 2: _browser exists, kill called and set to None
    kill_called = 0

    class MockBrowser:
        async def kill(self):
            nonlocal kill_called
            kill_called += 1

    browser_service._browser = MockBrowser()
    asyncio.run(browser_service.close_browser())
    assert kill_called == 1
    assert browser_service._browser is None

    # Case 3: Idempotent call
    asyncio.run(browser_service.close_browser())
    assert kill_called == 1


def test_i_milvus_client_lifecycle():
    import concurrent.futures
    import time
    from app.infra import milvus_client

    # Case 1: client is None
    milvus_client.client = None
    milvus_client.close_client()
    assert milvus_client.client is None

    # Case 2: client exists, close called and set to None
    close_called = 0

    class MockClient:
        def close(self):
            nonlocal close_called
            close_called += 1

    milvus_client.client = MockClient()
    milvus_client.close_client()
    assert close_called == 1
    assert milvus_client.client is None

    # Case 3: Idempotent call
    milvus_client.close_client()
    assert close_called == 1

    # Case 4: Thread-safe double-checked lock on get_client
    created = 0

    class FakeMilvusClient:
        def __init__(self, *args, **kwargs):
            nonlocal created
            created += 1
            time.sleep(0.01)

    orig_mc = milvus_client.MilvusClient
    try:
        milvus_client.MilvusClient = FakeMilvusClient
        milvus_client.client = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda _: milvus_client.get_client(), range(5)))
        assert created == 1, f"Expected 1 client created, got {created}"
        assert len(set(results)) == 1, "Expected identical singleton client instance"
    finally:
        milvus_client.client = None
        milvus_client.MilvusClient = orig_mc


SCENARIOS = [
    ("A config._resolve_path", test_a_resolve_path),
    ("B bootstrap os.rename 兜底补丁", test_b_bootstrap_rename_fallback),
    ("C guess_mime_type", test_c_guess_mime),
    ("D DesktopAgent 截图历史处理", test_d_desktop_screenshot_history),
    ("E DesktopAgent 未应答 tool_calls", test_e_unanswered_tool_calls),
    ("F SQL 加固与超时只读", test_f_sql_hardening),
    ("G rag 同源图片描述只展开一次", test_g_rag_image_description_dedup),
    ("H browser_service 幂等关闭", test_h_browser_close_idempotent),
    ("I milvus_client 生命周期与双重检查锁", test_i_milvus_client_lifecycle),
]


def main():
    print("=" * 70)
    print("开始运行纯函数单元冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e!r}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
