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
  H browser_service 幂等关闭
  I milvus_client 生命周期与双重检查锁
  K Neo4j 图约束确保与异常容错
  L1 Tavily Extract 正常提取与截断
  L2 Tavily Extract failed_results 容错
  L3 Tavily Extract 状态码错误与 Key 脱敏
  L4 Tavily Extract 非法 URL 与缺失 Key 拦截
  L5 web_search 异常字符串识别与加固
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


def test_k_neo4j_schema_constraints():
    from app.knowledge.media import builder

    executed = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, stmt, **kw):
            executed.append(stmt)

    class FakeDriver:
        def __init__(self):
            self.closed = False

        def session(self):
            return FakeSession()

        def close(self):
            self.closed = True

    # 1. 断言 6 条语句都带 IF NOT EXISTS 且都被执行
    fake_driver = FakeDriver()
    builder.ensure_graph_schema(fake_driver)
    assert len(executed) == 6, f"预期执行 6 条约束语句，实际执行: {len(executed)}"
    for stmt in executed:
        assert "IF NOT EXISTS" in stmt, f"缺少 IF NOT EXISTS: {stmt}"
        assert "IS UNIQUE" in stmt, f"缺少 IS UNIQUE: {stmt}"
        assert "REQUIRE" in stmt, f"缺少 REQUIRE: {stmt}"

    # 确保涵盖 6 类节点实体
    for entity in ("Work", "Actress", "Actor", "Tag", "Series", "Year"):
        assert any(f":{entity}" in s for s in executed), f"缺少针对实体 {entity} 的约束"

    # 2. session.run 抛异常时 ensure_graph_schema 只告警不中断
    class FailingSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, stmt, **kw):
            raise RuntimeError("Simulated Neo4j session run error")

    class FailingDriver:
        def session(self):
            return FailingSession()

    # 不抛异常即为通过
    builder.ensure_graph_schema(FailingDriver())

    # 3. MediaDataBuilder.close 关闭 driver 测试
    b = builder.MediaDataBuilder.__new__(builder.MediaDataBuilder)
    test_driver = FakeDriver()
    b.neo4j_driver = test_driver
    b.close()
    assert test_driver.closed is True
    assert b.neo4j_driver is None
    # 幂等调用
    b.close()




def test_l1_webpage_extract_and_truncate():
    """L1 Tavily Extract 正常提取与截断验证"""
    import asyncio
    import json
    import httpx
    from app.infra import browser_service
    from app.tools.browser_tools import read_webpage_content

    orig_key = getattr(config, "TAVILY_API_KEY", None)
    orig_client = browser_service._http_client
    max_chars = getattr(config, "WEBPAGE_MAX_CHARS", 30000)
    long_md = "# Long Markdown\n" + ("abcdefghij" * 4000)  # 40016 chars > 30000

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.tavily.com/extract"
        assert request.headers.get("Authorization") == "Bearer mock_tavily_key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["urls"] == ["https://example.com/test-article"]
        assert body["extract_depth"] == "basic"
        assert body["format"] == "markdown"
        return httpx.Response(200, json={
            "results": [{"url": "https://example.com/test-article", "raw_content": long_md}],
            "failed_results": []
        })

    try:
        config.TAVILY_API_KEY = "mock_tavily_key"
        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        # 1. 验证 fetch_generic_webpage_content 原样返回完整 Markdown，不截断
        raw_res = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com/test-article"))
        assert raw_res == long_md, "fetch_generic_webpage_content 应原样返回 Markdown 正文"

        # 2. 验证 read_webpage_content 按 config.WEBPAGE_MAX_CHARS 正确截断
        truncated = asyncio.run(read_webpage_content.ainvoke({"url": "https://example.com/test-article"}))
        assert truncated.startswith(long_md[:max_chars]), "read_webpage_content 应保留前 WEBPAGE_MAX_CHARS 字符"
        assert f"…（内容过长，已截断至前 {max_chars} 字符）" in truncated
        assert len(truncated) == max_chars + len(f"\n…（内容过长，已截断至前 {max_chars} 字符）")
    finally:
        config.TAVILY_API_KEY = orig_key
        browser_service._http_client = orig_client


def test_l2_webpage_failed_results():
    """L2 Tavily Extract failed_results 容错与空正文处理"""
    import asyncio
    import httpx
    from app.infra import browser_service

    orig_key = getattr(config, "TAVILY_API_KEY", None)
    orig_client = browser_service._http_client

    def handler_failed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "results": [],
            "failed_results": [{"url": "https://example.com/failed", "error": "Failed to retrieve content"}]
        })

    def handler_empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "results": [{"url": "https://example.com/empty", "raw_content": "   "}],
            "failed_results": []
        })

    try:
        config.TAVILY_API_KEY = "mock_tavily_key"

        # 1. failed_results 数组非空
        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_failed))
        res_fail = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com/failed"))
        assert res_fail.startswith("Error:"), f"预期以 Error: 开头，实际: {res_fail}"
        assert "Failed to retrieve content" in res_fail
        assert "https://example.com/failed" in res_fail

        # 2. results raw_content 为空
        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_empty))
        res_empty = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com/empty"))
        assert res_empty.startswith("Error:"), f"预期以 Error: 开头，实际: {res_empty}"
        assert "网页内容提取为空" in res_empty
    finally:
        config.TAVILY_API_KEY = orig_key
        browser_service._http_client = orig_client


def test_l3_webpage_errors_and_redaction():
    """L3 Tavily Extract 状态码错误、网络异常与 API Key 脱敏"""
    import asyncio
    import httpx
    from app.infra import browser_service

    orig_key = getattr(config, "TAVILY_API_KEY", None)
    orig_client = browser_service._http_client
    sensitive_key = "tvly-secret-key-123456789abcdef"

    try:
        config.TAVILY_API_KEY = sensitive_key

        # 1. HTTP 401 携带 Key 明文响应
        def handler_401(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": f"Unauthorized key: {sensitive_key}"})

        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_401))
        res_401 = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_401.startswith("Error: Tavily Extract 接口请求失败 (HTTP 401):"), res_401
        assert sensitive_key not in res_401, "API Key 未脱敏"
        assert "***" in res_401

        # 2. HTTP 429 配额耗尽
        def handler_429(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": f"Quota exceeded for {sensitive_key}"})

        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_429))
        res_429 = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_429.startswith("Error: Tavily Extract 接口请求失败 (HTTP 429):"), res_429
        assert sensitive_key not in res_429, "API Key 未脱敏"
        assert "***" in res_429

        # 3. HTTP 500 非 JSON 响应
        def handler_500(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text=f"Crash dump with {sensitive_key}")

        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_500))
        res_500 = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_500.startswith("Error: Tavily Extract 接口请求失败 (HTTP 500):"), res_500
        assert sensitive_key not in res_500, "API Key 未脱敏"
        assert "***" in res_500

        # 4. HTTP 200 非法 JSON
        def handler_bad_json(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not a valid json")

        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_bad_json))
        res_json = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_json == "Error: Tavily Extract 返回了非法的 JSON 响应。"

        # 5. 超时异常 (TimeoutException)
        def handler_timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout(f"Timed out with key {sensitive_key}")

        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_timeout))
        res_timeout = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_timeout == "Error: 网页内容提取超时 (https://example.com)。"
        assert sensitive_key not in res_timeout

        # 6. 网络连接错误 (HTTPError / Exception)
        def handler_net_err(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"Connection refused with {sensitive_key}")

        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler_net_err))
        res_err = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_err.startswith("Error: 网页内容提取请求异常:"), res_err
        assert sensitive_key not in res_err, "API Key 未脱敏"
        assert "***" in res_err

    finally:
        config.TAVILY_API_KEY = orig_key
        browser_service._http_client = orig_client


def test_l4_webpage_url_and_key_guard():
    """L4 Tavily Extract 非法 URL 与缺失 API Key 短路拦截"""
    import asyncio
    import httpx
    from app.infra import browser_service

    orig_key = getattr(config, "TAVILY_API_KEY", None)
    orig_client = browser_service._http_client

    def guard_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("短路校验失败：不应发起任何网络请求！")

    try:
        browser_service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(guard_handler))
        config.TAVILY_API_KEY = "mock_tavily_key"

        # 1. 非法 URL 拦截测试
        bad_urls = ["", "   ", "ftp://example.com/file", "javascript:alert(1)", "http://", "relative/path"]
        for bad_url in bad_urls:
            res = asyncio.run(browser_service.fetch_generic_webpage_content(bad_url))
            assert res == "Error: 无效的 URL，必须是以 http:// 或 https:// 开头的完整网址。", (
                f"URL '{bad_url}' 未被正确拦截: {res}"
            )

        # 2. 缺失 API Key 拦截测试
        config.TAVILY_API_KEY = ""
        res_nokey = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_nokey == "Error: 未配置 TAVILY_API_KEY，无法调用云端网页解析服务。", res_nokey

        config.TAVILY_API_KEY = "   "
        res_space = asyncio.run(browser_service.fetch_generic_webpage_content("https://example.com"))
        assert res_space == "Error: 未配置 TAVILY_API_KEY，无法调用云端网页解析服务。", res_space

    finally:
        config.TAVILY_API_KEY = orig_key
        browser_service._http_client = orig_client


def test_l5_web_search_hardening():
    """L5 web_search 异常字符串识别、空结果与 Key 守卫"""
    from app.tools import knowledge_tools

    orig_key = getattr(config, "TAVILY_API_KEY", None)
    orig_cls = knowledge_tools.TavilySearchResults

    try:
        # 1. 未配置 TAVILY_API_KEY
        config.TAVILY_API_KEY = ""
        res_nokey = knowledge_tools.web_search.invoke({"query": "python"})
        assert res_nokey == "Error: 未配置 TAVILY_API_KEY，网络搜索不可用。", res_nokey

        # 2. 模拟 TavilySearchResults.run 返回异常 repr 字符串
        config.TAVILY_API_KEY = "mock_key"

        class MockTavilySearchError:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, query: str):
                return "HTTPError('401 Client Error')"

        knowledge_tools.TavilySearchResults = MockTavilySearchError
        res_err_str = knowledge_tools.web_search.invoke({"query": "python"})
        assert res_err_str == "Error: 网络搜索失败: HTTPError('401 Client Error')", res_err_str

        # 3. 模拟 TavilySearchResults.run 返回空列表 []
        class MockTavilySearchEmpty:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, query: str):
                return []

        knowledge_tools.TavilySearchResults = MockTavilySearchEmpty
        res_empty = knowledge_tools.web_search.invoke({"query": "python"})
        assert res_empty == "未搜索到相关网络结果。", res_empty

        # 4. 模拟 TavilySearchResults.run 返回正常列表
        class MockTavilySearchNormal:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, query: str):
                return [{"url": "https://python.org", "content": "Python documentation"}]

        knowledge_tools.TavilySearchResults = MockTavilySearchNormal
        res_normal = knowledge_tools.web_search.invoke({"query": "python"})
        assert res_normal == str([{"url": "https://python.org", "content": "Python documentation"}]), res_normal

        # 5. 模拟抛出 Exception
        class MockTavilySearchRaise:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, query: str):
                raise RuntimeError("Search socket closed")

        knowledge_tools.TavilySearchResults = MockTavilySearchRaise
        res_raise = knowledge_tools.web_search.invoke({"query": "python"})
        assert res_raise == "Error: 网络搜索失败: Search socket closed", res_raise

    finally:
        config.TAVILY_API_KEY = orig_key
        knowledge_tools.TavilySearchResults = orig_cls


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
    ("K Neo4j 图约束确保与异常容错", test_k_neo4j_schema_constraints),
    ("L1 Tavily Extract 正常提取与截断", test_l1_webpage_extract_and_truncate),
    ("L2 Tavily Extract failed_results 容错", test_l2_webpage_failed_results),
    ("L3 Tavily Extract 状态码错误与 Key 脱敏", test_l3_webpage_errors_and_redaction),
    ("L4 Tavily Extract 非法 URL 与缺失 Key 拦截", test_l4_webpage_url_and_key_guard),
    ("L5 web_search 异常字符串识别与加固", test_l5_web_search_hardening),
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
