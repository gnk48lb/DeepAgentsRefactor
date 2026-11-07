"""
tests/smoke_session.py

会话层离线冒烟测试（不需要任何 API Key / Docker / 网络）：
  A 建表 / 默认会话 / 活动指针 / 标题自动生成 / 前缀解析
  B 事件追加与读取    C 幂等 key    D 待处理游标    E turn_state    F 搜索
  G 关闭重开          H 并发 emit   I delete_session
  J session_log.mirror_messages     K session_tools
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import config
from app.infra import session_store
from app.infra.session_store import INBOUND_EVENT_TYPES, SessionStore
from app.services import session_log
from app.tools import session_tools


class _Env:
    """临时目录里的 SessionStore；退出时先关连接（含 -wal / -shm）再删目录（Windows 要求）。"""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="sess_")
        self.path = os.path.join(self.dir, "sub", "sessions.db")  # 父目录不存在，验证自动创建
        self.store = SessionStore(self.path)
        session_store.set_store(self.store)
        return self

    def reopen(self):
        self.store.close()
        self.store = SessionStore(self.path)
        session_store.set_store(self.store)
        return self.store

    def __exit__(self, *exc):
        session_store.close_store()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def test_a_sessions_basics():
    with _Env() as env:
        s = env.store
        assert s.count_sessions() == 0
        sid = s.get_active_session_id()                       # 首次无会话 → 自动建默认会话
        assert s.count_sessions() == 1 and s.get_session(sid).title == ""
        first = s.get_events(sid)
        assert [e.type for e in first] == ["session_created"], first
        assert s.get_active_session_id() == sid               # 幂等

        s.emit_event(sid, "user_message", {"text": "帮我查一下济南\n今天的天气", "processed_text": "x", "has_image": False})
        assert s.get_session(sid).title == "帮我查一下济南 今天的天气"[: config.SESSION_TITLE_MAX_CHARS]
        s.emit_event(sid, "user_message", {"text": "第二句不应改标题"})
        assert "第二句" not in s.get_session(sid).title

        b = s.create_session("手动标题" + "长" * 50, activate=False)
        assert len(b.title) == config.SESSION_TITLE_MAX_CHARS
        assert s.get_active_session_id() == sid               # activate=False 不动指针
        s.set_active_session(b.id)
        assert s.get_active_session_id() == b.id
        try:
            s.set_active_session("nope")
            raise AssertionError("不存在的会话应抛 ValueError")
        except ValueError:
            pass

        # 前缀解析
        assert s.resolve_session(sid).id == sid and s.resolve_session(sid[:8]).id == sid
        assert s.resolve_session("zzzz") is None and s.resolve_session("") is None
        # 构造多义：两个 id 共享前缀
        with s._tx() as c:
            c.execute("INSERT INTO sessions(id,title,created_at,updated_at) VALUES ('abc111','',1,1)")
            c.execute("INSERT INTO sessions(id,title,created_at,updated_at) VALUES ('abc222','',1,1)")
        try:
            s.resolve_session("abc")
            raise AssertionError("多义前缀应抛 ValueError")
        except ValueError:
            pass
        assert s.resolve_session("abc1").id == "abc111"
        s.rename_session(sid, "改名")
        assert s.get_session(sid).title == "改名"
        lst = s.list_sessions(limit=2)
        assert len(lst) == 2 and lst[0].updated_at >= lst[1].updated_at
        assert s.get_session(sid).turns == 2 and s.get_session(sid).events == 3


def test_b_events():
    with _Env() as env:
        s = env.store
        a, b = s.create_session("a").id, s.create_session("b").id
        for i in range(5):
            s.emit_event(a, "user_message" if i % 2 == 0 else "assistant_message", {"text": f"a{i}"})
        s.emit_event(b, "user_message", {"text": "b0"})
        ev_a = s.get_events(a, limit=100)
        assert [e.seq for e in ev_a] == [1, 2, 3, 4, 5, 6], [e.seq for e in ev_a]     # session_created 占 seq 1
        assert [e.seq for e in s.get_events(b, limit=100)] == [1, 2]                  # 会话间互不影响
        assert [e.seq for e in s.get_events(a, start_seq=3, limit=2)] == [3, 4]
        assert [e.seq for e in s.get_last_events(a, limit=3)] == [4, 5, 6]            # 升序返回最新 N 条
        only_user = s.get_events(a, types=["user_message"], limit=100)
        assert [e.payload["text"] for e in only_user] == ["a0", "a2", "a4"]
        assert [e.seq for e in s.get_last_events(a, limit=1, types=["assistant_message"])] == [5]
        assert s.last_seq(a) == 6 and s.count_events(a) == 6 and s.count_events(a, ["user_message"]) == 3
        assert isinstance(ev_a[1].payload, dict)


def test_c_idempotent_key():
    with _Env() as env:
        s = env.store
        sid = s.create_session("k").id
        n0 = s.count_events(sid)
        seq = s.emit_event(sid, "tool_call", {"calls": []}, key="msg:1")
        assert seq == n0 + 1
        assert s.emit_event(sid, "tool_call", {"calls": [1]}, key="msg:1") is None
        assert s.count_events(sid) == n0 + 1
        other = s.create_session("o").id
        assert s.emit_event(other, "tool_call", {}, key="msg:1") is not None      # key 只在会话内唯一
        assert s.emit_event(sid, "tool_call", {}, key=None) is not None
        assert s.emit_event(sid, "tool_call", {}, key=None) is not None            # 无 key 不去重


def test_d_pending_cursor():
    with _Env() as env:
        s = env.store
        assert INBOUND_EVENT_TYPES == ("user_message", "approval")
        sid = s.create_session("p").id
        assert s.get_pending_events(sid) == [] and s.sessions_with_pending() == []
        s.emit_event(sid, "assistant_message", {"text": "x"})
        s.emit_event(sid, "tool_result", {"text": "x"})
        assert s.get_pending_events(sid) == []                                       # 非 inbound 不算待处理
        q1 = s.emit_event(sid, "user_message", {"text": "q1"})
        q2 = s.emit_event(sid, "approval", {"answer": "Y"})
        assert [e.seq for e in s.get_pending_events(sid)] == [q1, q2]
        assert s.sessions_with_pending() == [sid]
        s.mark_processed(sid, up_to_seq=q1)
        assert [e.seq for e in s.get_pending_events(sid)] == [q2]
        s.mark_processed(sid, up_to_seq=1)                                           # 游标只前进不后退
        assert s.get_session(sid).processed_seq == q1
        env.reopen()
        assert [e.seq for e in env.store.get_pending_events(sid)] == [q2]            # 关闭重开后游标保持
        env.store.mark_processed(sid)
        assert env.store.get_pending_events(sid) == [] and env.store.sessions_with_pending() == []
        assert env.store.get_session(sid).processed_seq == env.store.last_seq(sid)


def test_e_turn_state():
    with _Env() as env:
        s = env.store
        sid = s.create_session("t").id
        assert s.load_turn_state(sid) is None
        big = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * (1024 * 1024 + 10)}}]
        s.save_turn_state(sid, "原话", big)
        ts = s.load_turn_state(sid)
        assert ts.user_text == "原话" and ts.outbox == big
        s.save_turn_state(sid, None, [])                                              # 覆盖写
        ts = s.load_turn_state(sid)
        assert ts.user_text is None and ts.outbox == []
        s.save_turn_state(sid, "x", big)
        env.reopen()
        assert env.store.load_turn_state(sid).outbox == big                           # 重启后仍在
        env.store.clear_turn_state(sid)
        assert env.store.load_turn_state(sid) is None


def test_f_search():
    with _Env() as env:
        s = env.store
        a, b = s.create_session("a").id, s.create_session("b").id
        s.emit_event(a, "user_message", {"text": "查一下 Beijing 的天气"})
        s.emit_event(a, "expert_report", {"expert": "MapAgent", "task": "天气", "text": "北京今天晴 25度 100% 湿度_低"})
        s.emit_event(b, "user_message", {"text": "查一下上海的菜谱"})
        assert len(s.search_events("查一下")) == 2
        assert len(s.search_events("查一下 天气")) == 1                               # 多关键词 AND
        assert len(s.search_events("beijing")) == 1                                   # 大小写不敏感
        assert len(s.search_events("北京 晴")) == 1                                   # 中文
        assert len(s.search_events("100%")) == 1 and len(s.search_events("%")) == 1   # % 被转义
        assert len(s.search_events("湿度_低")) == 1 and len(s.search_events("湿度_高")) == 0
        assert len(s.search_events("_")) == 1                                         # _ 被转义，不当作通配符
        assert len(s.search_events("查一下", session_id=b)) == 1                      # 限定会话
        assert s.search_events("", session_id=a) == [] and s.search_events("   ") == []
        res = s.search_events("查一下", limit=1)
        assert len(res) == 1 and res[0].session_id == b                               # 最新优先
        assert len(s.search_events("北京", types=["user_message"])) == 0


def test_g_reopen_integrity():
    with _Env() as env:
        s = env.store
        sid = s.create_session("g").id
        for i in range(10):
            s.emit_event(sid, "user_message", {"text": f"中文 {i}"})
        before = [(e.seq, e.type, e.payload) for e in s.get_events(sid, limit=100)]
        env.reopen()
        s2 = env.store
        assert [(e.seq, e.type, e.payload) for e in s2.get_events(sid, limit=100)] == before
        assert s2.get_active_session_id() == sid
        assert s2.emit_event(sid, "user_message", {"text": "继续"}) == len(before) + 1
        assert s2._conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert s2._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_h_concurrent_emit():
    with _Env() as env:
        s = env.store
        sid = s.create_session("h").id
        n_threads, per = 8, 25
        errs = []

        def work(k):
            try:
                for i in range(per):
                    s.emit_event(sid, "tool_result", {"text": f"{k}-{i}"})
            except Exception as e:
                errs.append(e)

        ts = [threading.Thread(target=work, args=(k,)) for k in range(n_threads)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert not errs, errs
        seqs = [e.seq for e in s.get_events(sid, limit=10000)]
        assert seqs == list(range(1, n_threads * per + 2)), (len(seqs), seqs[:5])


def test_i_delete_session():
    with _Env() as env:
        s = env.store
        a, b = s.create_session("a").id, s.create_session("b").id
        s.emit_event(b, "user_message", {"text": "keep"})
        s.save_turn_state(b, "u", [])
        s.set_active_session(b)
        assert s.delete_session(b) is True
        assert s.get_session(b) is None and s.get_events(b) == [] and s.load_turn_state(b) is None
        assert s.search_events("keep") == []
        assert s.get_active_session_id() == a                                          # 指针回落
        assert s.delete_session("nope") is False
        s.delete_session(a)
        new = s.get_active_session_id()                                                # 全删后自动建默认会话
        assert new not in (a, b) and s.count_sessions() == 1


def _mk_msgs(big=False):
    task_call = {"name": "task", "args": {"subagent_type": "MapAgent", "description": "查济南天气"}, "id": "c_task"}
    tool_call = {"name": "get_current_time", "args": {}, "id": "c_time"}
    todo_call = {"name": "write_todos", "args": {"todos": []}, "id": "c_todo"}
    sess_call = {"name": "search_session_history", "args": {"query": "x"}, "id": "c_sess"}
    long_text = "很长" * 15000 if big else "晴 25度"
    return [
        HumanMessage(content="<long_term_memory>\n我住济南\n</long_term_memory>\n查天气", id="h1"),
        AIMessage(content="", tool_calls=[task_call, tool_call, todo_call, sess_call], id="ai1"),
        ToolMessage(content=long_text + " data:image/png;base64,AAAA", tool_call_id="c_task", name="task", id="t1"),
        ToolMessage(content="2026-11-07", tool_call_id="c_time", name="get_current_time", id="t2"),
        ToolMessage(content="ok", tool_call_id="c_todo", name="write_todos", id="t3"),
        ToolMessage(content="回溯结果", tool_call_id="c_sess", name="search_session_history", id="t4"),
        AIMessage(content="济南今天晴", id="ai2"),
    ]


def test_j_mirror_messages():
    with _Env() as env:
        s = env.store
        sid = s.create_session("j").id
        base = s.count_events(sid)
        msgs = _mk_msgs()
        n = session_log.mirror_messages(s, sid, msgs)
        evs = s.get_events(sid, start_seq=base + 1, limit=100)
        types = [e.type for e in evs]
        assert n == len(evs) == 3 and types == ["tool_call", "expert_report", "tool_result"], types
        tc = evs[0].payload["calls"]
        assert [c["name"] for c in tc] == ["get_current_time"], tc                       # task/write_todos/会话工具被跳过
        er = evs[1].payload
        assert er["expert"] == "MapAgent" and er["task"] == "查济南天气" and "晴 25度" in er["text"], er
        assert "[图片已省略]" in er["text"] and "base64" not in er["text"]                # data URI 被清洗
        assert evs[2].payload["name"] == "get_current_time" and evs[2].payload["text"] == "2026-11-07"
        blob = json.dumps([e.payload for e in evs], ensure_ascii=False)
        assert "long_term_memory" not in blob and "济南今天晴" not in blob and "回溯结果" not in blob
        # 整体重放幂等（含摘要压缩后只剩尾部消息的情况）
        assert session_log.mirror_messages(s, sid, msgs) == 0
        assert session_log.mirror_messages(s, sid, msgs[2:]) == 0
        assert s.count_events(sid) == base + 3
        # 超长截断
        sid2 = s.create_session("j2").id
        session_log.mirror_messages(s, sid2, _mk_msgs(big=True))
        rep = [e for e in s.get_events(sid2, limit=100) if e.type == "expert_report"][0].payload
        assert rep.get("truncated") is True and len(rep["text"]) < config.SESSION_EVENT_TEXT_MAX_CHARS + 100, len(rep["text"])
        # tool_call 缺 message.id 时退化为 tool_call id 拼接，仍幂等
        m = AIMessage(content="", tool_calls=[{"name": "foo", "args": {"a": 1}, "id": "zzz"}])
        m.id = None
        assert session_log.mirror_messages(s, sid, [m]) == 1 and session_log.mirror_messages(s, sid, [m]) == 0


def _cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def test_k_session_tools():
    with _Env() as env:
        s = env.store
        a = s.create_session("天气会话").id
        b = s.create_session("菜谱会话").id                                             # 活动会话 = b
        for i in range(25):
            s.emit_event(a, "user_message", {"text": f"A消息{i} 济南天气"})
        s.emit_event(b, "user_message", {"text": "B消息 红烧肉怎么做"})
        s.emit_event(b, "expert_report", {"expert": "KnowledgeAgent", "task": "红烧肉", "text": "步骤" * 2000})

        names = [t.name for t in session_tools.TOOLS]
        assert names == ["get_session_events", "search_session_history"], names
        for t in session_tools.TOOLS:
            assert "config" not in t.get_input_schema().model_json_schema().get("properties", {}), "config 不应出现在工具 schema"
            assert t.description.strip()

        g = session_tools.get_session_events
        # 会话解析优先级：参数 > RunnableConfig.thread_id > 活动会话
        assert "A消息" in g.invoke({"session_id": a[:8]}, config=_cfg(b))
        assert "A消息" in g.invoke({}, config=_cfg(a))
        assert "B消息" in g.invoke({})                                                 # 无 config → 活动会话 b
        # start_seq<=0 取最新 N 条；>=1 向后切片
        out = g.invoke({"start_seq": 0, "limit": 3, "session_id": a})
        assert "A消息24" in out and "A消息22" in out and "A消息21" not in out, out
        out = g.invoke({"start_seq": 3, "limit": 2, "session_id": a})
        assert "#3 " in out and "#4 " in out and "#5 " not in out and "start_seq=5" in out, out
        assert "A消息1 " in out and "A消息2 " in out and "A消息0" not in out, out      # seq1=session_created, seq3=A消息1
        # limit 上限 20
        out = g.invoke({"start_seq": 1, "limit": 999, "session_id": a})
        assert out.count("\n#") + (1 if "\n#" not in out and "#1 " in out else 0) <= 20, out.count("\n#")
        # types 过滤
        out = g.invoke({"types": "expert_report", "session_id": b})
        assert "expert_report/KnowledgeAgent" in out and "B消息" not in out
        # 单条截断 vs limit=1 放宽
        out = g.invoke({"types": "expert_report", "limit": 5, "session_id": b})
        assert "start_seq=" in out and len(out) < config.SESSION_TOOL_OUTPUT_MAX_CHARS + 300
        seq = [e for e in s.get_events(b, limit=100) if e.type == "expert_report"][0].seq
        one = g.invoke({"start_seq": seq, "limit": 1, "session_id": b})
        short = g.invoke({"start_seq": seq, "limit": 2, "session_id": b})
        assert len(one) > len(short) and len(one) <= config.SESSION_TOOL_OUTPUT_MAX_CHARS + 300, (len(one), len(short))
        # 总输出上限
        for i in range(20):
            s.emit_event(a, "expert_report", {"expert": "X", "task": "t", "text": "数据" * 600})
        out = g.invoke({"limit": 20, "session_id": a})
        assert len(out) <= config.SESSION_TOOL_OUTPUT_MAX_CHARS + 400, len(out)
        # 异常转文本
        assert "找不到会话" in g.invoke({"session_id": "nope"})
        s.close()                                                                       # 底层故障：不抛，转文本
        out = g.invoke({"session_id": a})
        assert "失败" in out, out


def test_k2_search_tool():
    with _Env() as env:
        s = env.store
        a = s.create_session("天气会话").id
        b = s.create_session("菜谱会话").id
        s.emit_event(a, "expert_report", {"expert": "MapAgent", "task": "天气", "text": "济南今天晴，气温 25 度，适合出门"})
        s.emit_event(b, "user_message", {"text": "红烧肉怎么做"})
        f = session_tools.search_session_history
        out = f.invoke({"query": "气温"})
        assert a[:8] in out and "天气会话" in out and "MapAgent" in out and "气温" in out, out
        assert b[:8] not in out
        assert "红烧肉" in f.invoke({"query": "红烧肉"})                                 # 默认跨会话
        assert "没有找到" in f.invoke({"query": "红烧肉", "session_id": a})                # 限定会话
        assert "红烧肉" in f.invoke({"query": "红烧肉", "session_id": b[:8]})
        assert "没有找到" in f.invoke({"query": "火星基地"}) and "换个关键词" in f.invoke({"query": "火星基地"})
        ov = f.invoke({"query": ""})                                                     # 空 query → 概览
        assert "天气会话" in ov and "菜谱会话" in ov and "轮" in ov, ov
        out = f.invoke({"query": "济南 晴", "limit": 99})
        assert a[:8] in out
        assert "找不到会话" in f.invoke({"query": "x", "session_id": "nope"})


SCENARIOS = [
    ("A 会话 / 活动指针 / 标题 / 前缀解析", test_a_sessions_basics),
    ("B 事件追加与读取", test_b_events),
    ("C 幂等 key", test_c_idempotent_key),
    ("D 待处理游标", test_d_pending_cursor),
    ("E turn_state 往返", test_e_turn_state),
    ("F 搜索", test_f_search),
    ("G 关闭重开数据完整", test_g_reopen_integrity),
    ("H 多线程并发 emit", test_h_concurrent_emit),
    ("I delete_session", test_i_delete_session),
    ("J mirror_messages", test_j_mirror_messages),
    ("K1 get_session_events 工具", test_k_session_tools),
    ("K2 search_session_history 工具", test_k2_search_tool),
]


def main():
    print("=" * 70)
    print("开始运行会话层离线冒烟测试...")
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
