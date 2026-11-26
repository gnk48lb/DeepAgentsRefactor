"""
tests/live_qq_push.py

QQ C2C「主动消息」（不带 msg_id）真机探针：只由用户运行，Claude Code / CI 不得运行。
定时任务到点后要靠这条路径推送，这里把“阶段 1 读文档得到、没有在真机上验证过”的几件事量出来：
  ① 无 msg_id 的纯文本能不能发出去；② 相同内容短间隔连发是否被去重（40054005）；
  ③ 连发 8 条会不会限频，限频多久恢复，旧文档里“每月只能主动发 4 条”是否成立；
  ④ 含 https://example.com 与裸域名 example.com 的文本是否被拒（40054010）；
  ⑤ 单条文本最大可发长度（40054007）；⑥ 用户在 QQ 里关闭“允许主动发送消息”之后的错误码。
量出的值交给你决定 SCHEDULER_PUSH_MIN_INTERVAL_SECONDS、SCHEDULER_PUSH_MAX_CHARS、reminder 是否带时间后缀，
以及 plugins/chat.py 错误码表里哪些码归 permanent。

用法：
    uv run python tests/live_qq_push.py --live --openid <目标用户 openid> [--tag 名字] [--only 1,2,3]
                                        [--last-interaction "YYYY-MM-DD HH:MM"] [--port 18765]

  没有 --live 只打印用法并以退出码 0 退出（此时不导入 nonebot、不联网、不启动任何服务）；--help 同样完全离线。
  默认运行 ①～⑤；⑥ 只在显式 --only 6 时运行（它要求你先在 QQ 里关闭“允许主动发送消息”）。

做法：脚本在本进程里起一个只含 QQ 适配器的最小 NoneBot（不加载 plugins/，换一个不冲突的端口，日志压到 INFO），
等机器人连上网关后，在 driver.on_bot_connect 里直接调用 Bot.send_to_c2c（与生产的主动推送同一条路径），
跑完全部用例、刷新输出后用 os._exit 退出（不走 NoneBot 的优雅关闭，免得挂住）。
配置照抄 bot.py：从仓库根目录的 .env 读取 QQ_BOTS，所以脚本会先切换到仓库根目录。

每个用例打印 [Case N] 描述 与 PASS / FAIL，量出来的值用一行 `MEASURE key=value` 打印，结尾集中打印
`=== MEASUREMENTS ===` 块（整块复制给我）。任何用例 FAIL 不中断其余用例，最后汇总并以退出码 1 结束。
PASS 表示“这个用例按设计跑完并量到了值”（例如 ② 里第二条被去重也算 PASS，结果记在 MEASURE 里）；
FAIL 表示前提不满足或出了意外（例如 ① 发不出去、⑥ 居然发出去了）。
绝不打印令牌 / AppSecret / openid 全文：openid 只显示前 8 位，异常信息里的 openid 也会被替换。
"""

from app import bootstrap
bootstrap.apply()

import argparse
import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PORT = 18765
LEN_RESOLUTION = 50          # ⑤ 里二分查找最大可发长度的精度（字符）
PACE_SECONDS = 1.2           # ④⑤⑥ 里相邻两条之间的间隔，避免把别的用例的限频带进来
RATE_LIMIT_WAIT_SECONDS = 65  # ④⑤ 遇到限频时的等待；③ 里“等 65 秒看是否恢复”也用它
CONNECT_TIMEOUT_SECONDS = 60

USAGE = """\
QQ 主动消息真机探针（不带 --live 什么都不做）：
    uv run python tests/live_qq_push.py --live --openid <目标用户 openid> [--tag 名字] [--only 1,2,3] \\
        [--last-interaction "YYYY-MM-DD HH:MM"] [--port 18765]

运行前必读：
  1. 先停掉 bot.py（同一个 AppID 同时连两个网关会互相顶掉）。脚本运行期间也不要在 QQ 里给机器人发消息。
  2. 先在 QQ 里给机器人发一条消息（让它认得你；openid 只在当前 AppID 范围内有效，
     owner 的 openid 就是 .env 里 OWNER_USER_IDS 的某一项）。
  3. 用例 ① ② ③ ④ ⑤ 默认全跑，会给目标用户的 QQ 发出约 20～25 条测试消息（含 8 条连发和最长 4000 字的文本）。
     用例 ③ 触发限频时会多等 65 秒，④ ⑤ 遇到限频会自己等 65 秒再重试一次，整套大约 3～6 分钟。
  4. 用例 ⑥ 需要你先在 QQ 里手动关闭对机器人的“允许主动发送消息”，再单独运行：
         uv run python tests/live_qq_push.py --live --openid <openid> --tag off --only 6
     ⑥ 不在默认集合里，不带 --only 6 不会运行。测完记得把开关打开。
  5. 判断是否存在“互动窗口”（用户多久没跟机器人说话就发不出主动消息）：在刚聊完 / 隔 1 天 / 隔 7 天后
     各运行一次，用不同的 --tag，并用 --last-interaction 注明上次互动时间（只作备注写进输出）。
     隔 1 天、隔 7 天那两次只要 ① 就够了，建议加 --only 1，少打扰用户。
  6. 把结尾 `=== MEASUREMENTS ===` 到 `=== END MEASUREMENTS ===` 之间的整块原样贴回给我。
"""

measurements: dict[str, str] = {}
results: list[tuple[int, str, bool, str]] = []

_openid = ""
_tag = "run"
_seq = 0
_sent_ok = 0                 # 本次运行里成功发出的主动消息条数（含进入审核的）


def measure(key: str, value) -> None:
    measurements[key] = str(value)
    print(f"MEASURE {key}={value}", flush=True)


def mask_openid(value: str) -> str:
    return value[:8] + "…" if value else ""


def scrub(text) -> str:
    """异常信息里可能带 openid（接口路径里有它）：替换成前 8 位，再压成单行。"""
    s = str(text)
    if _openid:
        s = s.replace(_openid, mask_openid(_openid))
    return " ".join(s.split())


# 错误码分类：来源 = 平台文档，全部“未实测”；T0 实测后由我回传结果，再修订 plugins/chat.py 里的正式表。
_CODE_CLASS = {
    40034105: "permanent",     # 无主动消息权限（平台文档，未实测）
    40054013: "permanent",     # 用户拒绝接收（平台文档，未实测）
    40034100: "transient",     # 频率限制（平台文档，未实测）
    50055001: "transient",     # 要求重试（平台文档，未实测）
    40054016: "transient",     # 机器人离线（平台文档，未实测）
    40054005: "dedupe",        # 内容重复（平台文档，未实测）
    40054007: "too_long",      # 内容过长（平台文档，未实测）
    40054010: "url_rejected",  # 含不允许的链接（平台文档，未实测）
}


@dataclass
class Outcome:
    ok: bool
    kind: str                  # ok | audit | action_failed | rate_limit | unauthorized | network | api_unavailable | other
    status: int | None = None
    code: int | None = None
    message: str = ""
    trace_id: str = ""
    exc: str = ""
    elapsed: float = 0.0
    retried: bool = False      # 遇到限频，等 65 秒后重发过一次

    @property
    def klass(self) -> str:
        if self.ok:
            return "ok"
        if self.kind in ("rate_limit", "network", "api_unavailable"):
            return "transient"
        if self.kind == "unauthorized":
            return "unauthorized"
        if self.kind == "action_failed":
            return _CODE_CLASS.get(self.code, "unknown")   # 表外的码：正式实现里按 transient 处理
        return "unknown"

    @property
    def is_rate_limited(self) -> bool:
        return self.kind == "rate_limit" or self.code == 40034100

    def brief(self) -> str:
        if self.ok:
            tail = "（进入审核，未立即送达）" if self.kind == "audit" else ""
            return f"ok{tail} {self.elapsed:.2f}s" + (" [等待后重发]" if self.retried else "")
        parts = [f"FAILED {self.exc or self.kind}"]
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.code is not None:
            parts.append(f"code={self.code}")
        parts.append(f"class={self.klass}")
        if self.message:
            parts.append(f"msg={self.message[:100]!r}")
        if self.trace_id:
            parts.append(f"trace={self.trace_id}")
        if self.retried:
            parts.append("[等待后重发]")
        return " ".join(parts)

    def short(self) -> str:
        """MEASURE 里用的一个词：ok 或 code=…/kind。"""
        if self.ok:
            return "ok"
        return f"code={self.code}" if self.code is not None else self.kind


async def send(bot, text: str) -> Outcome:
    """发一条无 msg_id 的纯文本主动消息，不抛异常，返回结构化结果。"""
    global _sent_ok
    from nonebot.adapters.qq import exception as qq_exc

    t0 = time.perf_counter()
    try:
        await bot.send_to_c2c(openid=_openid, message=text)
    except qq_exc.AuditException:
        _sent_ok += 1
        return Outcome(True, "audit", elapsed=time.perf_counter() - t0)
    except qq_exc.RateLimitException as e:
        return _failed("rate_limit", e, t0)
    except qq_exc.UnauthorizedException as e:
        return _failed("unauthorized", e, t0)
    except qq_exc.ActionFailed as e:
        return _failed("action_failed", e, t0)
    except qq_exc.NetworkError as e:
        return Outcome(False, "network", message=scrub(getattr(e, "msg", "") or ""),
                       exc=type(e).__name__, elapsed=time.perf_counter() - t0)
    except qq_exc.ApiNotAvailable as e:
        return Outcome(False, "api_unavailable", exc=type(e).__name__, message=scrub(e),
                       elapsed=time.perf_counter() - t0)
    except Exception as e:
        return Outcome(False, "other", exc=type(e).__name__, message=scrub(e),
                       elapsed=time.perf_counter() - t0)
    _sent_ok += 1
    return Outcome(True, "ok", elapsed=time.perf_counter() - t0)


def _failed(kind: str, e, t0: float) -> Outcome:
    trace = ""
    try:
        trace = scrub(e.trace_id or "")
    except Exception:
        pass
    return Outcome(False, kind, status=e.status_code, code=e.code, message=scrub(e.message or ""),
                   trace_id=trace, exc=type(e).__name__, elapsed=time.perf_counter() - t0)


async def send_patient(bot, text: str) -> Outcome:
    """④⑤⑥ 用：遇到限频就等 65 秒后重发一次（被限频拒收的消息没有送达，重发不会触发去重）。"""
    o = await send(bot, text)
    if not o.ok and o.is_rate_limited:
        print(f"  遇到限频（{o.short()}），等待 {RATE_LIMIT_WAIT_SECONDS} 秒后重发一次……", flush=True)
        await asyncio.sleep(RATE_LIMIT_WAIT_SECONDS)
        o = await send(bot, text)
        o.retried = True
    return o


def uniq(label: str) -> str:
    """每条测试消息内容都不同（序号 + 时间），避免互相触发去重。不含句点，免得被当成域名。"""
    global _seq
    _seq += 1
    return f"[T0 {_tag}#{_seq} {time.strftime('%H:%M:%S')}] {label}"


def text_of_length(n: int, label: str) -> str:
    """恰好 n 个字符的中文纯文本（生产里推送的也是中文）；开头带编号，保证内容唯一。"""
    global _seq
    _seq += 1
    head = f"[T0 {_tag}#{_seq} len={n}] {label} "
    body = "测试长度探针" * (n // 6 + 1)
    return (head + body)[:n]


CASES: dict[int, tuple[str, object]] = {}


def case(number: int, description: str):
    """装饰器：登记一个用例，运行时打印 PASS / FAIL，失败不中断其余用例。"""
    def deco(fn):
        async def run(bot):
            print(f"\n[Case {number}] {description}", flush=True)
            try:
                await fn(bot)
                print("  PASS", flush=True)
                results.append((number, description, True, ""))
            except AssertionError as e:
                print(f"  FAIL: {e}", flush=True)
                results.append((number, description, False, str(e)))
            except Exception as e:
                detail = f"{type(e).__name__}: {scrub(e)[:300]}"
                print(f"  FAIL: {detail}", flush=True)
                results.append((number, description, False, detail))
        CASES[number] = (description, run)
        return run
    return deco


@case(1, "无 msg_id 的主动文本能否发出")
async def c1(bot):
    o = await send(bot, uniq("主动消息探针①：无 msg_id 的纯文本"))
    print(f"  {o.brief()}")
    measure("c1_result", o.short())
    measure("c1_class", o.klass)
    if not o.ok:
        measure("c1_code", o.code)
        measure("c1_status", o.status)
        measure("c1_message", o.message[:100])
        if o.klass == "permanent":
            print("  ⚠️ 这是“无权限 / 拒收”类失败：主动推送的前提不成立，请把整块 MEASUREMENTS 贴回给我，方案要改。")
    assert o.ok, o.brief()


@case(2, "相同内容间隔 1 秒连发两次，第二次是否被去重（40054005）")
async def c2(bot):
    text = uniq("去重探针②：这条内容会连发两次")
    first = await send(bot, text)
    print(f"  第 1 次: {first.brief()}")
    await asyncio.sleep(1)
    second = await send(bot, text)
    print(f"  第 2 次: {second.brief()}")
    measure("c2_first", first.short())
    measure("c2_second", second.short())
    measure("c2_second_class", second.klass)
    measure("c2_deduped", second.code == 40054005)
    if second.ok:
        print("  结论：1 秒间隔的相同内容没有被去重。")
    elif second.code == 40054005:
        print("  结论：被去重了（40054005）。reminder 的推送正文将永远带 [MM-DD HH:MM] 后缀。")
    else:
        print("  结论：第二次失败，但不是去重码；见上面的 code。")
    assert first.ok, f"第 1 次就失败了：{first.brief()}"


@case(3, "连续发 8 条不同内容（无间隔）；再等 65 秒看是否恢复；核对“每月只能主动发 4 条”")
async def c3(bot):
    sent_before = _sent_ok
    ok_count = 0
    first_fail: tuple[int, Outcome] | None = None
    t0 = time.perf_counter()
    for i in range(1, 9):
        o = await send(bot, uniq(f"限频探针③：第 {i}/8 条"))
        print(f"  第 {i} 条: {o.brief()}")
        if o.ok:
            ok_count += 1
        elif first_fail is None:
            first_fail = (i, o)
    burst = time.perf_counter() - t0
    measure("c3_burst_ok", f"{ok_count}/8")
    measure("c3_burst_seconds", f"{burst:.1f}")
    measure("c3_first_fail_index", first_fail[0] if first_fail else "none")
    measure("c3_first_fail_code", first_fail[1].short() if first_fail else "none")
    measure("c3_first_fail_class", first_fail[1].klass if first_fail else "none")
    measure("c3_proactive_ok_in_run_before_case", sent_before)
    measure("c3_proactive_ok_in_run_after_burst", _sent_ok)
    if first_fail:
        print(f"  等待 {RATE_LIMIT_WAIT_SECONDS} 秒后再发 1 条，看限频是否恢复……", flush=True)
        await asyncio.sleep(RATE_LIMIT_WAIT_SECONDS)
        after = await send(bot, uniq("限频探针③：等待 65 秒后的 1 条"))
        print(f"  等待后: {after.brief()}")
        measure("c3_after_65s", after.short())
        measure("c3_recovered", after.ok)
    else:
        print("  8 条全部成功，不触发限频，跳过 65 秒等待。")
        measure("c3_after_65s", "skipped(8条全部成功)")
        measure("c3_recovered", "n/a")
    if _sent_ok > 4:
        print(f"  本次运行已成功主动发送 {_sent_ok} 条（>4）：旧文档里“每月只能主动发 4 条”在这个机器人 / 用户上不成立（至少不是按条数硬限）。")
    else:
        print(f"  本次运行成功主动发送 {_sent_ok} 条（≤4）：还不能推翻“每月 4 条”的说法，要看上面的失败码。")
    assert ok_count > 0, "8 条一条都没发出去：" + (first_fail[1].brief() if first_fail else "")


@case(4, "含 https://example.com 与裸域名 example.com 的文本是否被拒（40054010）")
async def c4(bot):
    with_url = await send_patient(bot, uniq("链接探针④a：含链接 https://example.com 的文本"))
    print(f"  含 https 链接: {with_url.brief()}")
    await asyncio.sleep(PACE_SECONDS)
    bare = await send_patient(bot, uniq("链接探针④b：含裸域名 example.com 的文本"))
    print(f"  含裸域名:      {bare.brief()}")
    measure("c4_https_url", with_url.short())
    measure("c4_https_url_rejected", with_url.code == 40054010)
    measure("c4_bare_domain", bare.short())
    measure("c4_bare_domain_rejected", bare.code == 40054010)
    print("  无论结果如何，生产里都保留 _strip_urls（保守）。")
    for name, o in (("含链接", with_url), ("含裸域名", bare)):
        assert o.ok or o.klass == "url_rejected", f"{name}的结果既不是成功也不是 40054010：{o.brief()}"


@case(5, "文本长度 500 / 1000 / 2000 / 4000：找出最大可发长度（40054007）")
async def c5(bot):
    last_ok = 0
    first_bad: int | None = None
    bad_outcome: Outcome | None = None
    for n in (500, 1000, 2000, 4000):
        o = await send_patient(bot, text_of_length(n, "长度探针⑤"))
        print(f"  {n} 字: {o.brief()}")
        measure(f"c5_len_{n}", o.short())
        await asyncio.sleep(PACE_SECONDS)
        if o.ok:
            last_ok = n
            continue
        assert o.kind == "action_failed", f"{n} 字的失败不是平台拒收类（{o.kind}），没法判断长度上限：{o.brief()}"
        first_bad, bad_outcome = n, o
        break
    if first_bad is None:
        measure("c5_max_ok_len", ">=4000")
        measure("c5_min_bad_len", "none")
        measure("c5_suggested_push_max_chars", min(1500, int(0.8 * 4000)))
        print("  4000 字也能发出去：最大可发长度 ≥ 4000，SCHEDULER_PUSH_MAX_CHARS 保持 1500。")
        return
    assert bad_outcome is not None
    if bad_outcome.code != 40054007:
        print(f"  ⚠️ 超长被拒的错误码是 {bad_outcome.code}，不是文档里的 40054007；正式错误码表要按实测改。")
    lo, hi = last_ok, first_bad          # lo 已确认可发（0 表示 500 字都不行），hi 已确认被拒
    while hi - lo > LEN_RESOLUTION:
        mid = (lo + hi) // 2
        o = await send_patient(bot, text_of_length(mid, "长度探针⑤二分"))
        print(f"  二分 {mid} 字: {o.brief()}")
        await asyncio.sleep(PACE_SECONDS)
        if o.ok:
            lo = mid
        else:
            assert o.kind == "action_failed", f"二分到 {mid} 字时出现平台拒收以外的失败：{o.brief()}"
            hi = mid
    measure("c5_max_ok_len", lo)
    measure("c5_min_bad_len", hi)
    measure("c5_resolution", LEN_RESOLUTION)
    measure("c5_bad_code", bad_outcome.short())
    suggested = min(1500, int(0.8 * lo))
    measure("c5_suggested_push_max_chars", suggested)
    print(f"  最大可发长度约 {lo}（精度 ±{LEN_RESOLUTION}），按决策表 SCHEDULER_PUSH_MAX_CHARS 取 min(1500, floor(0.8×{lo})) = {suggested}。")


@case(6, "用户关闭“允许主动发送消息”之后再发：记录错误码")
async def c6(bot):
    o = await send(bot, uniq("关闭主动消息后的探针⑥"))
    print(f"  {o.brief()}")
    measure("c6_result", o.short())
    measure("c6_status", o.status)
    measure("c6_code", o.code)
    measure("c6_message", o.message[:100])
    measure("c6_class", o.klass)
    if not o.ok and o.klass != "permanent":
        print("  这个码不在文档给的 permanent 组（40034105 / 40054013）里，T2 会把它按实测加进 permanent 组并标注“已实测”。")
    assert not o.ok, "发送成功了：请先在 QQ 里关闭对机器人的“允许主动发送消息”，再用 --tag off --only 6 运行。"


def finish() -> int:
    print("\n=== MEASUREMENTS ===")
    for key, value in measurements.items():
        print(f"{key}={value}")
    print("=== END MEASUREMENTS ===")
    failed = [(n, d, why) for n, d, ok_, why in results if not ok_]
    print(f"\n结果：{len(results) - len(failed)}/{len(results)} 个用例通过。")
    for n, d, why in failed:
        print(f"  FAIL [Case {n}] {d}: {why}")
    sys.stdout.flush()
    sys.stderr.flush()
    return 1 if failed else 0


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except Exception:
        return "unknown"


def _hours_since(text: str) -> str:
    try:
        then = time.mktime(time.strptime(text, "%Y-%m-%d %H:%M"))
    except Exception:
        return "unparsed"
    return f"{(time.time() - then) / 3600:.1f}"


def _tag_arg(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}", value):
        raise argparse.ArgumentTypeError("--tag 只能由字母、数字、下划线、连字符组成，长度 1~16")
    return value


def _only_arg(value: str) -> list[int]:
    try:
        numbers = sorted({int(p) for p in value.split(",") if p.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError("--only 的格式是用逗号分隔的用例编号，如 1,2,3") from None
    if not numbers or any(n not in CASES for n in numbers):
        raise argparse.ArgumentTypeError(f"--only 的编号必须在 {sorted(CASES)} 里")
    return numbers


def run_live(args) -> int:
    global _openid, _tag
    _openid = args.openid
    _tag = args.tag
    selected = args.only or [n for n in sorted(CASES) if n != 6]

    os.chdir(ROOT)                      # nonebot.init() 从当前目录读 .env，和 bot.py 一样
    import nonebot
    from nonebot.adapters.qq import Adapter as QQAdapter
    from nonebot.adapters.qq import Bot as QQBot

    nonebot.init(host="127.0.0.1", port=args.port, log_level="INFO")
    driver = nonebot.get_driver()
    driver.register_adapter(QQAdapter)

    state = {"connected": False, "watchdog": None}

    async def watchdog():
        await asyncio.sleep(CONNECT_TIMEOUT_SECONDS)
        if not state["connected"]:
            print(f"\n❌ {CONNECT_TIMEOUT_SECONDS} 秒内机器人没有连上 QQ 网关：检查 .env 里的 QQ_BOTS、网络，"
                  "以及 bot.py 是否还在运行。", flush=True)
            os._exit(2)

    @driver.on_startup
    async def _start_watchdog():
        state["watchdog"] = asyncio.create_task(watchdog())   # 留一个引用，防止任务被回收

    @driver.on_bot_connect
    async def _on_connect(bot: QQBot):
        if state["connected"]:
            return
        state["connected"] = True
        code = 1
        try:
            print("\n机器人已连上网关，开始跑用例。", flush=True)
            measure("tag", args.tag)
            measure("last_interaction", args.last_interaction or "unset")
            if args.last_interaction:
                measure("hours_since_last_interaction", _hours_since(args.last_interaction))
            measure("run_started_local", time.strftime("%Y-%m-%d %H:%M:%S"))
            measure("cases", ",".join(str(n) for n in selected))
            measure("openid_prefix", mask_openid(_openid))
            measure("bot_id_suffix", "…" + str(bot.self_id)[-4:])
            measure("qq_is_sandbox", getattr(getattr(bot.adapter, "qq_config", None), "qq_is_sandbox", "unknown"))
            measure("nonebot_version", _version("nonebot2"))
            measure("qq_adapter_version", _version("nonebot-adapter-qq"))
            for n in selected:
                await CASES[n][1](bot)
            measure("proactive_ok_in_run", _sent_ok)
        except BaseException as e:    # 含取消：无论如何都要把已量到的值打印出来再退出
            print(f"\n❌ 探针中断: {type(e).__name__}: {scrub(e)[:300]}", flush=True)
        finally:
            code = finish()
            os._exit(code)

    try:
        nonebot.run()
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="QQ C2C 主动消息真机探针（只由用户运行）")
    parser.add_argument("--live", action="store_true", help="真的连接 QQ 并给目标用户发测试消息")
    parser.add_argument("--openid", help="目标用户的 openid（--live 时必填；只显示前 8 位）")
    parser.add_argument("--tag", type=_tag_arg, default="run", help="本次运行的名字，写进输出，用来区分多次运行")
    parser.add_argument("--only", type=_only_arg, help="只运行这些用例，如 1,2,3；⑥ 只有显式写 6 才会运行")
    parser.add_argument("--last-interaction", metavar='"YYYY-MM-DD HH:MM"',
                        help="目标用户上一次和机器人互动的时间（本地时间），仅作备注写进输出")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"最小 NoneBot 的监听端口（默认 {DEFAULT_PORT}）")
    args = parser.parse_args(argv)
    if not args.live:
        print(USAGE)
        return 0
    if not args.openid or not re.fullmatch(r"[A-Za-z0-9]{8,64}", args.openid):
        print("❌ --live 需要 --openid <目标用户 openid>（8～64 位字母数字）。")
        return 2
    if args.last_interaction:
        args.last_interaction = " ".join(args.last_interaction.split())[:40]
    print(f"开始 QQ 主动消息探针：tag={args.tag}，目标 openid={mask_openid(args.openid)}（会给该用户发测试消息）……", flush=True)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
