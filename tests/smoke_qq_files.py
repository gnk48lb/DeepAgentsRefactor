"""
tests/smoke_qq_files.py

QQ 文件接收（owner + member）离线冒烟测试：不需要 API Key / Docker / 网络 / Milvus / 真实 Modal。
  1~5   配置、文件名清洗、L0 纯函数、下载地址校验、附件提取
  6~8   下载（httpx.MockTransport）、owner 落盘、member 直传 Modal（FakeModal）
  9     modal_sandbox 新函数
  10~11 在途闸门、未消费文件清单与注入
  12~13 chat_service 集成、QQ 插件
  14~17 session_tools、清理、提示词与技能、架构
"""
import asyncio
import json
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app import config
from app.utils import files as F
from app.utils.files import IncomingFile

ROOT = Path(__file__).resolve().parents[1]


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"期望 {exc.__name__}，实际 {type(e).__name__}: {e}")
    raise AssertionError(f"期望抛 {exc.__name__}，但没有抛：{fn.__name__}{args!r}")


# ═════════════════════════════════════════════════════════════════════
# 1 配置默认值与夹紧
# ═════════════════════════════════════════════════════════════════════
_CONFIG_KEYS = (
    "QQ_FILE_ENABLED", "QQ_FILE_MEMBER_ENABLED", "QQ_FILE_MAX_MB_OWNER", "QQ_FILE_MAX_MB_MEMBER",
    "QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS", "QQ_FILE_MAX_PER_MESSAGE", "QQ_FILE_MAX_PER_SESSION",
    "QQ_FILE_MAX_CONCURRENT", "QQ_INBOX_TTL_DAYS", "QQ_FILE_INJECT_MAX_ITEMS", "QQ_FILE_SAFE_NAMES",
)


def _config_with_env(**overrides):
    """在干净的子进程里按给定环境变量导入 app.config，返回这批配置项的值（config 在 import 时解析，只能这样测）。"""
    env = {k: v for k, v in os.environ.items() if k not in _CONFIG_KEYS}
    env.update({k: str(v) for k, v in overrides.items()})
    code = (
        "import json\nfrom app import config\n"
        f"print(json.dumps({{k: getattr(config, k) for k in {list(_CONFIG_KEYS)!r}}}))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr[-500:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_1_config_defaults_and_clamping():
    d = _config_with_env()
    assert d == {
        "QQ_FILE_ENABLED": True, "QQ_FILE_MEMBER_ENABLED": True, "QQ_FILE_MAX_MB_OWNER": 20,
        "QQ_FILE_MAX_MB_MEMBER": 10, "QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS": 30, "QQ_FILE_MAX_PER_MESSAGE": 5,
        "QQ_FILE_MAX_PER_SESSION": 20, "QQ_FILE_MAX_CONCURRENT": 3, "QQ_INBOX_TTL_DAYS": 7,
        "QQ_FILE_INJECT_MAX_ITEMS": 10, "QQ_FILE_SAFE_NAMES": True,
    }, d

    # 越界夹紧
    d = _config_with_env(
        QQ_FILE_MAX_MB_OWNER=0, QQ_FILE_MAX_MB_MEMBER=999, QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS=1,
        QQ_FILE_MAX_PER_MESSAGE=99, QQ_FILE_MAX_PER_SESSION=0, QQ_FILE_MAX_CONCURRENT=99,
        QQ_INBOX_TTL_DAYS=-3, QQ_FILE_INJECT_MAX_ITEMS=999,
    )
    assert (d["QQ_FILE_MAX_MB_OWNER"], d["QQ_FILE_MAX_MB_MEMBER"]) == (1, 50), d
    assert d["QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS"] == 5, d
    assert (d["QQ_FILE_MAX_PER_MESSAGE"], d["QQ_FILE_MAX_PER_SESSION"]) == (20, 1), d
    assert d["QQ_FILE_MAX_CONCURRENT"] == 10 and d["QQ_INBOX_TTL_DAYS"] == 0 and d["QQ_FILE_INJECT_MAX_ITEMS"] == 50, d
    d = _config_with_env(QQ_FILE_MAX_MB_OWNER=1000, QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS=99999)
    assert d["QQ_FILE_MAX_MB_OWNER"] == 100 and d["QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS"] == 300, d

    # 垃圾值回落默认
    d = _config_with_env(
        QQ_FILE_MAX_MB_OWNER="abc", QQ_FILE_MAX_MB_MEMBER="1.5", QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS="",
        QQ_FILE_MAX_PER_MESSAGE="x", QQ_FILE_MAX_PER_SESSION="!", QQ_FILE_MAX_CONCURRENT="nan",
        QQ_INBOX_TTL_DAYS="seven", QQ_FILE_INJECT_MAX_ITEMS="  ",
    )
    assert (d["QQ_FILE_MAX_MB_OWNER"], d["QQ_FILE_MAX_MB_MEMBER"], d["QQ_FILE_DOWNLOAD_TIMEOUT_SECONDS"]) == (20, 10, 30), d
    assert (d["QQ_FILE_MAX_PER_MESSAGE"], d["QQ_FILE_MAX_PER_SESSION"], d["QQ_FILE_MAX_CONCURRENT"]) == (5, 20, 3), d
    assert (d["QQ_INBOX_TTL_DAYS"], d["QQ_FILE_INJECT_MAX_ITEMS"]) == (7, 10), d
    assert _config_with_env(QQ_INBOX_TTL_DAYS="0")["QQ_INBOX_TTL_DAYS"] == 0

    # bool 写法
    for raw in ("1", "true", "TRUE", "Yes", " yes "):
        d = _config_with_env(QQ_FILE_ENABLED=raw, QQ_FILE_MEMBER_ENABLED=raw, QQ_FILE_SAFE_NAMES=raw)
        assert d["QQ_FILE_ENABLED"] and d["QQ_FILE_MEMBER_ENABLED"] and d["QQ_FILE_SAFE_NAMES"], raw
    for raw in ("0", "false", "no", "off", "random"):
        d = _config_with_env(QQ_FILE_ENABLED=raw, QQ_FILE_MEMBER_ENABLED=raw, QQ_FILE_SAFE_NAMES=raw)
        assert not (d["QQ_FILE_ENABLED"] or d["QQ_FILE_MEMBER_ENABLED"] or d["QQ_FILE_SAFE_NAMES"]), raw


# ═════════════════════════════════════════════════════════════════════
# 2 sanitize_filename
# ═════════════════════════════════════════════════════════════════════
def _check_sanitized(out, max_stem=60):
    assert isinstance(out, str) and out, repr(out)
    assert "/" not in out and "\\" not in out, out
    assert not any(ord(c) < 32 or ord(c) == 127 for c in out), repr(out)
    stem, _ = F.split_suffix(out)
    assert len(stem) <= max_stem + 1, (out, len(stem))      # 保留名前面加的下划线可能多 1 位，但保留名本身很短，实际不会越界


def test_2_sanitize_filename():
    table = {
        "report.pdf": "report.pdf",
        "../../etc/passwd": "passwd",
        "..\\..\\windows\\system32\\cmd.exe": "cmd.exe",
        "C:\\a\\b.txt": "b.txt",
        "a/b.txt": "b.txt",
        "/abs/path/data.CSV": "data.csv",
        "CON.txt": "_CON.txt",
        "con": "_con",
        "aux": "_aux",
        "LPT1": "_LPT1",
        "nul.tar.gz": "_nul.tar.gz",
        "COM9.log": "_COM9.log",
        "CONSOLE.txt": "CONSOLE.txt",              # 只有整段等于保留名才加前缀
        "name. . .": "name",
        "name.txt.": "name.txt",
        "   spaced name .txt  ": "spaced name.txt",       # stem 末尾的空格也去掉
        ".bashrc": "bashrc",
        "...hidden.md": "hidden.md",
        "a\x00b\x1fc.txt": "a_b_c.txt",
        "evil\u202egnp.exe": "evil_gnp.exe",         # U+202E 方向控制符
        "a<b>c:d\"e?f*g|h.txt": "a_b_c_d_e_f_g_h.txt",
        "a.tar.gz": "a.tar.gz",
        "noext": "noext",
        "weird.suffix_too_long_x": "weird.suffix_too_long_x",    # 后缀含下划线，不算后缀
        "报告 2024.XLSX": "报告 2024.xlsx",
        "表情😀.png": "表情😀.png",
        "": "file",
        "   ": "file",
        "...": "file",
        "/": "file",
        ".pdf": "pdf",
        "e\u0301.txt": "\u00e9.txt",                  # NFC 归一
    }
    for raw, want in table.items():
        got = F.sanitize_filename(raw)
        assert got == want, (raw, got, want)
        _check_sanitized(got)

    for bad in (None, 123, b"abc", ["a"], {"a": 1}, object()):
        assert F.sanitize_filename(bad) == "file", bad

    long_name = "x" * 500 + ".pdf"
    got = F.sanitize_filename(long_name)
    assert got == "x" * 60 + ".pdf", got
    got = F.sanitize_filename("中" * 200 + ".docx", max_stem=10)
    assert got == "中" * 10 + ".docx", got
    # 超长后缀不算后缀，整体作 stem 截断
    got = F.sanitize_filename("a." + "b" * 40)
    _check_sanitized(got)

    # 随机 / 对抗性输入：永不抛、输出恒合法
    import random
    rng = random.Random(7)
    alphabet = list("ab./\\:*?\"<>| \t\n\x00\u202e中😀CONauxLPT1") + ["..", "../", "..\\"]
    for _ in range(2000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        _check_sanitized(F.sanitize_filename(raw))


# ═════════════════════════════════════════════════════════════════════
# 3 其余 L0 纯函数
# ═════════════════════════════════════════════════════════════════════
def test_3_other_pure_functions():
    # make_disk_name
    a = F.make_disk_name("report.pdf", "abcd1234")
    assert a == "report_abcd1234.pdf", a
    assert F.make_disk_name("report.pdf", "abcd1234") == a                       # 同内容同名
    assert F.make_disk_name("report.pdf", "ffff0000") != a                       # 不同内容不同名
    assert F.make_disk_name("Data.CSV", "abcd1234") == "Data_abcd1234.csv"       # 后缀小写保留
    assert F.make_disk_name("my report (1)$.CSV", "abcd1234") == "my_report__1___abcd1234.csv"
    assert F.make_disk_name("a b&c;d!e#f'g\"h.txt", "abcd1234") == "a_b_c_d_e_f_g_h_abcd1234.txt"
    assert F.make_disk_name("销售 数据.xlsx", "abcd1234") == "销售_数据_abcd1234.xlsx"
    assert F.make_disk_name("noext", "abcd1234") == "noext_abcd1234"
    assert F.make_disk_name("a.tar.gz", "abcd1234") == "a.tar_abcd1234.gz"
    assert "/" not in F.make_disk_name("../x", "abcd1234") and "\\" not in F.make_disk_name("..\\x", "abcd1234")

    # split_suffix
    assert F.split_suffix("a.tar.gz") == ("a.tar", ".gz")
    assert F.split_suffix("noext") == ("noext", "")
    assert F.split_suffix("a.PDF") == ("a", ".pdf")
    assert F.split_suffix("a.verylongsuffix1") == ("a.verylongsuffix1", "")
    assert F.split_suffix("a.中文") == ("a.中文", "")
    assert F.split_suffix("") == ("", "") and F.split_suffix(None) == ("", "")

    # format_file_size
    for n, want in [(0, "0 B"), (512, "512 B"), (1023, "1023 B"), (1024, "1.0 KB"), (36045, "35.2 KB"),
                    (int(1.2 * 1024 * 1024), "1.2 MB"), (int(1.5 * 1024 ** 3), "1.5 GB"),
                    (-5, "0 B"), ("abc", "0 B"), (None, "0 B"), (float("nan"), "0 B"), (float("inf"), "0 B"),
                    (1024 ** 4, "1024.0 GB")]:
        got = F.format_file_size(n)
        assert got == want, (n, got, want)
    assert F.format_file_size(1024 * 1024 - 1) == "1.0 MB"          # 四舍五入后不出现 1024.0 KB

    # validate_session_subdir
    sid = uuid.uuid4().hex
    assert F.validate_session_subdir(sid) == sid
    assert F.validate_session_subdir("abc-DEF_123") == "abc-DEF_123"
    for bad in ("..", "a/b", "abc", "x" * 65, "a b c d e f g", "../../etcc", "abcdefgh\n", "", None, 5):
        _raises(ValueError, F.validate_session_subdir, bad)
    try:
        F.validate_session_subdir("../../secret-path")
    except ValueError as e:
        assert "secret" not in str(e)                                  # 文案不含入参

    # receipt_name
    assert F.receipt_name("a.py", safe=True) == "a[.]py"
    assert F.receipt_name("a.tar.gz", safe=True) == "a[.]tar[.]gz"
    assert F.receipt_name("a.py", safe=False) == "a.py"

    # guess_suffix_from_bytes
    assert F.guess_suffix_from_bytes(b"%PDF-1.7 rest") == ".pdf"
    assert F.guess_suffix_from_bytes(b"\x89PNG\r\n\x1a\n....") == ".png"
    assert F.guess_suffix_from_bytes(b"\xff\xd8\xff\xe0....") == ".jpg"
    assert F.guess_suffix_from_bytes(b"GIF89a....") == ".gif" and F.guess_suffix_from_bytes(b"GIF87a") == ".gif"
    assert F.guess_suffix_from_bytes(b"PK\x03\x04....") == ".zip"
    assert F.guess_suffix_from_bytes(b"\x1f\x8b\x08") == ".gz"
    assert F.guess_suffix_from_bytes(b"hello world") == "" and F.guess_suffix_from_bytes(b"") == ""
    assert F.guess_suffix_from_bytes(None) == "" and F.guess_suffix_from_bytes("PK") == ""
    assert F.guess_suffix_from_bytes(bytearray(b"%PDF-1")) == ".pdf"

    # classify_content_type
    for ct, want in [("image/png", "image"), ("IMAGE/JPEG", "image"), ("image", "image"), ("audio/mpeg", "audio"),
                     ("voice", "audio"), ("video/mp4", "video"), ("file", "file"), ("application/pdf", "file"),
                     ("text/plain", "file"), ("application/octet-stream", "file"), ("", "file"), (None, "file"), (5, "file")]:
        assert F.classify_content_type(ct) == want, (ct, F.classify_content_type(ct), want)


# ═════════════════════════════════════════════════════════════════════
# 4 normalize_download_url
# ═════════════════════════════════════════════════════════════════════
def test_4_normalize_download_url():
    n = F.normalize_download_url
    assert n("https://multimedia.nt.qq.com.cn/download?fileid=abc&rkey=SECRET") == \
        "https://multimedia.nt.qq.com.cn/download?fileid=abc&rkey=SECRET"
    assert n("  https://a.example.com/x  ") == "https://a.example.com/x"
    assert n("//a.example.com/x") == "https://a.example.com/x"
    assert n("a.example.com/x?y=1") == "https://a.example.com/x?y=1"
    assert n("http://a.example.com/x") == "https://a.example.com/x"                # 升级为 https，不降级
    assert n("HTTP://A.example.com/x") == "https://A.example.com/x"
    assert n("https://a.example.com:443/x") == "https://a.example.com:443/x"

    bad = [
        "", "   ", None, 5, b"https://a.example.com",
        "http://127.0.0.1/x", "https://10.0.0.5/x", "https://192.168.1.1/x", "https://[::1]/x", "https://[2001:db8::1]/x",
        "https://2130706433/x", "https://127.1/x", "https://0x7f.0.0.1/x",
        "https://localhost/x", "https://intranet/x", "https://printer.local/x", "https://db.internal/x", "https://a.localhost/x",
        "https://user:pw@a.example.com/x", "https://a.example.com@evil.com/x", "https://@a.example.com/x",
        "https://a.example.com:8443/x", "http://a.example.com:80/x", "https://a.example.com:abc/x",
        "https://a.example.com/x y", "https://a.example.com/x\ny", "https://a.example.com/\x00",
        "https://a.example.com\\@evil.com/",
        "https://" + "a" * 2048 + ".com/",
        "file:///etc/passwd", "ftp://a.example.com/x", "javascript:alert(1)", "gopher://a.example.com/",
        "https://", "https:///x", "https://a..example.com/x", "https://.example.com/x", "https://-a.example.com/x",
    ]
    for raw in bad:
        assert n(raw) is None, raw
    for raw in (None, 5, [], {}):
        assert n(raw) is None


# ═════════════════════════════════════════════════════════════════════
# 5 incoming_files_from_attachments
# ═════════════════════════════════════════════════════════════════════
class _Att:
    def __init__(self, content_type="file", filename=None, size=None, url=None):
        self.content_type, self.filename, self.size, self.url = content_type, filename, size, url


class _Seg:
    def __init__(self, type_, **data):
        self.type, self.data = type_, data


def test_5_incoming_files_from_attachments():
    fn = F.incoming_files_from_attachments

    atts = [
        _Att("file", "报告.pdf", 1234, "https://a.example.com/1"),
        _Att("image/png", "x.png", 10, "https://a.example.com/2"),
        _Att("audio/mpeg", "v.mp3", 10, "https://a.example.com/3"),
        _Att("video/mp4", "v.mp4", 10, "https://a.example.com/4"),
        _Att("voice", "v.silk", 10, "https://a.example.com/5"),
        _Att("application/pdf", "b.pdf", 99, "https://a.example.com/6"),
        _Att("file", "no-url.txt", 1, None),
        _Att("file", "blank-url.txt", 1, "   "),
        _Att("file", "dup.pdf", 5, "https://a.example.com/1"),
    ]
    got = fn(atts, [_Seg("file", url="https://ignored.example.com/x")])
    assert got == [
        IncomingFile(url="https://a.example.com/1", filename="报告.pdf", size=1234, content_type="file"),
        IncomingFile(url="https://a.example.com/6", filename="b.pdf", size=99, content_type="application/pdf"),
    ], got                                                      # attachments 非空时只用它，按 url 去重保序

    # dict 形式；缺字段 / 脏字段
    got = fn([{"content_type": "file", "filename": None, "size": "abc", "url": "https://a.example.com/d"},
              {"content_type": None, "filename": 5, "size": -3, "url": "https://a.example.com/e"}], None)
    assert got == [IncomingFile(url="https://a.example.com/d", filename="", size=0, content_type="file"),
                   IncomingFile(url="https://a.example.com/e", filename="", size=0, content_type="")], got

    # attachments 为空 / None -> 回落到 file 段
    segs = [_Seg("text", text="hi"), _Seg("file", url="https://a.example.com/s1"), _Seg("image", url="https://a.example.com/i"),
            _Seg("file", url="https://a.example.com/s1"), _Seg("file", url=""), _Seg("file")]
    for empty in (None, [], ()):
        assert fn(empty, segs) == [IncomingFile(url="https://a.example.com/s1")], empty
    assert fn(None, None) == [] and fn(None, []) == []

    # 只有图片附件：不回落、也没有文件
    assert fn([_Att("image/png", "x.png", 1, "https://a.example.com/i")], [_Seg("file", url="https://a.example.com/s")]) == []

    # 非法 URL 也保留（下载阶段再报错）
    got = fn([_Att("file", "x.txt", 1, "ftp://bad/x"), _Att("file", "y.txt", 1, "http://127.0.0.1/y")], None)
    assert [g.url for g in got] == ["ftp://bad/x", "http://127.0.0.1/y"], got

    # 异常输入不抛
    class _Boom:
        def __iter__(self):
            yield _Att("file", "ok.txt", 1, "https://a.example.com/ok")
            raise RuntimeError("boom")

    assert fn(_Boom(), None) == [IncomingFile(url="https://a.example.com/ok", filename="ok.txt", size=1, content_type="file")]
    for weird in (5, "str", object(), [object()], [None], [5]):
        assert isinstance(fn(weird, weird), list)


# ═════════════════════════════════════════════════════════════════════
# 汇总
# ═════════════════════════════════════════════════════════════════════
SCENARIOS = [
    ("1 配置默认值与夹紧", test_1_config_defaults_and_clamping),
    ("2 sanitize_filename", test_2_sanitize_filename),
    ("3 其余 L0 纯函数", test_3_other_pure_functions),
    ("4 normalize_download_url", test_4_normalize_download_url),
    ("5 incoming_files_from_attachments", test_5_incoming_files_from_attachments),
]


def main():
    print("=" * 70)
    print("开始运行 QQ 文件接收离线冒烟测试...")
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
