"""模型角色配置与双 provider 构造的离线冒烟测试（不访问网络）。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATED_ROLES = (
    "WORKER_LLM", "ORCHESTRATOR_LLM", "KNOWLEDGE_LLM",
    "INGESTION_VLM", "BROWSER_VLM", "DESKTOP_VLM",
)
for _role in MIGRATED_ROLES:
    os.environ[f"{_role}_PROVIDER"] = "magpie"
    os.environ.pop(f"{_role}_MODEL", None)
for _name in ("UTILITY_LLM_MODEL", "VLM_MODEL", "EXTRACTOR_LLM_MODEL"):
    os.environ.pop(_name, None)
os.environ["GEMINI_API_KEY"] = "dummy"
os.environ["SILICON_FLOW_API_KEY"] = "dummy"
os.environ["MAGPIE_API_KEY"] = "local"
os.environ["MAGPIE_BASE_URL"] = "http://127.0.0.1:3425"
os.environ["MAGPIE_MODEL"] = "minimax-code-global/MiniMax-M3.1-Flash"

from app import config  # noqa: E402
from app.infra import models  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402


def test_default_magpie_roles():
    for key in ("worker_llm", "orchestrator_llm", "desktop_llm", "knowledge_llm", "ingestion_vlm"):
        llm = getattr(models, key)
        assert isinstance(llm, ChatAnthropic), (key, type(llm))
        assert llm.model == config.MAGPIE_MODEL
        assert llm.anthropic_api_url == config.MAGPIE_BASE_URL
        assert llm.max_tokens == config.MAGPIE_MAX_TOKENS == 16384
        assert llm.default_request_timeout == config.MAGPIE_TIMEOUT_SECONDS
        assert llm.max_retries == config.MAGPIE_MAX_RETRIES
        assert llm._get_ls_params()["ls_provider"] == "anthropic"
        assert getattr(models, key) is llm, f"{key} 应保持惰性单例"
    assert models.worker_llm is not models.utility_llm
    assert models.knowledge_llm is not models.vlm
    assert models.knowledge_llm is not models.ingestion_vlm
    models.validate_required_keys()


def test_retained_gemini_roles():
    for key, expected in (
        ("utility_llm", "gemini-3.5-flash-lite"),
        ("vlm", "gemini-3.5-flash-lite"),
        ("extractor_llm", "gemini-3.5-flash-lite"),
    ):
        llm = getattr(models, key)
        assert isinstance(llm, ChatGoogleGenerativeAI), (key, type(llm))
        assert llm.model == expected


def test_missing_gemini_key_is_scoped():
    original = config.GEMINI_API_KEY
    try:
        config.GEMINI_API_KEY = None
        models._cache.clear()
        assert isinstance(models.worker_llm, ChatAnthropic), "Magpie 构造不应要求 Gemini key 或发起请求"
        for key in ("utility_llm", "vlm", "extractor_llm"):
            try:
                getattr(models, key)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{key} 缺少 Gemini key 时应抛出 ValueError")
        try:
            models.validate_required_keys()
        except ValueError:
            pass
        else:
            raise AssertionError("固定 Gemini 角色存在时，启动校验仍应要求 Gemini key")
    finally:
        config.GEMINI_API_KEY = original
        models._cache.clear()


def _run_child(code: str, provider: str):
    env = dict(os.environ)
    env["WORKER_LLM_PROVIDER"] = provider
    env.pop("WORKER_LLM_MODEL", None)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )


def test_rollback_and_unknown_provider():
    rollback = _run_child(
        "from app.infra import models\n"
        "from langchain_google_genai import ChatGoogleGenerativeAI\n"
        "m = models.worker_llm\n"
        "assert isinstance(m, ChatGoogleGenerativeAI), type(m)\n"
        "assert m.model == 'gemini-3.5-flash-lite', m.model\n",
        "gemini",
    )
    assert rollback.returncode == 0, rollback.stdout + rollback.stderr

    unknown = _run_child(
        "from app.infra import models\n"
        "try:\n"
        "    _ = models.worker_llm\n"
        "except ValueError as e:\n"
        "    assert 'WORKER_LLM_PROVIDER' in str(e), e\n"
        "else:\n"
        "    raise AssertionError('未知 provider 应报错')\n",
        "unknown",
    )
    assert unknown.returncode == 0, unknown.stdout + unknown.stderr


def test_text_extraction_ignores_thinking():
    from app.utils.text import extract_text
    content = [
        {"type": "thinking", "thinking": "private", "signature": "sig"},
        {"type": "text", "text": "visible "},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "answer"},
    ]
    assert extract_text(content) == "visible answer"


SCENARIOS = [
    ("默认 Magpie 角色与单例", test_default_magpie_roles),
    ("轻量角色固定 Gemini", test_retained_gemini_roles),
    ("缺 Gemini key 的作用域", test_missing_gemini_key_is_scoped),
    ("Gemini 回退与未知 provider", test_rollback_and_unknown_provider),
    ("文本提取忽略 thinking", test_text_extraction_ignores_thinking),
]


def main():
    failed = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
    if failed:
        raise SystemExit(1)
    print(f"模型配置测试全部通过: {len(SCENARIOS)}/{len(SCENARIOS)}")


if __name__ == "__main__":
    main()
