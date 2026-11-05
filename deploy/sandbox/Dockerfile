FROM python:3.13-slim

LABEL gnk48.sandbox.version=2

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    MPLBACKEND=Agg \
    MATPLOTLIBRC=/opt/mplrc \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/work

RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 sandbox && \
    useradd -u 10001 -g sandbox -d /work -m -s /bin/sh sandbox && \
    mkdir -p /work/outputs /opt/mplrc /opt/mplcache /tmp/mpl && \
    chown -R sandbox:sandbox /work /opt/mplcache /tmp/mpl

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY matplotlibrc /opt/mplrc/matplotlibrc

# 预热字体缓存到 /opt/mplcache
RUN MPLCONFIGDIR=/opt/mplcache python -c "import matplotlib.pyplot as plt" && \
    chmod -R 755 /opt/mplcache

# 构建期自检：渲染含中文与负号的图，断言无 Glyph missing 警告与 findfont 噪声
RUN MPLCONFIGDIR=/opt/mplcache python - << "EOF"
import sys, warnings, io
import matplotlib
import matplotlib.pyplot as plt

stderr_capture = io.StringIO()
old_stderr = sys.stderr
sys.stderr = stderr_capture

with warnings.catch_warnings(record=True) as warns:
    warnings.simplefilter("always")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [-1.5, 0.0, 2.5], label="示例曲线")
    ax.set_title("测试中文与负号 -1.5")
    ax.set_xlabel("横轴 (X)")
    ax.set_ylabel("纵轴 (Y)")
    ax.legend()
    fig.savefig("/tmp/selftest.png")
    plt.close(fig)

sys.stderr = old_stderr
err_out = stderr_capture.getvalue()
assert "findfont" not in err_out, f"findfont noise in stderr: {err_out}"
for w in warns:
    msg = str(w.message)
    assert "Glyph" not in msg and "missing" not in msg.lower(), f"Glyph missing warning: {msg}"
print("Build-time font self check passed successfully.")
EOF

RUN rm -f /tmp/selftest.png

USER 10001:10001
WORKDIR /work
CMD ["sleep", "infinity"]
