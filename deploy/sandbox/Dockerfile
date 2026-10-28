FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "pandas==2.3.3" \
    "numpy==2.2.6" \
    "matplotlib==3.10.9"