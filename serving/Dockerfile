# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    MODEL_CACHE_DIR=/var/lib/autogluon/model-cache \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /var/lib/autogluon/model-cache \
    && chown -R app:app /var/lib/autogluon

WORKDIR /app
COPY pyproject.toml ./
COPY uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --link-mode=copy

COPY app ./app
RUN chown -R app:app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
