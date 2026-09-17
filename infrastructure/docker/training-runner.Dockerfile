# syntax=docker/dockerfile:1.7

# AutoGluon 1.6.1 requires torch >=2.10,<2.14. Installing it once in this
# reusable runner avoids replacing Torch during every ZenML step build.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /uvx /usr/local/bin/

COPY infrastructure/docker/training-runner-requirements.txt /tmp/training-runner-requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages \
    -r /tmp/training-runner-requirements.txt \
    && rm -f /tmp/training-runner-requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
