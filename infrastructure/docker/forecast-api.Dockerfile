FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "docker>=7.1,<8" \
    "fastapi>=0.115,<1" \
    "httpx>=0.27,<1" \
    "uvicorn[standard]>=0.30,<1"

COPY src/mlops_project/__init__.py /app/mlops_project/__init__.py
COPY src/mlops_project/serving/__init__.py /app/mlops_project/serving/__init__.py
COPY src/mlops_project/serving/server.py /app/mlops_project/serving/server.py

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "mlops_project.serving.server:app", "--host", "0.0.0.0", "--port", "8000"]
