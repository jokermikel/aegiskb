FROM python:3.11-slim

WORKDIR /app

# 依赖单独一层：只改代码时不会触发依赖重装。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 健康检查走 /health，让编排系统能判断容器是否真的可用。
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# 启动前先应用迁移。生产环境已关闭 AUTO_CREATE_SCHEMA，结构只能由 Alembic 管理；
# 多实例部署时应把迁移拆成独立的一次性任务，避免并发执行同一次升级。
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
