# 把服务打包成一个"到哪都能跑"的镜像。多阶段思想:依赖层和代码层分开,
# 依赖没变就走缓存,改代码不用重装那一大坨依赖。
#
# 用官方 uv 镜像(自带 uv + Python 3.11),省得自己装。
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# uv 的两个环境变量:
#   COMPILE_BYTECODE=1  预编译 .pyc,启动更快
#   LINK_MODE=copy      容器里没有硬链接优化,直接复制(避免告警)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ① 只先拷【依赖清单】,单独装依赖。这一层只要 pyproject/uv.lock 没变就命中缓存,
#    改业务代码时不会重装 torch 那些大件。--no-dev 排除测试工具,--no-install-project
#    先不装本项目(因为代码还没拷进来)。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ② 再拷源码,把本项目自己装进去(editable)。
COPY . .
RUN uv sync --frozen --no-dev

# 服务监听 8000
EXPOSE 8000

# 健康检查:容器编排(Docker/K8s)靠它判断容器是否活着 —— 正好用我们 P2 做的 /health。
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# 启动服务。生产可加 --workers N(按 CPU 核数)提高并发。
CMD ["uv", "run", "uvicorn", "ruixue_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
