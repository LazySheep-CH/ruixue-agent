# 把服务打包成一个"到哪都能跑"的镜像。多阶段思想:依赖层和代码层分开,
# 依赖没变就走缓存,改代码不用重装那一大坨依赖。
#
# 基础镜像用 Docker Hub 的 python,不用 ghcr.io/astral-sh/uv。
# 原因很实际:ghcr.io 在国内经常连不通(本机实测 build 直接 EOF 失败),
# 而基础镜像拉不下来 = 整条部署链断在第一步。
# 基础镜像必须来自【构建机确实能访问的仓库】,这不是风格问题。
# uv 从 PyPI 装并锁版本 —— 构建工具的版本也要可复现。
FROM python:3.11-slim-bookworm

RUN pip install --no-cache-dir uv==0.5.11

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

# ② 把嵌入模型【烤进镜像】,而不是运行时下载。
#
#    为什么必须这么做:SentenceTransformer("BAAI/bge-small-zh-v1.5") 默认在首次调用时
#    去 HuggingFace 拉 ~100MB。放到运行时的后果是——
#      · 第一个用户要等模型下完,冷启动几十秒;
#      · 每次扩容/重启新容器都重下一遍;
#      · 生产机(尤其国内)连不上 HF 时服务直接起不来 —— 把一个构建期就能解决的
#        问题,变成了线上的可用性风险。
#    构建期下一次、运行期永不联网,才是正确的分界。
#
#    HF_HOME 固定到镜像内路径,保证运行时找的就是构建时下的那份。
#    构建机连不上 HF 时:--build-arg HF_ENDPOINT=https://hf-mirror.com 走镜像站。
ARG HF_ENDPOINT=""
ENV HF_HOME=/opt/hf
RUN HF_ENDPOINT="$HF_ENDPOINT" uv run --no-project python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# ③ 再拷源码,把本项目自己装进去。
COPY . .
RUN uv sync --frozen --no-dev

# ④ 构建期断言:预测模型产物必须在镜像里。
#
#    models/ 是 gitignore 的(92MB 二进制不进 git),所以镜像必须在【有训练产物的
#    机器上】构建,再推镜像仓库、部署端拉镜像 —— 产物随镜像走,不随 git 走
#    (标准做法:代码进 git,模型产物进制品库)。
#    没有这道断言,就会安静地打出一个"三个预测工具全报 FileNotFoundError"的镜像,
#    直到用户提问才发现。宁可构建失败,不可线上失败。
RUN for m in dr wvtr ts; do \
      test -f "models/predictors/$m/${m}_model.joblib" || \
      { echo "❌ 缺少预测模型产物 models/predictors/$m/。先训练:uv run python scripts/train/train.py $m"; exit 1; }; \
    done

# 运行期强制离线:模型已在镜像里,不该再有任何下载。真缺文件就立刻报错,
# 而不是偷偷联网 —— 那样问题会拖到线上才暴露。
ENV HF_HUB_OFFLINE=1

# ⑤ 不用 root 跑:容器被攻破时,攻击者拿到的也只是个无权限用户。
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app /opt/hf
USER appuser

# 服务监听 8000
EXPOSE 8000

# 健康检查:容器编排(Docker/K8s)靠它判断容器是否活着 —— 正好用我们 P2 做的 /health。
# start-period 给足 60s:进程起来后还要加载嵌入模型和三个树模型。
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# 启动服务。worker 数走环境变量而不是写死:它受 PostgreSQL max_connections 约束
# (每 worker 约 30 条连接),不能按 CPU 核数拍脑袋加。详见 docs/运维手册.md。
ENV UVICORN_WORKERS=2
CMD ["sh", "-c", "uv run uvicorn ruixue_app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
