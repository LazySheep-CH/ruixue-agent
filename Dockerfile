# 生产镜像:多阶段构建。
#
# builder 阶段装依赖、烤模型;runtime 阶段只拷【最终的 .venv 和源码】。
# 为什么必须分两段:中间层是【累加】的 —— 在同一段里先装 CUDA 版 torch 再删掉,
# 镜像照样带着那 6GB(删除只是加了一层"标记删除")。只有换阶段重新 COPY,
# 那些层才真的不进最终镜像。实测 12.6GB → 见文件末尾。
#
# 基础镜像用 Docker Hub 的 python,不用 ghcr.io/astral-sh/uv:
# ghcr.io 在国内经常连不通(本机实测 build 直接 EOF),而基础镜像拉不下来
# = 整条部署链断在第一步。基础镜像必须来自【构建机确实能访问的仓库】。
FROM python:3.11-slim-bookworm AS builder

# uv 从 PyPI 装并锁版本 —— 构建工具的版本也要可复现。
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

# ② 再拷源码,把本项目自己装进去。
COPY . .
RUN uv sync --frozen --no-dev

# ③ 把 torch 换成【CPU 版】。
#
#    必须放在最后一次 uv sync 【之后】:uv sync 的语义是"让环境和 lock 完全一致",
#    放前面会被它原样装回去。
#
#    为什么值得专门做:Linux 上 pip/uv 默认装的是 CUDA 版 torch,实测占
#        nvidia 3.6G + torch 1.7G + triton 691M ≈ 6GB
#    而服务器上【没有 GPU,一行 CUDA 代码都不会跑】。这 6GB 全是死重:
#    推镜像慢、拉镜像慢、云盘还要为它付钱。
#
#    lock 文件保持不动 —— 本机开发要用 GPU 训练和灌向量(RTX 3090),
#    改 lock 会把本地也降级成 CPU。**服务端的取舍不该反过来绑架开发端。**
RUN uv pip install --python /app/.venv/bin/python --reinstall \
        torch --index-url https://download.pytorch.org/whl/cpu \
 && rm -rf /app/.venv/lib/python3.11/site-packages/nvidia \
           /app/.venv/lib/python3.11/site-packages/triton \
    && find /app/.venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ④ 把嵌入模型【烤进镜像】,而不是运行时下载。
#
#    为什么必须这么做:SentenceTransformer("BAAI/bge-small-zh-v1.5") 默认在首次调用时
#    去 HuggingFace 拉 ~100MB。放到运行时的后果是——
#      · 第一个用户要等模型下完,冷启动几十秒;
#      · 每次扩容/重启新容器都重下一遍;
#      · 生产机(尤其国内)连不上 HF 时服务直接起不来 —— 把一个构建期就能解决的
#        问题,变成了线上的可用性风险。
#    构建期下一次、运行期永不联网,才是正确的分界。
#
#    放在换 CPU torch 【之后】还有个附带好处:它顺便验证了 CPU 版 torch 真能
#    把模型加载起来 —— 构建期就验过,不用等线上。
#
# ⚠ HF_ENDPOINT 默认给【官方地址】而不是空串。
#   空字符串 ≠ 未设置:HF_ENDPOINT="" 会被 huggingface_hub 当成 base URL,
#   报 httpx.UnsupportedProtocol: Request URL is missing an 'http://' protocol。
#   这是环境变量最经典的坑 —— "没配"和"配成空"是两回事。
#   国内构建机改镜像站:--build-arg HF_ENDPOINT=https://hf-mirror.com
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_HOME=/opt/hf
# ⚠ 直接调 .venv 的解释器。不能用 `uv run --no-project`——它的意思正是
#   "忽略本项目环境",于是在临时空环境里跑,import sentence_transformers 直接失败。
RUN HF_ENDPOINT="$HF_ENDPOINT" /app/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# ⑤ 构建期断言:预测模型产物必须在镜像里。
#
#    models/ 是 gitignore 的(二进制不进 git),所以镜像必须在【有训练产物的
#    机器上】构建,再推镜像仓库、部署端拉镜像 —— 产物随镜像走,不随 git 走
#    (标准做法:代码进 git,模型产物进制品库)。
#    没有这道断言,就会安静地打出一个"三个预测工具全报 FileNotFoundError"的镜像,
#    直到用户提问才发现。宁可构建失败,不可线上失败。
#
#    ⚠ 名字必须【大写】(DR/WVTR/TS)—— 那是 predictors/schema.py 里 MODELS 的 key。
#    我第一版写成小写,在 Windows 上自查通过(NTFS 不区分大小写),一进 Linux 容器
#    就找不到文件。这类"本机好好的、一上服务器就挂"的问题,只有真把构建跑一遍才暴露 ——
#    这也正是这道断言存在的意义。
RUN for m in DR WVTR TS; do \
      test -f "models/predictors/$m/${m}_model.joblib" || \
      { echo "❌ 缺少预测模型产物 models/predictors/$m/。先训练:uv run python scripts/train/train.py $m"; exit 1; }; \
    done

# ⑥ 构建期断言:配置文件必须在,且里面不许出现明文密钥。
#
#    config/config.yaml 是【要进镜像】的 —— 它写的是 api_key: $DEEPSEEK_API_KEY
#    这种占位符,运行时才从环境变量解析,本身不含密钥。我一开始把它当密钥文件
#    排除掉了,结果 app 一启动就 FileNotFoundError(实测踩过)。
#
#    但"万一有人图省事把明文 key 写进去"是真实风险 —— 镜像一旦推到仓库,
#    谁能拉就谁能看。所以不靠约定管,靠这道断言:出现 sk- 开头的字面量就让
#    构建失败。**把纪律变成机制**,人会忘,断言不会。
RUN test -f config/config.yaml || { echo "❌ 缺 config/config.yaml"; exit 1; }
RUN if grep -qE '(api_key|token|secret)[[:space:]]*:[[:space:]]*.?sk-' config/config.yaml; then \
      echo "❌ config/config.yaml 里有明文密钥!改成 \$ENV_VAR 占位符再构建"; exit 1; \
    fi


# ── 运行阶段:只带最终产物,不带 uv、pip 缓存和被替换掉的旧 torch ──────────
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# 先建用户,再带着 --chown 拷贝。
#
# ⚠ 不能拷完再 `RUN chown -R`:改文件属主会让联合文件系统把【每一个文件】
#   都复制进新的一层 —— 2.4GB 的内容在镜像里存两遍。实测这一条就白白多出 4GB。
#   COPY --chown 在拷的时候就把属主定好,不产生第二份。
RUN useradd -m -u 1001 appuser

# venv 里存的是绝对路径,所以两个阶段的 WORKDIR 必须一致(都是 /app),
# 否则拷过来的解释器找不到自己的 site-packages。
COPY --from=builder --chown=appuser:appuser /app /app
COPY --from=builder --chown=appuser:appuser /opt/hf /opt/hf

# 直接把 venv 放进 PATH —— 运行阶段不装 uv,也就不能再用 `uv run`。
# 少一个运行期依赖,也少一个可以被利用的工具。
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    PYTHONDONTWRITEBYTECODE=1
# 运行期强制离线:模型已在镜像里,不该再有任何下载。真缺文件就立刻报错,
# 而不是偷偷联网 —— 那样问题会拖到线上才暴露。
ENV HF_HUB_OFFLINE=1

# 不用 root 跑:容器被攻破时,攻击者拿到的也只是个无权限用户。
USER appuser

EXPOSE 8000

# 健康检查:容器编排(Docker/K8s)靠它判断容器是否活着 —— 用 P2 做的 /health。
# start-period 给足 60s:进程起来后还要加载嵌入模型和三个树模型。
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# worker 数走环境变量而不是写死:它受 PostgreSQL max_connections 约束
# (每 worker 约 30 条连接),不能按 CPU 核数拍脑袋加。详见 docs/运维手册.md。
ENV UVICORN_WORKERS=2
# ⚠ 必须写 exec ——【容器化最经典的坑】。
#
# 不写 exec 时,PID 1 是 `sh`,uvicorn 是它的子进程。而 shell 【不会】把
# SIGTERM 转发给子进程:docker stop 发的信号 sh 收到了、uvicorn 没收到,
# 于是优雅停机的代码一行都不执行,等宽限期一到直接 SIGKILL 硬杀。
#
# 实测就是这样:我写好了 runs.shutdown()(排空在途运行、把没跑完的落库为失败),
# 日志里却连一句 "Shutting down" 都没有 —— 因为进程根本没收到信号。
# 这种 bug 单看代码永远发现不了,只有真的 docker restart 一次才知道。
#
# exec 让 uvicorn 【替换】掉 shell、自己成为 PID 1,信号就直达了。
# (仍然要用 sh -c,因为 ${UVICORN_WORKERS} 需要 shell 做变量展开。)
CMD ["sh", "-c", "exec uvicorn ruixue_app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
