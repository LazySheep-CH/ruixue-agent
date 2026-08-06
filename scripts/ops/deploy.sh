#!/usr/bin/env bash
# 一条命令部署/升级生产环境。
#
#   bash scripts/ops/deploy.sh
#
# 做什么:前置检查 → 构建镜像 → 起基础设施 → 跑迁移 → 滚动起应用 → 冒烟验证
#
# 为什么要有这个脚本而不是让人手敲 docker compose:
#   部署是【最容易出错又最不能出错】的一步。手敲会漏 -f prod.yml、会忘了跑迁移、
#   会在健康检查还没通过时就宣布上线。把顺序和检查固化成脚本,才是可重复的。
#
# 每一步失败都立刻停(set -e),绝不"带着错误继续往下走"。

set -euo pipefail

cd "$(dirname "$0")/../.."
COMPOSE="docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml"

say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0. 前置检查:把"跑到一半才发现"提前到"根本不开始" ──────────
say "前置检查"
[ -f docker/.env ]      || die "缺 docker/.env(基础设施配置)"
[ -f docker/.env.prod ] || die "缺 docker/.env.prod,先 cp docker/.env.prod.example docker/.env.prod 并填真值"

# 密钥不能是占位符 —— 用默认 JWT_SECRET 上线 = 任何人都能伪造登录态
grep -q '^JWT_SECRET=.\+' docker/.env.prod || die "docker/.env.prod 里 JWT_SECRET 为空,必须设置"
grep -q '^DEEPSEEK_API_KEY=sk-' docker/.env.prod || die "docker/.env.prod 里 DEEPSEEK_API_KEY 没填"

# 预测模型产物必须在本机 —— 镜像靠它构建(models/ 不进 git,见 Dockerfile ④)
for m in dr wvtr ts; do
  [ -f "models/predictors/$m/${m}_model.joblib" ] || \
    die "缺预测模型 models/predictors/$m/,先训练:uv run python scripts/train/train.py $m"
done
echo "  ✓ 配置与模型产物齐备"

# ── 1. 构建镜像 ────────────────────────────────────────────────
say "构建镜像(首次会下载嵌入模型,约几分钟)"
$COMPOSE build

# ── 2. 基础设施先行,并【等它真的健康】────────────────────────
# 不等的话,迁移会连上一个还在初始化的 PG 然后失败。
say "启动基础设施(PostgreSQL / Redis / Milvus)"
$COMPOSE up -d --wait postgres redis milvus
echo "  ✓ 依赖已就绪"

# ── 3. 数据库迁移(一次性任务,失败就中止部署)──────────────────
say "执行数据库迁移"
$COMPOSE run --rm migrate || die "迁移失败 —— 已中止,线上仍是旧版本(这是好事)"

# ── 4. 起应用 ──────────────────────────────────────────────────
say "启动应用(app / web / nginx)"
$COMPOSE up -d --wait app web nginx

# ── 5. 冒烟验证:不通过就不算部署成功 ─────────────────────────
# "容器起来了"不等于"服务能用"。必须打一次真实请求。
say "冒烟验证"
PORT="$(grep -E '^HTTP_PORT=' docker/.env.prod | cut -d= -f2 || echo 80)"
for i in $(seq 1 30); do
  if curl -fsS "http://localhost:${PORT:-80}/api/health/ready" >/dev/null 2>&1; then
    echo "  ✓ /api/health/ready 通过"
    curl -fsS "http://localhost:${PORT:-80}/" >/dev/null && echo "  ✓ 前端页面可访问"
    say "部署完成 → http://localhost:${PORT:-80}"
    exit 0
  fi
  sleep 2
done

# 走到这里说明服务没起来。直接把日志摆出来,省得再去手敲一遍 logs。
$COMPOSE logs --tail=50 app nginx
die "健康检查 60 秒内未通过,日志见上。回滚:APP_TAG=<上一个版本> bash scripts/ops/deploy.sh"
