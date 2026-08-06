#!/usr/bin/env bash
# 从备份恢复 PostgreSQL。
#
# ⚠️ 这是【破坏性】操作:会覆盖目标库的现有数据。故:
#   1. 必须显式传备份文件路径(不猜"最新那份")
#   2. 非交互场景要显式设 CONFIRM=yes,防止手滑
#
# 用法:
#   bash scripts/ops/restore.sh backups/ruixue_20260727_120000.dump
#   RESTORE_DB=ruixue_verify bash scripts/ops/restore.sh <file>   # 恢复到另一个库(演练用)
#
# 恢复完 Milvus 需要重建:uv run python scripts/load_milvus.py
# (向量是 PG 正文的派生物,不进备份 —— 见 backup.sh 说明)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINER="${PG_CONTAINER:-rxmvp-postgres}"
FILE="${1:-}"

[ -z "$FILE" ] && { echo "用法: bash scripts/ops/restore.sh <备份文件>" >&2; exit 1; }
[ -f "$FILE" ] && : || { echo "备份文件不存在: $FILE" >&2; exit 1; }

set -a; source "$ROOT/docker/.env"; set +a
TARGET_DB="${RESTORE_DB:-$POSTGRES_DB}"

echo "[恢复] 源文件 : $FILE"
echo "[恢复] 目标库 : $TARGET_DB"
if [ "${CONFIRM:-}" != "yes" ]; then
  echo "⚠️  这会【覆盖】$TARGET_DB 的现有数据。确认请重跑并加 CONFIRM=yes" >&2
  exit 1
fi

# 目标库不存在就建(恢复演练常恢复到新库,不动生产库)
docker exec "$CONTAINER" psql -U "$POSTGRES_USER" -d postgres -tc \
  "SELECT 1 FROM pg_database WHERE datname='$TARGET_DB'" | grep -q 1 || {
  echo "[恢复] 目标库不存在,创建 $TARGET_DB"
  docker exec "$CONTAINER" createdb -U "$POSTGRES_USER" "$TARGET_DB"
}

# --clean --if-exists:先删同名对象再建,保证恢复到"干净的那一份",不和残留混在一起
# --no-owner:忽略原属主,换机器恢复时不会因为角色不存在而失败
echo "[恢复] 进行中……"
docker exec -i "$CONTAINER" pg_restore -U "$POSTGRES_USER" -d "$TARGET_DB" \
  --clean --if-exists --no-owner < "$FILE" 2>&1 | grep -vE "^pg_restore: (警告|warning)" || true

echo "[恢复] 完成。核对行数:"
docker exec "$CONTAINER" psql -U "$POSTGRES_USER" -d "$TARGET_DB" -At -c \
  "SELECT '  '||relname||' : '||n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT 6;"
echo "[提醒] 如需检索功能,重建 Milvus 向量:uv run python scripts/load_milvus.py"
