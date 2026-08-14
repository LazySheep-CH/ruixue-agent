#!/usr/bin/env bash
# 备份 PostgreSQL(数据的 source of truth)。
#
# 为什么只备份 PG、不备份 Milvus:
#   PG 里的东西【丢了就没了】—— 用户账号、对话历史、文档与分块正文、BM25 词频。
#   Milvus 里只有向量,是 PG 正文的【派生物】:丢了可以用 scripts/load_milvus.py
#   从 PG 重算(约几十分钟),不必用备份去扛。这就是"数据是资产、索引是派生物"。
#
# 用法:
#   bash scripts/ops/backup.sh              # 备份到 backups/
#   BACKUP_DIR=/mnt/nas bash scripts/ops/backup.sh
#
# 生产建议:加进 crontab 每日跑一次,并把 backups/ 同步到【另一台机器或对象存储】——
# 备份和数据库在同一块盘上,那块盘坏了就一起没了,等于没备份。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/backend/backups}"
KEEP_DAYS="${KEEP_DAYS:-14}"          # 保留天数,超过的自动清理
CONTAINER="${PG_CONTAINER:-rxmvp-postgres}"

# 从 docker/.env 读凭据 —— 不在脚本里硬编码密码
set -a; source "$ROOT/docker/.env"; set +a

mkdir -p "$BACKUP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
FILE="$BACKUP_DIR/ruixue_${STAMP}.dump"

echo "[备份] 开始 → $FILE"
# -Fc = 自定义压缩格式:比纯 SQL 小很多,且 pg_restore 支持并行恢复、可选表恢复
docker exec "$CONTAINER" pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > "$FILE"

SIZE=$(du -h "$FILE" | cut -f1)
echo "[备份] 完成,大小 $SIZE"

# 校验:pg_restore --list 能列出目录结构 = 文件结构完整(不是半截或空文件)。
# 备份不校验 = 不知道自己有没有备份 —— 很多事故是"以为有备份"。
if docker exec -i "$CONTAINER" pg_restore --list < "$FILE" > /dev/null 2>&1; then
  echo "[校验] 通过:备份文件结构完整"
else
  echo "[校验] ✗ 失败:备份文件可能损坏!" >&2
  exit 1
fi

# 清理过期备份
find "$BACKUP_DIR" -name "ruixue_*.dump" -mtime "+$KEEP_DAYS" -delete 2>/dev/null || true
echo "[备份] 现存 $(find "$BACKUP_DIR" -name 'ruixue_*.dump' | wc -l) 份(保留 ${KEEP_DAYS} 天)"
