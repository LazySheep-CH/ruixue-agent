#!/usr/bin/env bash
# 把本地备份同步到【异地】—— 另一台机器或对象存储。
#
# 为什么必须有这一步:
#   backup.sh 产出的文件和数据库在同一块盘上。盘坏、机器被误清、勒索加密,
#   备份和数据一起没 —— 那不叫备份,叫副本。异地才是备份的下半句。
#
# 用法(OFFSITE_TARGET 决定去哪,写进 docker/.env 或环境变量):
#   OFFSITE_TARGET=oss://ruixue-backup/pg       # 阿里云 OSS(需装 ossutil 并 config)
#   OFFSITE_TARGET=user@host:/data/ruixue       # 另一台机器(需配好 ssh 免密)
#
#   bash scripts/ops/backup_offsite.sh          # 同步 backups/ 里最近一份
#   bash scripts/ops/backup_offsite.sh --all    # 同步全部(首次用)
#
# 生产 crontab(先本地备份,成功了再异地;放一条里,备份失败就不会传半截):
#   0 3 * * * cd /path/to/repo && bash scripts/ops/backup.sh && bash scripts/ops/backup_offsite.sh
#
# 两条纪律:
#   ① 传完必须【回读校验】—— 拿远端的文件大小和本地比对。网络传输是会静默截断的,
#      "命令返回 0"不等于"文件完整到达"。很多事故是"以为异地有备份"。
#   ② 没配置 OFFSITE_TARGET 时【退出码非零 + 大写警告】,不能安静跳过 ——
#      安静跳过的结果就是半年后才发现异地从来没同步过。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"

# 允许把 OFFSITE_TARGET 写在 docker/.env 里(和数据库凭据同一处,少一个要记的地方)
if [ -f "$ROOT/docker/.env" ]; then set -a; source "$ROOT/docker/.env"; set +a; fi

if [ -z "${OFFSITE_TARGET:-}" ]; then
  echo "!! 未配置 OFFSITE_TARGET —— 备份仍只在本机,盘坏了就全没了。" >&2
  echo "   在 docker/.env 里加一行,例如:" >&2
  echo "     OFFSITE_TARGET=oss://你的bucket/pg      (阿里云 OSS,需 ossutil)" >&2
  echo "     OFFSITE_TARGET=user@host:/data/ruixue   (另一台机器,需 ssh 免密)" >&2
  exit 2
fi

# 选文件:默认最近一份;--all 全部(首次同步用)
if [ "${1:-}" = "--all" ]; then
  FILES=("$BACKUP_DIR"/ruixue_*.dump)
else
  # ls -t 按时间排序取最新 —— 备份文件名里带时间戳,字典序=时间序,但 ls -t 更直接
  LATEST="$(ls -t "$BACKUP_DIR"/ruixue_*.dump 2>/dev/null | head -1 || true)"
  [ -n "$LATEST" ] || { echo "!! $BACKUP_DIR 里没有备份文件,先跑 backup.sh" >&2; exit 2; }
  FILES=("$LATEST")
fi

local_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

fail=0
for f in "${FILES[@]}"; do
  name="$(basename "$f")"
  size_local="$(local_size "$f")"
  echo "[异地] $name ($((size_local / 1024 / 1024))MB) → $OFFSITE_TARGET"

  case "$OFFSITE_TARGET" in
    oss://*)
      # ossutil:阿里云官方 CLI。cp 幂等(同名覆盖),stat 回读远端元数据做校验。
      ossutil cp -f "$f" "$OFFSITE_TARGET/$name" >/dev/null
      size_remote="$(ossutil stat "$OFFSITE_TARGET/$name" | awk '/Content-Length/{print $NF}' | tr -d '\r')"
      ;;
    *@*:*)
      host="${OFFSITE_TARGET%%:*}"; dir="${OFFSITE_TARGET#*:}"
      scp -q "$f" "$OFFSITE_TARGET/$name"
      size_remote="$(ssh "$host" "stat -c %s '$dir/$name' 2>/dev/null || stat -f %z '$dir/$name'")"
      ;;
    *)
      echo "!! OFFSITE_TARGET 格式不认识:$OFFSITE_TARGET(支持 oss://… 或 user@host:/path)" >&2
      exit 2
      ;;
  esac

  # 回读校验:大小必须一致。不做这步,"传成功"只是"命令没报错"。
  if [ "$size_local" = "${size_remote:-}" ]; then
    echo "[校验] 通过:远端 $size_remote 字节,与本地一致"
  else
    echo "!! [校验] 失败:本地 $size_local vs 远端 ${size_remote:-取不到} —— 这份异地备份不可信" >&2
    fail=1
  fi
done

exit $fail
