"""告警检查:读 /metrics,按规则判断,超标就发通知。

    uv run python scripts/check_alerts.py                      # 检查一次,打印结果
    uv run python scripts/check_alerts.py --webhook <url>      # 超标时 POST 到 webhook
    uv run python scripts/check_alerts.py --list               # 只列规则,不检查

放进 cron(每 5 分钟一次):
    */5 * * * * cd /path/to/repo && uv run python scripts/check_alerts.py --webhook $ALERT_WEBHOOK

## 为什么是脚本 + cron,而不是 Alertmanager

有指标没人看等于没有监控。补上"有人看"这一步有两条路:
上一整套 Prometheus + Alertmanager(三个容器、三份配置、一套面板),
或者一个读 /metrics 的脚本挂 cron。

我们这个规模,后者足够 —— 而且**因为指标本身是 Prometheus 格式的,
将来换成真 Alertmanager 时,应用侧一行都不用改**,只是把这个脚本扔掉。
接口按标准来、实现先简陋,这是刻意的取舍。

## 规则设计的两条纪律

**① 每条规则要能说清"该做什么"。** 只报"失败率 30%"没有用,
值班的人第一反应是"然后呢"。所以每条规则都带 `action` —— 直接告诉他先看哪儿。

**② 不能在没有流量时报警。** 凌晨没人用,失败率分母是 0 ——
写不好就会每天半夜报一次假警。假警报的真正代价不是吵醒人,
是**让所有人开始忽略警报**,那时真出事也没人看。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_URL = "http://127.0.0.1:8000/metrics"
TIMEOUT_S = 15


@dataclass(frozen=True)
class Rule:
    name: str
    level: str  # critical / warning
    describe: str
    action: str  # 触发后先做什么 —— 见模块说明纪律 ①


def _get(metrics: dict[str, float], key: str, default: float = 0.0) -> float:
    return metrics.get(key, default)


def evaluate(m: dict[str, float]) -> list[tuple[Rule, str]]:
    """按规则检查,返回 [(规则, 具体数值描述)]。"""
    fired: list[tuple[Rule, str]] = []

    # ── 依赖挂了:这是"服务活着但不干活"的直接原因 ──
    for dep in ("postgres", "redis", "milvus"):
        if _get(m, f'ruixue_dep_up{{dep="{dep}"}}', 1) < 1:
            fired.append(
                (
                    Rule(
                        f"dep_down_{dep}",
                        "critical",
                        f"{dep} 连不上",
                        f"先看容器是否在跑(docker ps),再看 {dep} 自己的日志。"
                        + ("PG 挂了服务基本不可用。" if dep == "postgres" else "")
                        + (
                            "Milvus 挂了知识问答和记忆召回失效,但对话仍能继续。"
                            if dep == "milvus"
                            else ""
                        )
                        + (
                            "Redis 挂了限流/配额降级放行,成本闸门临时失效。"
                            if dep == "redis"
                            else ""
                        ),
                    ),
                    "不可达",
                )
            )

    # ── 失败率:注意分母为 0 的情况 ──
    total = _get(m, "ruixue_runs_total_1h")
    ratio = _get(m, "ruixue_runs_failure_ratio_1h")
    # 至少要有 5 次运行才判失败率。1 次里失败 1 次 = 100%,那是噪声不是信号。
    # 见模块说明纪律 ②:假警报会让人开始忽略警报。
    if total >= 5 and ratio >= 0.3:
        fired.append(
            (
                Rule(
                    "high_failure_ratio",
                    "critical",
                    "最近 1 小时运行失败率过高",
                    "看 runs 表里 status='failed' 的 error 字段(已脱敏但够定位),"
                    "多数情况是模型 API 或某个外部依赖在抖。",
                ),
                f"{ratio:.0%}({int(total)} 次运行)",
            )
        )
    elif total >= 5 and ratio >= 0.1:
        fired.append(
            (
                Rule("elevated_failure_ratio", "warning", "失败率偏高", "留意,暂不需处理。"),
                f"{ratio:.0%}({int(total)} 次运行)",
            )
        )

    # ── 容量:排队说明并发不够 ──
    inflight, cap = _get(m, "ruixue_inflight_runs"), _get(m, "ruixue_capacity_runs", 1)
    if cap and inflight >= 0.8 * cap:
        fired.append(
            (
                Rule(
                    "capacity_near_limit",
                    "warning",
                    "在途运行逼近容量上限",
                    "再高就会开始返回 503。压测结论是瓶颈在内存(677MB/worker),"
                    "扩容前先确认机器内存够不够。",
                ),
                f"{int(inflight)}/{int(cap)}",
            )
        )

    # ── 卡住的运行:reap_stale 只在启动时跑,中途卡住没人管 ──
    running = _get(m, "ruixue_runs_running")
    if running > 0 and inflight == 0:
        fired.append(
            (
                Rule(
                    "orphan_running",
                    "warning",
                    "库里有 running 记录但进程里没有在途任务",
                    "说明上次进程非正常退出留下了残留;重启会由 reap_stale 清掉,"
                    "但在此之前这些用户会一直等一个不会完成的运行。",
                ),
                f"{int(running)} 条",
            )
        )

    # ── 存储:checkpoints 是已知会无限增长的那张表 ──
    ckpt = sum(
        _get(m, f'ruixue_table_bytes{{table="{t}"}}')
        for t in ("checkpoints", "checkpoint_writes", "checkpoint_blobs")
    )
    if ckpt > 5 * 1024**3:
        fired.append(
            (
                Rule(
                    "checkpoint_bloat",
                    "warning",
                    "checkpoint 表体积过大",
                    "这是已知技术债:每个图步骤都存一份快照,只增不删。"
                    "需要按 thread 的最后活跃时间做清理(见运维手册)。",
                ),
                f"{ckpt / 1024**3:.1f} GB",
            )
        )

    # ── 指标自己取不全 ──
    if _get(m, "ruixue_metrics_errors") > 0:
        fired.append(
            (
                Rule(
                    "metrics_incomplete",
                    "warning",
                    "部分指标采集失败",
                    "上面某些判断可能是基于缺失数据做的,先看服务日志里 ruixue.metrics 的告警。",
                ),
                f"{int(_get(m, 'ruixue_metrics_errors'))} 块",
            )
        )
    return fired


_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+0-9.eE]+)$")


def parse_metrics(text: str) -> dict[str, float]:
    """解析 Prometheus 文本。注释行跳过,其余按 `名字{标签} 值` 取。"""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if m := _LINE_RE.match(line):
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                continue
    return out


def fetch(url: str, token: str) -> str:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return r.read().decode("utf-8")


def notify(webhook: str, fired: list[tuple[Rule, str]]) -> None:
    """把告警 POST 出去。通用 JSON,飞书/Slack 的自定义机器人都能接。

    发送失败只打印不抛 —— **告警通道挂了不该让检查脚本本身退出非零**,
    否则 cron 会连告警都发不出来还没人知道。
    """
    worst = "critical" if any(r.level == "critical" for r, _ in fired) else "warning"
    text = f"【瑞雪 · {worst}】{len(fired)} 条告警\n" + "\n".join(
        f"· [{r.level}] {r.describe}:{v}\n  → {r.action}" for r, v in fired
    )
    body = json.dumps({"msg_type": "text", "content": {"text": text}}, ensure_ascii=False)
    try:
        req = urllib.request.Request(
            webhook, data=body.encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            print(f"已发送告警(HTTP {r.status})")
    except Exception as e:
        print(f"⚠ 告警发送失败:{type(e).__name__}: {e}", file=sys.stderr)


BACKUP_DIR = Path(__file__).resolve().parent.parent / "backups"
BACKUP_MAX_AGE_H = 26  # 每日备份 + 2 小时余量


def check_backup_age() -> list[tuple[Rule, str]]:
    """备份是否还在正常产出。

    「备份悄悄停跑」是最危险的一类静默失败:cron 被注释、磁盘满、容器改名,
    备份停了不会有任何报错 —— 直到真出事那天才发现最后一份是三个月前的。
    这条检查跑在宿主机(cron 所在),备份文件也在宿主机,读文件系统即可。
    """
    if not BACKUP_DIR.exists():
        return []  # 没有备份目录:可能在远端跑检查,不在这里误报
    import time

    dumps = sorted(BACKUP_DIR.glob("ruixue_*.dump"), key=lambda f: f.stat().st_mtime)
    if not dumps:
        return [
            (
                Rule(
                    "backup_missing",
                    "critical",
                    "备份目录里一份备份都没有",
                    "立刻手动跑 bash scripts/ops/backup.sh,并检查 crontab 是否还在。",
                ),
                "0 份",
            )
        ]
    age_h = (time.time() - dumps[-1].stat().st_mtime) / 3600
    if age_h > BACKUP_MAX_AGE_H:
        return [
            (
                Rule(
                    "backup_stale",
                    "critical",
                    "最新备份已过期",
                    "检查 crontab 与磁盘空间;备份停跑不会自己报错,只会在出事那天才被发现。",
                ),
                f"{age_h:.0f} 小时前({dumps[-1].name})",
            )
        ]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default="", help="API Key / JWT;/metrics 需要鉴权")
    ap.add_argument("--webhook", default="", help="有告警时 POST 到这里")
    ap.add_argument("--list", action="store_true", help="只列出会检查哪些规则")
    args = ap.parse_args()

    if args.list:
        print("检查项:依赖可达性(PG/Redis/Milvus)、运行失败率、容量水位、")
        print("        残留 running 记录、checkpoint 表体积、指标采集完整性")
        return 0

    try:
        text = fetch(args.url, args.token)
    except Exception as e:
        # 拉不到指标本身就是最高级别的告警:服务可能整个挂了。
        print(f"❌ [critical] 拉取 {args.url} 失败:{type(e).__name__}: {e}", file=sys.stderr)
        if args.webhook:
            notify(
                args.webhook,
                [
                    (
                        Rule(
                            "metrics_unreachable",
                            "critical",
                            "指标端点拉不到",
                            "先确认服务进程还在。",
                        ),
                        str(e)[:80],
                    )
                ],
            )
        return 2

    fired = evaluate(parse_metrics(text))
    fired += check_backup_age()
    if not fired:
        print("✅ 一切正常")
        return 0

    for r, v in fired:
        print(
            f"{'❌' if r.level == 'critical' else '⚠'} [{r.level}] {r.describe}:{v}\n   → {r.action}"
        )
    if args.webhook:
        notify(args.webhook, fired)
    # critical 才返回非零 —— warning 天天有,让 cron 天天报错等于噪声
    return 1 if any(r.level == "critical" for r, _ in fired) else 0


if __name__ == "__main__":
    raise SystemExit(main())
