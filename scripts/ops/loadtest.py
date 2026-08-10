"""压测:找出我们【自己系统】的瓶颈在哪。

## 为什么必须压,不能算

在这之前,我们对容量的所有说法都是算术:"16 路并发 × 17.7s/题 ≈ 3254 次/小时"。
算术假设每一层都线性、没有互相争抢 —— 而真实系统里瓶颈几乎总在意料之外的地方
(连接池、GIL、序列化、日志 IO)。**没压过的容量数字,面试时经不起一句
"你怎么知道瓶颈在哪"。**

## 为什么【不打真模型】

打真模型测出来的是 DeepSeek 的限速和网络,不是我们的系统。那是供应商的容量,
不是我们的。所以压测分两种模式:

    --mode gate   只压【闸门层】:鉴权 → 限流 → 配额 → 容量 → 建 Run
                  这是每个请求都要过的路径,也是我们真正能优化的部分。
                  不产生模型调用,不花钱,可以打很高的并发。

    --mode real   打真模型,少量请求,量端到端延迟。用来校准单题耗时。

先用 gate 找系统瓶颈,再用 real 校准单题成本 —— 两者相乘才是真实吞吐。

用法:
    uv run python scripts/ops/loadtest.py --mode gate --concurrency 50 --requests 2000
    uv run python scripts/ops/loadtest.py --mode real --requests 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

sys.stdout.reconfigure(encoding="utf-8")


def _percentile(xs: list[float], p: float) -> float:
    """第 p 百分位。压测只看平均值是【严重误导】——
    平均 200ms 可能藏着 5% 的请求要 10 秒,而那 5% 正是会投诉的用户。"""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return xs[k]


def _one(client: httpx.Client, url: str, headers: dict, payload: dict | None) -> tuple[int, float]:
    t = time.perf_counter()
    try:
        r = (
            client.get(url, headers=headers)
            if payload is None
            else client.post(url, headers=headers, json=payload)
        )
        return r.status_code, (time.perf_counter() - t) * 1000
    except Exception:
        return 0, (time.perf_counter() - t) * 1000


def run(base: str, key: str, mode: str, concurrency: int, total: int) -> None:
    if mode == "gate":
        # 打健康就绪端点:它会真的去数据库跑一次查询,能压到连接池,
        # 但不触发模型调用。是"闸门 + 数据库"这段路径的代表。
        url, payload = f"{base}/api/health/ready", None
    else:
        url = f"{base}/api/chat"
        payload = {"thread_id": "loadtest", "message": "1公顷等于多少亩?"}

    headers = {"X-API-Key": key}
    codes: dict[int, int] = {}
    lat: list[float] = []

    print(f"压测 {mode}:{total} 个请求,并发 {concurrency} → {url}")
    t0 = time.perf_counter()
    with httpx.Client(timeout=120) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(_one, client, url, headers, payload) for _ in range(total)]
            done = 0
            for f in as_completed(futs):
                code, ms = f.result()
                codes[code] = codes.get(code, 0) + 1
                lat.append(ms)
                done += 1
                if done % max(1, total // 10) == 0:
                    print(f"  {done}/{total}", flush=True)
    dt = time.perf_counter() - t0

    ok = sum(v for k, v in codes.items() if 200 <= k < 300)
    print(f"\n耗时 {dt:.1f}s   吞吐 {total / dt:.1f} req/s   成功 {ok}/{total}")
    print("状态码分布:", dict(sorted(codes.items())))
    print(
        f"延迟(ms):p50 {_percentile(lat, 50):.0f}  p90 {_percentile(lat, 90):.0f}  "
        f"p99 {_percentile(lat, 99):.0f}  max {max(lat):.0f}  平均 {statistics.fmean(lat):.0f}"
    )
    # 429/503 不是"失败",是【闸门在正常工作】—— 分开报,否则会把"成功限流"
    # 误读成"系统挂了",进而去优化一个根本不存在的问题。
    if codes.get(429) or codes.get(503):
        print(
            f"  ↑ 其中 429(限流){codes.get(429, 0)} 个、503(容量){codes.get(503, 0)} 个 —— "
            "这是闸门在按设计拦截,不是故障"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1")
    ap.add_argument("--key", default="smoke-key")
    ap.add_argument("--mode", choices=["gate", "real"], default="gate")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--requests", type=int, default=1000)
    a = ap.parse_args()
    run(a.base, a.key, a.mode, a.concurrency, a.requests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
