"""批量跑 MinerU:data/raw 的 PDF → D:/LLM_RuiXue/mineru-out/<类别>/<document_id>/auto/

设计要点:
  · 用 document_id(16字符)做临时硬链接名喂给 MinerU
      → 输出目录名 = document_id(短路径,绕开 Windows 260 上限;且天然对回 manifest)
  · 硬链接零拷贝(同盘 NTFS),不多占 4GB
  · 分批调用(模型每批只加载一次;单篇一次会反复加载,极慢)
  · 断点续跑:已有输出的跳过 → 中断了直接再跑一次即可
  · 打印进度 / 速度 / 预计剩余时间

用法: uv run python scripts/run_mineru_batch.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJ = Path(__file__).resolve().parent.parent
MINERU = Path(r"D:/LLM_RuiXue/mineru-tool/.venv/Scripts/mineru.exe")
OUT_ROOT = Path(r"D:/LLM_RuiXue/mineru-out")
TMP_ROOT = Path(r"D:/LLM_RuiXue/_mineru_tmp")  # 必须和 data/raw 同盘(D:),硬链接才成立
BATCH = 40


def load_id_map() -> dict[str, str]:
    """从 raw_manifest 读 {sanitized_filename: document_id}。"""
    mf = PROJ / "data" / "raw" / "raw_manifest.jsonl"
    rows = [json.loads(line) for line in mf.read_text(encoding="utf-8").splitlines()]
    return {r["sanitized_filename"]: r["document_id"] for r in rows}


def is_done(out_dir: Path, did: str) -> bool:
    """已完成 = 该文档目录下【任意】子目录里有 content_list.json。

    注意:MinerU 的输出子目录名 = 它实际用的方法(auto/ 或 txt/ 或 ocr/)。
    不能写死 'auto':-m txt 模式的输出目录名不同,会被误判为未解析。
    """
    return any((out_dir / did).glob("*/*_content_list.json"))


def fmt(sec: float) -> str:
    h, m = int(sec // 3600), int(sec % 3600 // 60)
    return f"{h}小时{m}分" if h else f"{m}分{int(sec % 60)}秒"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="auto=让MinerU自己判断(默认);txt=强制文本模式(auto挂死时用它救)",
    )
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = (
        "huggingface"  # 模型已缓存;走这条是因为国内 modelscope 被代理掐
    )
    env["HTTP_PROXY"] = env["HTTPS_PROXY"] = "http://127.0.0.1:10808"
    env["PYTHONUTF8"] = "1"

    id_map = load_id_map()
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    # 收集待办:(类别, 源pdf, document_id)
    todo = []
    for cat in ("literature", "standards"):
        out = OUT_ROOT / cat
        out.mkdir(parents=True, exist_ok=True)
        pdfs = sorted((PROJ / "data" / "raw" / cat).glob("*.pdf"))
        pending = []
        for p in pdfs:
            did = id_map.get(p.name)
            if not did:
                print(f"  注意:manifest 里找不到 {p.name[:40]},跳过")
                continue
            if not is_done(out, did):  # 断点续跑:已有产出就跳过
                pending.append((cat, p, did))
        print(f"{cat}: 共 {len(pdfs)} 篇,待跑 {len(pending)}")
        todo += pending

    total = len(todo)
    if not total:
        print("\n全部已完成 ✅")
        return
    print(f"\n合计待跑 {total} 篇,每批 {args.batch} 篇,method={args.method}。开始…\n")

    # 先按类别分组,再各自分批 —— 一批绝不能跨类别(输出目录不同)。
    # (曾经的 bug:直接切片会让一批跨两类,过滤后剩下的被无声丢弃。)
    by_cat: dict[str, list] = {}
    for cat, pdf, did in todo:
        by_cat.setdefault(cat, []).append((cat, pdf, did))

    batches = []
    for cat, items in by_cat.items():
        for i in range(0, len(items), args.batch):
            batches.append(items[i : i + args.batch])

    t0 = time.time()
    finished = 0
    for batch in batches:
        cat = batch[0][0]

        tmp = Path(tempfile.mkdtemp(prefix="b_", dir=str(TMP_ROOT)))
        try:
            for _, pdf, did in batch:
                os.link(pdf, tmp / f"{did}.pdf")  # 硬链接:零拷贝 + 短名
            subprocess.run(
                [
                    str(MINERU),
                    "-p",
                    str(tmp),
                    "-o",
                    str(OUT_ROOT / cat),
                    "-b",
                    "pipeline",
                    "-l",
                    "ch",
                    "-m",
                    args.method,
                ],
                env=env,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # MinerU 日志太吵,只看我们的进度
                timeout=60 * 60,  # 一批最多 1 小时;卡死的不能拖死整个批处理
            )
        except subprocess.TimeoutExpired:
            print("  注意:该批超时(1小时),跳过继续 —— 未产出的会在下次重跑时重试")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        finished += len(batch)
        elapsed = time.time() - t0
        per = elapsed / finished
        eta = per * (total - finished)
        pct = finished / total * 100
        print(
            f"[{finished}/{total}] {pct:5.1f}% | 已用 {fmt(elapsed)} | "
            f"{per:.1f}秒/篇 | 预计还需 {fmt(eta)}"
        )

    print(f"\n完成 ✅ 总耗时 {fmt(time.time() - t0)}")
    print(f"输出:{OUT_ROOT}")


if __name__ == "__main__":
    main()
