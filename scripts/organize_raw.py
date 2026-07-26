"""把源 PDF 按类型归类 + 规范文件名 → data/raw/{literature,standards,uncertain}/。

默认 dry-run 只报告分类结果,不动文件;加 --execute 才真正复制。
可重复运行(幂等):以 sha256 去重,已在清单里的跳过 → 以后新增文件直接再跑即可。

用法:
  预演:  uv run python scripts/organize_raw.py
  执行:  uv run python scripts/organize_raw.py --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _ext(p: Path) -> str:
    r"""Windows 长路径(>260字符)绕过:加 \\?\ 扩展长度前缀。"""
    return "\\\\?\\" + str(p)


SRC = Path(r"D:/LLM_RuiXue/RuiXue-Intelligent-Mulch-System/storage/data/raw/literature/mulch-pdf")
DST = Path(__file__).resolve().parent.parent / "data" / "raw"
MANIFEST = DST / "raw_manifest.jsonl"

# ── 分类规则(标准/规范的信号)──────────────────────────────
# 强信号 → 判定为 standards(基本不会误伤论文)
STD_NUMBER = re.compile(
    r"(?<![0-9A-Za-z])(?:"  # 前瞻:前缀前面不能是字母/数字(否则哈希里的 DB1880 会误判)
    r"GB[ /]?T?[ ]?\d{3,}"  # GB 12345 / GB/T 12345 / GBT 35795
    r"|DB[ ]?\d{2}[ /T+]*\d{2,}"  # DB15/T 2525 / DB15T+2525 / DB5331/T 12
    r"|NY[ /]?T[ ]?\d{3,}"  # NY/T 4540 / NYT4540
    r"|ISO[ /]?\d{3,}"  # ISO 17088
    r"|T[ /][A-Z]{2,}[ ]?\d{2,}"  # 团标 T/CNTAC 55
    r")"
)  # 注意:不用 re.I —— 标准号前缀都是大写,加了 re.I 会把英文小写 t 到处误匹配


def _norm(name: str) -> str:
    """分类前归一化:全角 ／＋ → 半角(不改真实文件名,只用于匹配)。"""
    return name.replace("／", "/").replace("＋", "+")


STD_STRONG_WORD = re.compile(r"规程")  # 论文几乎不会自称"规程"
BOOK_TITLE_STD = re.compile(r"《[^》]*(规范|规程|标准|技术要求)[^》]*》")
FDIS = re.compile(r"\bFDIS\b", re.I)  # 标准草案终稿
# 弱信号 → 判定为 uncertain(含这些词但没强信号,交人工确认)
STD_WEAK_WORD = re.compile(r"规范|标准|技术要求|通则|导则")


# 人工复核覆盖(2026-07-14):对 uncertain 逐个人工判定,决策留档、可审计。
# 未来新增文件若命中弱信号仍会进 uncertain,等待下一次人工复核。
OVERRIDES: dict[str, str] = {
    # → 标准(团标/技术规范)
    "1967茄果类蔬菜全生物降解地膜覆盖栽培技术规范.pdf": "standards",
    "TJAASS 157-2024 TAHAASS 011—2024 大豆玉米带状复合种植覆膜生产技术规范.pdf": "standards",
    # → 论文(标题含"标准/规范"字样,但实为研究/综述)
    "中国农田地膜残留等级划分标准探析与构建_高海河.pdf": "literature",
    "国内外农用地膜使用政策、执行标准与回收状况_靳拓.pdf": "literature",
    "实现地膜供应链数字化助力农业绿色健康发展——GS1标准在农用地膜追溯领域的应用实践.pdf": "literature",
    "河南省高标准农田背景下的地膜处理与高效管理.pdf": "literature",
    "湖北省羊肚菌设施化、规范化和精细化栽培技术.pdf": "literature",
    "绿色发展背景下的耕地生态补偿标准量化研究——以广东市域为例.pdf": "literature",
    "聚乙烯吹塑农用地面覆盖薄膜标准比对分析.pdf": "literature",
    "高标准农田建设中节水灌溉技术的应用研究.pdf": "literature",
    "黔东地区仙草规范化栽培技术探讨.pdf": "literature",
}


def classify(name: str) -> tuple[str, str]:
    """返回 (类别, 理由)。类别 ∈ {standards, literature, uncertain}。"""
    if name in OVERRIDES:
        return OVERRIDES[name], "人工复核 override"
    name = _norm(name)
    if STD_NUMBER.search(name):
        return "standards", "强信号:标准号"
    if BOOK_TITLE_STD.search(name):
        return "standards", "强信号:《…规范/规程…》"
    if STD_STRONG_WORD.search(name):
        return "standards", "强信号:含'规程'"
    if FDIS.search(name):
        return "standards", "强信号:FDIS 标准草案"
    if STD_WEAK_WORD.search(name):
        return "uncertain", "弱信号:含'规范/标准'字样,需人工确认"
    return "literature", "无标准信号→论文"


# ── 文件名规范化 ──────────────────────────────────────────
def sanitize(name: str) -> str:
    """清理非法/碍事字符,但保留可读的中文标题(不改成 lit_0001 那种看不懂的 ID)。"""
    stem = Path(name).stem
    stem = stem.replace("...", "").replace("…", "")
    stem = re.sub(r"[：:]", "-", stem)  # 全/半角冒号 → -
    stem = re.sub(r"[+＋]", "_", stem)  # 加号 → _
    stem = re.sub(r'[\\/:*?"<>|]', "", stem)  # Windows 非法字符去掉
    stem = re.sub(r"[\s_]+", "_", stem).strip("_-")  # 折叠空白/下划线
    return stem[:120] + ".pdf"  # 限长,防超长路径


def detect_lang(path: Path, pages: int = 3) -> str:
    """按内容判断语言:读前几页,算中文字符占比。中文论文占比高,英文论文近 0。"""
    import pymupdf

    doc = pymupdf.open(_ext(path))
    text = "".join(doc[i].get_text() for i in range(min(pages, len(doc))))
    doc.close()
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    ratio = cjk / max(1, cjk + latin)
    return "zh" if ratio >= 0.15 else "en"


def reclassify_language() -> None:
    """对已复制的 literature 文件按内容判语言,把英文的移到 data/raw/non_chinese/,重写清单。"""
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    moved = 0
    for r in rows:
        if r["category"] != "literature":
            r.setdefault("language", "zh")  # 标准都是中文
            continue
        src = DST / r["category"] / r["sanitized_filename"]
        lang = detect_lang(src)
        r["language"] = lang
        if lang != "zh":
            (DST / "non_chinese").mkdir(exist_ok=True)
            shutil.move(_ext(src), _ext(DST / "non_chinese" / r["sanitized_filename"]))
            r["category"] = "non_chinese"
            moved += 1
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    zh = sum(1 for r in rows if r["category"] == "literature")
    print(f"语言过滤完成:英文 {moved} 篇 → data/raw/non_chinese/;中文论文剩 {zh} 篇。清单已更新。")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(_ext(path), "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="真正复制文件;默认只预演")
    ap.add_argument(
        "--filter-lang", action="store_true", help="对已复制文件按内容过滤语言,英文移出"
    )
    args = ap.parse_args()

    if args.filter_lang:
        reclassify_language()
        return

    pdfs = sorted(SRC.rglob("*.pdf"))
    buckets: dict[str, list[Path]] = {
        "literature": [],
        "standards": [],
        "uncertain": [],
    }
    reasons: dict[str, str] = {}
    for p in pdfs:
        cat, why = classify(p.name)
        buckets[cat].append(p)
        reasons[p.name] = why

    total_mb = sum(os.stat(_ext(p)).st_size for p in pdfs) / 1024 / 1024
    print(f"源目录共 {len(pdfs)} 个 PDF,合计 {total_mb:.0f} MB")
    print(f"  论文 literature : {len(buckets['literature'])}")
    print(f"  标准 standards  : {len(buckets['standards'])}")
    print(f"  待定 uncertain  : {len(buckets['uncertain'])}")

    print("\n===== standards(自动判为标准)=====")
    for p in buckets["standards"]:
        print(f"  [{reasons[p.name]}] {p.name}  →  {sanitize(p.name)}")

    print("\n===== uncertain(需你人工确认)=====")
    for p in buckets["uncertain"]:
        print(f"  {p.name}  →  {sanitize(p.name)}")

    if not args.execute:
        print("\n(预演模式,未复制任何文件。确认无误后加 --execute 执行。)")
        return

    # ── 执行:复制 + sha256 去重 + 写清单 ──
    seen: set[str] = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            seen.add(json.loads(line)["sha256"])
    copied, skipped = 0, 0
    lines: list[str] = []
    for cat, plist in buckets.items():
        (DST / cat).mkdir(parents=True, exist_ok=True)
        for p in plist:
            digest = sha256_of(p)
            if digest in seen:
                skipped += 1
                continue
            seen.add(digest)
            new_name = sanitize(p.name)
            dest = DST / cat / new_name
            if dest.exists():  # 同名不同内容 → 加短哈希避免覆盖
                dest = DST / cat / f"{dest.stem}_{digest[:8]}.pdf"
            shutil.copy2(_ext(p), _ext(dest))
            copied += 1
            lines.append(
                json.dumps(
                    {
                        "document_id": digest[:16],
                        "sha256": digest,
                        "category": cat,
                        "source": {
                            "literature": "期刊论文",
                            "standards": "标准规范",
                            "uncertain": "待定",
                        }[cat],
                        "original_filename": p.name,
                        "sanitized_filename": new_name,
                        "source_path": str(p),
                        "size_bytes": os.stat(_ext(p)).st_size,
                    },
                    ensure_ascii=False,
                )
            )
    with open(MANIFEST, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(f"\n执行完成:复制 {copied},跳过(已存在){skipped}。清单 → {MANIFEST}")


if __name__ == "__main__":
    main()
