"""造评测集:生成 → 裁判复审 → 只留合格的。

为什么要合成:系统还没上线,没有真实用户日志。企业在这个阶段的标准做法就是
合成 + 严格质检,等上线后用真实 query 逐步替换。

三个阶段(仅有生成、缺质检的题集不可用,复审与标注补全同样必要):

  ① 生成 —— 按【题型】× 【角色】出题,不是随便问
     题型:覆盖瑞雪用户真会问的六类(见 _KINDS)
     角色:研发工程师 / 技术服务 / 质检采购 —— 问法完全不同
           研发问"PBAT和PLA共混比例对断裂伸长率的影响"
           技术服务问"客户地里膜提前烂了是咋回事"

  ② 裁判复审 —— 换个 LLM 视角逐题判,四关全过才留:
     可答性  这段【真能】回答这个问题吗?         ← 第一版从没验过,地基都是虚的
     唯一性  换一篇文献的类似段落能不能也答?     ← 能 → 标准答案失效
     自足性  单独拿出来看得懂吗?
     真实性  真实用户会这么问吗?                 ← 毙掉"为什么公式里有笔误"这种

  ③ 落盘 —— 带上题型/角色标签,后面能【分组看】:
     哪类题最差?数值查询?还是因果解释?
     一个总分 0.787 什么也告诉不了你,分组才知道该修哪儿。

注意:仍然存在的偏差(合成评测集的天花板,必须知道):
    - 出题时 LLM 看着原文,措辞会向原文靠 → 分数偏高
    - 仅标一个 gold chunk 时,其他同样能回答的段落会被误判为未命中
    → 所以这个基线用于【对比】(A/B 谁好),不用于【对外报告绝对水平】。
      真实问题还是得领域专家出,这个先顶着。

用法:
    uv run python scripts/build_evalset.py --n 150
    uv run python scripts/build_evalset.py --n 150 --workers 8
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import text

from ruixue_agent.models import create_model
from ruixue_agent.persistence.engine import get_engine

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path("data/eval/evalset.jsonl")
REJECTED = Path("data/eval/rejected.jsonl")  # 毙掉的也留账,能看出题机器哪儿不行

# ── 题型:瑞雪的用户真会问的几类 ────────────────────────────
# 不是我拍的分类法 —— 是照着这个语料库里【真实存在】的内容归的:
# 1712 篇论文(材料/配方/降解/农艺试验)+ 37 份标准(指标/方法)。
_KINDS = [
    (
        "数值查询",
        "问一个具体的指标、参数或数值。答案应该是个数或一个明确的范围。",
    ),
    (
        "标准符合",
        "问某项要求、限值或试验方法 —— 标准/规程里规定了什么。",
    ),
    (
        "因果解释",
        "问【为什么】会这样、机理是什么。答案是一段解释,不是一个数。",
    ),
    (
        "条件推荐",
        "在某种作物/气候/土壤条件下,该选什么、该怎么做。",
    ),
    (
        "对比差异",
        "两种材料/处理/方案之间的差别。",
    ),
    (
        "工艺操作",
        "怎么做 —— 覆膜、回收、加工、测试的具体做法或步骤。",
    ),
]

_PERSONAS = [
    ("研发工程师", "关心材料、配方、性能机理。用词专业,会直接说材料牌号和指标名。"),
    (
        "技术服务",
        "关心地里的实际问题。说人话,不用学术腔,常从现象出发(膜烂了、不降解、烧苗)。",
    ),
    ("质检采购", "关心是否合规、指标够不够、怎么测。会提到标准号和验收要求。"),
]

_GEN = """你是地膜行业的{persona}。{persona_desc}

下面是内部文献库中的一段内容。请以你的身份,提出 1 个【{kind}】类的问题。
{kind_desc}

硬性要求:
1. 【这段内容必须真能回答它】—— 这是最重要的一条
2. 【唯一性】问题要具体到只有这一段能答。换一篇文献的类似段落也能答的话,
   就把限定词写进去(具体材料、具体条件、具体作物)。
   反例:"生产原纸的厂家是哪家?"  ← 废题,任何一篇的试剂表都能答
3. 【自足】不能有"本文""该研究""上述"这类指代,单独拿出来要看得懂
4. 【像人问的】用你这个身份平时的说法,不要照抄原文句子

如果这段内容出不出这一类的好题(内容是目录/致谢/参考文献/纯表头,
或者跟【{kind}】这个类型对不上),只回复:SKIP

只输出问题本身。

内容:
\"\"\"
{text}
\"\"\""""

_JUDGE = """你是评测集质检员。判断下面这道题够不够格进入检索评测集。

问题:{question}

标准答案段落:
\"\"\"
{text}
\"\"\"

逐条判断:
A. 可答性:光看这段内容,能不能回答这个问题?(答不了 = 这题的标准答案是错的)
B. 唯一性:这题是不是【只有】这类特定内容能答?如果地膜领域随便另一篇文献的
   类似段落也能答(比如"试剂厂家是哪家""厚度是多少"这种没有限定条件的),判否。
C. 自足性:问题单独拿出来,不看原文,能不能看懂?
D. 真实性:地膜行业的从业者会真的这么问吗?
   (反例:"为什么公式里有个笔误" —— 这是校对,不是提问)

四条【全部】通过才算合格。

只输出 JSON,不要别的:
{{"ok": true/false, "fail": "A/B/C/D 里没过的那条,全过就填空字符串", "why": "一句话"}}"""


def _one(llm, judge_llm, row, kind, persona) -> tuple[dict | None, dict | None]:
    """出一题 + 裁判复审。→ (合格的题, 被毙的题) 只会有一个不是 None。"""
    kind_name, kind_desc = kind
    p_name, p_desc = persona
    try:
        q = llm.invoke(
            _GEN.format(
                persona=p_name,
                persona_desc=p_desc,
                kind=kind_name,
                kind_desc=kind_desc,
                text=row["text"],
            )
        ).content.strip()
    except Exception as e:
        return None, {
            "stage": "生成",
            "error": str(e)[:80],
            "chunk_id": row["chunk_id"],
        }

    if "SKIP" in q.upper() or len(q) < 8:
        return None, {"stage": "生成", "fail": "SKIP", "chunk_id": row["chunk_id"]}

    # ── 裁判复审 ──
    try:
        raw = judge_llm.invoke(_JUDGE.format(question=q, text=row["text"])).content
        raw = raw[raw.find("{") : raw.rfind("}") + 1]  # 剥掉可能的 ```json 包裹
        verdict = json.loads(raw)
    except Exception as e:
        return None, {"stage": "复审", "error": str(e)[:80], "question": q}

    if not verdict.get("ok"):
        return None, {
            "stage": "复审",
            "fail": verdict.get("fail"),
            "why": verdict.get("why"),
            "question": q,
            "kind": kind_name,
        }

    return (
        {
            "question": q,
            # primary_gold = 出这道题的那个 chunk。
            # gold_chunk_ids 由 pool_evalset.py 补全 —— 库里可能还有别的段落同样能答,
            # 仅标一个会把检索到的其他正确答案误判为未命中。
            "primary_gold": row["chunk_id"],
            "gold_chunk_ids": [row["chunk_id"]],
            "kind": kind_name,  # ← 带标签,后面能分组看哪类题最差
            "persona": p_name,
            "gold_text": row["text"][:200],
            "section_path": row["section_path"],
            "title": row["title"],
            "year": row["year"],
            "source": row["source"],
        },
        None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument(
        "--workers", type=int, default=8, help="并发数 —— 串行 150 题要 30 分钟"
    )
    ap.add_argument("--model", default="deepseek-v4-flash")
    args = ap.parse_args()

    random.seed(42)  # 可复现:同种子 → 同样本 → 两次评测能对比
    llm = create_model(args.model)
    judge = create_model(args.model)

    # ── 分层抽样(stratified sampling),不是随机抽 ──
    #
    # 随机抽样的问题:标准规范仅占语料 2%,随机抽题时只占到 8/150。
    # 为什么?语料是 1712 篇论文 + 37 份标准,标准天然只占 2%。
    # 但业务上,"这膜符合 GB/T 35795 吗"比"降解机理"重要得多 ——
    # 【随机抽样 ≠ 按业务重要性抽样】。语料的分布不等于问题的分布。
    #
    # 所以按 source 配额抽:标准规范虽然只有 37 份,但要占 30% 的题。
    # 这个比例是【业务判断】,不是数据算出来的 —— 该由你按瑞雪的真实需求定。
    #
    # 长度门槛也按 source 分别设定,原因:
    #   第一版对两者都用 200~1200 字。结果标准的 1047 个父块只有 126 个够格(12%),
    #   被扔掉的 921 个里全是这种:
    #     "生物降解地膜主要原料为聚己二酸对苯二甲酸丁二醇酯(PBAT)…"
    #     "起垄栽培,垄高20.00 cm,垄宽50.00 cm"
    #     "宜选用厚度为 0.010 mm 的地膜。"
    #   —— 恰恰是瑞雪的人最会问的东西。
    #
    #   为什么会这样:标准这种文体就是【短条款】,一句话一节。
    #   200字门槛是照着论文的形状定的,拿它量标准 = 把标准最典型的形态整批砍掉。
    #   (和早期"只数 type=='text' 的字数"把表格型实测报告判成垃圾,是同一个错)
    #
    #   短块检索难度确实更大(文本短 → 向量信号弱),但那是【真实的系统弱点】,
    #   该量出来,不该靠筛掉题目来假装不存在。
    quota = {
        # source: (占比, 最短, 最长)
        "标准规范": (0.30, 60, 1200),
        "期刊论文": (0.70, 200, 1200),
    }
    rows = []
    with get_engine().connect() as conn:
        for src, (share, lo, hi) in quota.items():
            got = (
                conn.execute(
                    text("""
                SELECT c.chunk_id, c.text, c.section_path, d.title, d.year, d.source
                FROM chunks c JOIN documents d ON d.document_id = c.document_id
                WHERE c.kind = 'parent' AND d.source = :src
                  AND length(c.text) BETWEEN :lo AND :hi
                ORDER BY random() LIMIT :lim
            """),
                    # ×5:实测通过率约 30%(出题机 SKIP 一批,裁判再毙一批),得备够候选
                    {"src": src, "lo": lo, "hi": hi, "lim": int(args.n * share * 5)},
                )
                .mappings()
                .all()
            )
            rows.extend(dict(r) for r in got)
            print(
                f"  {src}: 候选 {len(got)} 段(目标占比 {share:.0%},长度 {lo}~{hi} 字)"
            )
    random.shuffle(rows)  # 打散,免得先跑完标准再跑论文导致配额失衡

    # 题型 × 角色 轮着配 —— 保证六类题都有,不是全挤在"数值查询"
    tasks = [
        (r, _KINDS[i % len(_KINDS)], _PERSONAS[i % len(_PERSONAS)])
        for i, r in enumerate(rows)
    ]
    print(f"\n候选共 {len(tasks)} 段,目标 {args.n} 题,并发 {args.workers}\n")

    t0 = time.time()
    kept, rejected = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ok, bad in pool.map(lambda t: _one(llm, judge, *t), tasks):
            if ok:
                kept.append(ok)
                if len(kept) % 20 == 0:
                    print(
                        f"  合格 {len(kept)}/{args.n}  (毙掉 {len(rejected)})  {time.time() - t0:.0f}s"
                    )
            elif bad:
                rejected.append(bad)
            if len(kept) >= args.n:
                break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with REJECTED.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'═' * 66}")
    print(f"合格 {len(kept)} 题 → {OUT}")
    print(
        f"毙掉 {len(rejected)} 题 → {REJECTED}  (通过率 {len(kept) / max(len(kept) + len(rejected), 1) * 100:.0f}%)"
    )
    print(f"耗时 {time.time() - t0:.0f}s")

    from collections import Counter

    print("\n题型分布:")
    for k, v in Counter(r["kind"] for r in kept).most_common():
        print(f"  {k:8}: {v:3}")
    print("角色分布:")
    for k, v in Counter(r["persona"] for r in kept).most_common():
        print(f"  {k:8}: {v:3}")
    print("\n毙掉的原因(裁判在毙什么 —— 看这个就知道出题机器哪儿不行):")
    for k, v in Counter(
        r.get("fail") or r.get("stage") for r in rejected
    ).most_common():
        print(f"  {k}: {v}")

    print(f"\n{'═' * 66}\n抽 4 题人工过目(别信没看过的数据):")
    for r in random.sample(kept, min(4, len(kept))):
        print(f"\n  [{r['kind']}/{r['persona']}] {r['question']}")
        print(f"    出处: {' > '.join((r['section_path'] or [])[:2])[:66]}")


if __name__ == "__main__":
    main()
