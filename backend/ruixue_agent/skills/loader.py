"""技能(skills):把已验证的操作流程 SOP沉淀成 Markdown,按需加载给 agent。

技能 vs 工具:
    工具 = 一个原子动作(查土壤、跑模型、检索知识)
    技能 = 一套流程:什么场景、先用哪个工具、再用哪个、结果怎么解读、有哪些坑

工具是"能做什么",技能是"该怎么做"。领域经验(比如"选配方要先看生育期长度,
降解率不能在生育期内过高,否则提前破膜")写进代码是浪费 —— 它会反复迭代,
且需要领域专家而非程序员来改。做成 Markdown 文件:改流程不动代码。

为什么"按需加载"而不是全塞进系统提示:
技能会越攒越多,全塞进系统提示会:1)烧 token 2)稀释注意力,反而让模型抓不住重点。
故按场景关键词匹配,只把相关的那一两条注入。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

logger = logging.getLogger("ruixue.skills")

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


@dataclass(frozen=True)
class Skill:
    name: str  # 技能名(取自文件名)
    triggers: tuple[str, ...]  # 触发关键词(命中即认为该场景相关)
    body: str  # 正文:SOP 步骤

    def matches(self, text: str) -> bool:
        return any(t in text for t in self.triggers)


def _parse(path: Path) -> Skill | None:
    """解析技能文件。首行 `# 名称`,其后 `triggers: a, b, c`,空行后为正文。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("技能 %s 读取失败(%s),跳过", path.name, type(e).__name__)
        return None
    m = re.search(r"^triggers:\s*(.+)$", raw, re.MULTILINE)
    if not m:
        logger.warning("技能 %s 缺少 triggers 行,跳过", path.name)
        return None
    triggers = tuple(t.strip() for t in m.group(1).split(",") if t.strip())
    body = raw[m.end() :].strip()
    return Skill(name=path.stem, triggers=triggers, body=body)


@cache
def load_skills() -> tuple[Skill, ...]:
    """加载 skills/ 下的全部技能(进程内缓存)。目录不存在则为空,不报错。"""
    if not SKILLS_DIR.is_dir():
        return ()
    skills = tuple(s for p in sorted(SKILLS_DIR.glob("*.md")) if (s := _parse(p)))
    logger.info("已加载 %d 个技能:%s", len(skills), [s.name for s in skills])
    return skills


def select_skills(user_text: str, limit: int = 2) -> list[Skill]:
    """按用户输入匹配相关技能,最多 limit 条(防止塞爆上下文)。"""
    return [s for s in load_skills() if s.matches(user_text)][:limit]


# 每条技能在上下文里的标题格式。中间件靠它判断"这条之前注入过没有",
# 所以它是约定好的标记,不能随手改 —— 改了要同步改 SKILL_HEADER_RE。
SKILL_HEADER = "【作业规程:{name}】"
SKILL_HEADER_RE = re.compile(r"【作业规程:(.+?)】")


def render(skills: list[Skill]) -> str:
    """把给定的技能渲染成可注入上下文的文本;空列表返回空串。

    和 render_skills 的区别:这个函数不负责挑选,由调用方决定注入哪几条 ——
    中间件需要"挑出匹配的,再剔掉已经注入过的",所以挑选和渲染必须能分开。
    """
    if not skills:
        return ""
    parts = [f"{SKILL_HEADER.format(name=s.name)}\n{s.body}" for s in skills]
    return "以下是本领域已验证的作业规程,请按其步骤与注意事项作答:\n\n" + "\n\n".join(parts)


def injected_names(text: str) -> set[str]:
    """从一段已注入的文本里,反查出它包含了哪几条技能(按标题解析)。"""
    return set(SKILL_HEADER_RE.findall(text))


def render_skills(user_text: str, limit: int = 2) -> str:
    """按用户输入匹配并渲染;没匹配到返回空串。(便捷入口,不做去重)"""
    return render(select_skills(user_text, limit))
