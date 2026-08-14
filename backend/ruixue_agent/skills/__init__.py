"""技能子系统:把已验证的作业流程(SOP)沉淀为 Markdown,按场景注入。"""

from ruixue_agent.skills.loader import (
    Skill,
    injected_names,
    load_skills,
    render,
    render_skills,
    select_skills,
)

__all__ = [
    "Skill",
    "injected_names",
    "load_skills",
    "render",
    "render_skills",
    "select_skills",
]
