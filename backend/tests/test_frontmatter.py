"""卷首提取:用例全部取自真实语料的野写法(实测 84.5% 有摘要,但写法有4种)。"""

from ruixue_agent.ingestion.utils.frontmatter import extract_abstract, extract_keywords


def test_case_a_mark_and_content_same_element():
    """情况A:标记和内容在同一元素,而且没有冒号(实测最常见)。"""
    texts = [
        "某某大学",
        "摘 要 为明确不同种类地膜在大棚西瓜的应用效果,以美都西瓜为材料",
        "关键词 地膜;西瓜",
    ]
    assert "为明确不同种类地膜" in extract_abstract(texts)


def test_case_b_mark_alone_content_in_next():
    """情况B:标记独占一个元素,内容在下一个。"""
    texts = ["摘要", "本试验探究了页岩土对沙漠表面硬度的改良作用", "关键词 沙漠;水泥"]
    assert "页岩土" in extract_abstract(texts)


def test_case_c_mark_in_the_middle():
    """情况C:标记挤在单位信息后面,不在元素开头 → 必须用 search 不是 match。"""
    texts = ["(1甘肃农业大学水利水电工程学院,兰州 730070)摘 要 为探明玉米覆膜效应"]
    assert "为探明玉米覆膜效应" in extract_abstract(texts)


def test_case_d_fullwidth_brackets_and_tags():
    """情况D:全角方括号 + sub 标签包着的冒号。"""
    texts = ["［摘 要］ 采用水稻移栽覆膜技术,通过测定4个处理土壤样本"]
    assert "采用水稻移栽覆膜技术" in extract_abstract(texts)
    kw = extract_keywords(["［关键词］ 地膜覆盖<sub>;</sub>有机肥<sub>;</sub>水稻"])
    assert kw == ["地膜覆盖", "有机肥", "水稻"]


def test_keywords_stop_at_next_mark():
    """关键词行后面常紧跟'中图分类号:...' → 要截断,不能吃进来。"""
    kw = extract_keywords(["关键词 双通道;YOLOv8s;棉花地膜 中图分类号:U495 文献标志码:A"])
    assert kw == ["双通道", "YOLOv8s", "棉花地膜"]


def test_no_frontmatter_is_safe():
    """新闻/短文没有摘要标记 → 返回空,不报错(15.5% 的文档是这样)。"""
    assert extract_abstract(["春耕备耕早字当头", "今年以来各地积极行动"]) == ""
    assert extract_keywords(["春耕备耕早字当头"]) == []
