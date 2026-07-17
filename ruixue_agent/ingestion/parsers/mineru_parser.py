"""MinerU parser:把 MinerU 的 content_list.json 映射成我们的 Document。

MinerU(重工具,单独环境)先批量跑好,产出 *_content_list.json;
本模块只做纯映射:一条 MinerU 元素 → 一个我们的 Element(或丢弃)。无重依赖、可单测。
"""

from __future__ import annotations

from collections import Counter

from ruixue_agent.ingestion.schema import Document, Element

# MinerU 的页面家具类型 → 一律丢弃(它已帮我们分好类,不用再频率检测)
_FURNITURE = {"header", "footer", "page_number"}

# 遇到的未知类型计数 —— 批处理跑完打出来看,别让新类型被静默吞掉
UNKNOWN_TYPES: Counter[str] = Counter()


def _join(items: list | None) -> str:
    """把 caption / list_items 这种字符串列表拼成一段文本(空列表→空串)。"""
    return "\n".join(items) if items else ""


def _map_element(raw: dict) -> Element | None:
    """一条 MinerU 元素 → 一个 Element;页面家具或未知类型返回 None(丢弃)。"""
    mtype = raw["type"]
    page = raw["page_idx"]  # 每种元素都有 page_idx
    bbox = raw.get("bbox")  # 每种都有 bbox

    # 页面家具 → 丢弃
    if mtype in _FURNITURE:
        return None

    # ── 已写好的范例:text ──
    # MinerU 的 text 元素:有 text_level 就是标题,没有就是正文段落。
    if mtype == "text":
        level = raw.get("text_level")  # 没有这个键时返回 None
        etype = "heading" if level else "paragraph"
        return Element(type=etype, text=raw["text"], page=page, level=level, bbox=bbox)

    # ── 轮到你:照着上面的范例,填下面 5 个 ──

    # ① table → "table":text 存 table_body(那段 HTML);
    #        meta 里放 caption(table_caption)、footnote(table_footnote)、img_path
    # 注意用 .get() 不用 []:全量实测 7794 个表格里有 251 个(3.2%)【缺 table_body】——
    # MinerU 认出"这块是表格"但内容一个字没解析出来(img_path/caption 也全空)= 空壳 → 丢弃。
    # (3 篇测试样本里全都有 table_body,是全量跑才炸出来的:真实数据的长尾永远更野。)
    if mtype == "table":
        body = raw.get("table_body")
        if not body:
            return None
        return Element(
            type="table",
            text=body,
            page=page,
            bbox=bbox,
            meta={
                "caption": _join(raw.get("table_caption")),
                "footnote": _join(raw.get("table_footnote")),
                "img_path": raw.get("img_path"),
            },
        )

    # ② equation → "formula":text 存 LaTeX;同样防御式取(实测有 1 个缺 text)
    if mtype == "equation":
        latex = raw.get("text")
        if not latex:
            return None
        return Element(
            type="formula",
            text=latex,
            page=page,
            bbox=bbox,
            meta={
                "img_path": raw.get("img_path"),
                "text_format": raw.get("text_format"),
            },
        )

    # ③ image / chart → "figure":这俩都归 figure;
    #        text 存标题(image 用 image_caption,chart 用 chart_caption,拼接);meta 放 img_path
    if mtype in ("image", "chart"):
        caption = raw.get("image_caption") or raw.get("chart_caption")
        return Element(
            type="figure",
            text=_join(caption),
            page=page,
            bbox=bbox,
            meta={"img_path": raw.get("img_path")},
        )

    # ④ list → "list":text 把 list_items 用换行拼起来(_join);meta 放 sub_type
    if mtype == "list":
        return Element(
            type="list",
            text=_join(raw.get("list_items")),
            page=page,
            bbox=bbox,
            meta={"sub_type": raw.get("sub_type")},
        )

    # ⑤ page_footnote → "footnote":text 存 raw["text"](meta 不用放,默认空 {})
    if mtype == "page_footnote":
        return Element(
            type="footnote",
            text=raw["text"],
            page=page,
            bbox=bbox,
        )

    # 没见过的新类型 → 丢弃,但【记一笔】:静默丢弃会掩盖问题。
    # (全量跑才发现语料里还有 aside_text 72 个、code 13 个 —— 3篇样本里根本没有。)
    UNKNOWN_TYPES[mtype] += 1
    return None


def parse_content_list(content_list: list[dict], meta: dict) -> Document:
    """整份 content_list.json → 一个 Document。meta 来自 raw_manifest 的一行。"""
    elements = [el for raw in content_list if (el := _map_element(raw)) is not None]
    return Document(
        document_id=meta["document_id"],
        filename=meta["original_filename"],
        file_type="pdf",
        sha256=meta["sha256"],
        source=meta["source"],
        parser="mineru",
        elements=elements,
    )
