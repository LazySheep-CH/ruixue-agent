"""chunk 层测试:章节路径栈算法(嵌套 + 同级弹栈)+ 父子分块。"""

from ruixue_agent.ingestion.stages.chunk import _section_paths, chunk_document
from ruixue_agent.ingestion.schema import Document, Element


def _el(type_, text, level=None):
    return Element(type=type_, text=text, page=0, level=level)


def test_section_paths_nesting_and_sibling_pop():
    els = [
        _el("heading", "标题", 1),
        _el("heading", "1 方法", 2),
        _el("paragraph", "用了A"),
        _el("heading", "1.1 细节", 3),  # 更深 → 嵌进 1 方法 下
        _el("paragraph", "细节X"),
        _el("heading", "2 结果", 2),  # 同级(L2)→ 弹掉 1.1 和 1 方法
        _el("paragraph", "得到B"),
    ]
    paths = _section_paths(els)
    assert paths == [
        ["标题"],
        ["标题", "1 方法"],
        ["标题", "1 方法"],
        ["标题", "1 方法", "1.1 细节"],  # 3 层嵌套建对
        ["标题", "1 方法", "1.1 细节"],
        ["标题", "2 结果"],  # "2 结果"来时弹掉了 1.1 和 1 方法
        ["标题", "2 结果"],
    ]


def test_chunk_document_parent_child():
    doc = Document(
        document_id="d",
        filename="f.pdf",
        file_type="pdf",
        sha256="h",
        source="期刊论文",
        parser="mineru",
        # 注意:正文长度须 ≥10 字(过短会被垃圾过滤器剔除)——
        # 子块垃圾过滤会把 <10 字的当噪声丢掉(真实语料里 <10 字的 100% 是垃圾)
        elements=[
            _el("heading", "1 方法", 2),
            _el("paragraph", "采用挤出复合和吹膜的方法制备复合薄膜"),
            _el("paragraph", "用SEM、XRD、DSC对薄膜进行表征分析"),
            _el("heading", "2 结果", 2),
            _el("paragraph", "结果表明覆膜显著提高地温二到三摄氏度"),
            _el("paragraph", "  "),  # 空元素 → 应被丢弃
        ],
    )
    chunks = chunk_document(doc)
    parents = [c for c in chunks if c.parent_id is None]
    children = [c for c in chunks if c.parent_id is not None]

    assert len(parents) == 2  # 两节 → 两个父块
    assert len(children) == 3  # 3 段有效正文 → 3 个子块(空的被丢)
    assert "挤出复合" in parents[0].text and "SEM" in parents[0].text  # 父块=整节
    assert parents[0].section_path == ["1 方法"]
    # 前两个子块都指向第一个父块
    assert children[0].parent_id == parents[0].chunk_id
    assert children[1].parent_id == parents[0].chunk_id


def _doc(elements):
    return Document(
        document_id="d",
        filename="f.pdf",
        file_type="pdf",
        sha256="h",
        source="期刊论文",
        parser="mineru",
        elements=elements,
    )


def test_long_paragraph_is_split():
    """长段落必须切开,否则 embedding 会悄悄截断、尾巴检索不到。"""
    long_text = (
        "生物降解地膜覆盖后地温提高显著,棉花出苗率提升明显。" * 30
    )  # 远超 400 字
    chunks = chunk_document(
        _doc([_el("heading", "2 结果", 2), _el("paragraph", long_text)])
    )
    children = [c for c in chunks if c.parent_id]
    assert len(children) > 1  # 被切成多段
    assert all(len(c.text) <= 400 for c in children)  # 每段都不超长


def test_junk_chunks_are_filtered():
    """垃圾子块不进向量库。用例全部取自真实语料采样(实测 [0-10)字 100%是垃圾)。"""
    junk = [
        "文献标志码:A",
        "中图分类号: S682. 32",
        "收稿日期 2024-06-05",
        "(责任编辑:梁俊红)",
        "(下转第64页)",
        "同2.1.6",
        "其中",
        "   ",
    ]
    doc = _doc([_el("heading", "2 结果", 2)] + [_el("paragraph", t) for t in junk])
    chunks = chunk_document(doc)
    assert [c for c in chunks if c.parent_id] == []  # 一个子块都不该产出


def test_figure_captions_survive():
    """不能误伤图注:它们 10~20 字,有检索价值(实测这个区间有三成是图注)。"""
    doc = _doc(
        [
            _el("heading", "2 结果", 2),
            _el("figure", "(a) 不同处理番茄叶片 SPAD 值"),  # 18字
            _el("figure", "图 2 不同种植模式对玉米农艺性状的影响"),  # 19字
        ]
    )
    kids = [c.text for c in chunk_document(doc) if c.parent_id]
    assert len(kids) == 2  # 两个图注都留下了
    assert "SPAD" in kids[0]


def test_table_becomes_row_sentences():
    """表格 → 行级人话子块(不是原样一块 HTML)。

    实测:84% 的表格 HTML 超 400 字会被 embedding 截断,而且模型看不懂 <td> 标签。
    转成每行一句自带表名+列名的人话后:超长率 84%→2.2%,而且"全杆的纤维素含量"能搜到了。
    """
    html = (
        "<table>"
        "<tr><td></td><td>纤维素</td><td>灰分</td></tr>"
        "<tr><td>全杆</td><td>31.16</td><td>2.4</td></tr>"
        "<tr><td>外壳</td><td>33.79</td><td>1.7</td></tr>"
        "</table>"
    )
    el = _el("table", html)
    el.meta["caption"] = "表1 三种试样成分比较"
    kids = [
        c.text
        for c in chunk_document(_doc([_el("heading", "3 表", 2), el]))
        if c.parent_id
    ]

    assert len(kids) == 2  # 两个数据行 → 两句
    assert kids[0] == "表:表1 三种试样成分比较。全杆:纤维素 31.16,灰分 2.4。"
    assert "<td>" not in kids[0]  # 标签清干净了
    assert "外壳" in kids[1] and "33.79" in kids[1]


def test_table_cells_latex_is_cleaned():
    """表格单元格里的行内 LaTeX 也要转 —— clean 阶段跳过了表格,这些一直没被清过。"""
    html = (
        "<table>"
        "<tr><td>处理</td><td> $\\mathrm { P } _ { 2 } 0 _ { 5 }$ </td></tr>"
        "<tr><td>NPK</td><td>750</td></tr>"
        "</table>"
    )
    kids = [
        c.text
        for c in chunk_document(_doc([_el("heading", "3 表", 2), _el("table", html)]))
        if c.parent_id
    ]
    assert "P205" in kids[0] or "P2O5" in kids[0]  # LaTeX 变人话了
    assert "\\mathrm" not in kids[0]
