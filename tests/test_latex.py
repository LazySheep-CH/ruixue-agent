"""行内 LaTeX → 文本。用例全部取自真实语料的采样。"""

from ruixue_agent.ingestion.utils.latex import latex_to_text, unwrap_inline_latex


def test_chemical_formula():
    assert latex_to_text(r"\mathrm { N H } _ { 3 }") == "NH3"
    assert latex_to_text(r"Z _ { \mathrm { n O } } / \mathrm { S i O } _ { 2 }") == "ZnO/SiO2"


def test_units_and_degree():
    assert latex_to_text(r"6 0 ^ { \circ } \mathrm { C }") == "60°C"
    assert latex_to_text(r"6 6 7 ~ \mathrm { m } ^ { 2 }") == "667 m2"  # ~ 是真空格
    assert latex_to_text(r"1 0 . 1 2 ~ \mathrm { g \cdot k g ^ { - 1 } }") == "10.12 g·kg-1"


def test_variables_and_ranges():
    assert (
        latex_to_text(r"\mathrm { W } 1 { > } \mathrm { W } 2 { > } \mathrm { W } 3") == "W1>W2>W3"
    )
    assert latex_to_text(r"\mathrm { B M R } _ { 4 5 }") == "BMR45"
    assert latex_to_text(r"0 \sim 8 0") == "0~80"


def test_italic_misrecognition_is_fixed():
    """MinerU 把斜体文字误判成公式 —— 转换后应还原成普通文字。"""
    assert latex_to_text(r"\it { ( M P }") == "(MP"


def test_unicode_mode():
    assert latex_to_text(r"\mathrm { N H } _ { 3 }", superscript="unicode") == "NH₃"
    assert (
        latex_to_text(r"1 0 . 1 2 ~ \mathrm { g \cdot k g ^ { - 1 } }", superscript="unicode")
        == "10.12 g·kg⁻¹"
    )


def test_unwrap_inside_a_paragraph():
    """整段正文里的 $...$ 都要被换掉。"""
    text = r"结果表明 $\mathrm { C a C O } _ { 3 }$ 的加入使温度达到 $6 0 ^ { \circ } \mathrm { C }$ 。"
    assert unwrap_inline_latex(text) == "结果表明 CaCO3 的加入使温度达到 60°C 。"


def test_broken_latex_does_not_crash():
    """怪公式不能炸掉整篇 —— 转不了就原样返回。"""
    assert latex_to_text(r"\begin{bad") is not None
