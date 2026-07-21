"""分词的测试 —— 这三个坑都是拿真实语料实测撞出来的。"""

from ruixue_agent.rag.text_segment import tokenize


def test_chinese_is_split():
    """核心:中文得切开,否则 PG 的 simple 配置把整句当一个词。"""
    out = tokenize("地膜厚度应不小于0.010mm")
    assert "地膜" in out.split()
    assert "厚度" in out.split()
    assert "0.010" in out.split()


def test_domain_terms_stay_whole():
    """领域术语不能被切碎。

    jieba 默认:"氧化生物双降解地膜" → 氧化/生物/双/降解/地膜
    切碎了就搜不到整词。词典是 BM25 在专业术语上的成败关键。
    """
    assert "全生物降解地膜" in tokenize("选用全生物降解地膜覆盖").split()
    assert "断裂标称应变" in tokenize("断裂标称应变纵横向均不小于300%").split()


def test_standard_code_stays_whole():
    """★ 标准号必须整体保留 —— "这膜符合 GB/T 35795 吗"是最高频查询之一。

    jieba 默认会把 "DB37/T2446-2013" 切成 DB37 / / / T2446 / - / 2013,彻底废掉。
    """
    assert "gb/t35795-2017" in tokenize("物理性能应符合 GB/T 35795-2017 的规定").split()
    assert "db37/t2446-2013" in tokenize("符合DB37/T2446-2013要求").split()


def test_standard_code_normalized():
    """写法不统一(有无空格、大小写)也要归一到同一个 token,否则搜不着。"""
    a = tokenize("符合 GB/T 35795 规定")
    b = tokenize("符合GB/T35795规定")
    c = tokenize("符合 gb/t 35795 规定")
    assert "gb/t35795" in a.split()
    assert a.split() == b.split() == c.split()


def test_latex_dropped():
    """LaTeX 公式切出来全是 $ \\ Delta m 这种噪音,对检索没价值。"""
    out = tokenize(r"每卷净质量偏差 $$\Delta m = m - m_{max}$$ 按式计算")
    assert "Delta" not in out and "$" not in out
    assert "质量" in out or "净" in out  # 正文还在


def test_english_and_numbers_survive():
    """夹在中文里的英文术语和数字 —— 这正是 simple 配置全线失败的地方。"""
    out = tokenize("主要原料为PBAT和PLA共混,厚度0.008mm").split()
    assert "pbat" in out and "pla" in out
    assert "0.008" in out


def test_superscript_normalized_to_plain():
    """★ Unicode 上标/下标归一化成 plain —— BM25 层的对齐要求。

    实测:g·kg-1 切成 "kg 1",g·kg⁻¹ 切成 "kg ¹" —— 完全不同,互相搜不到。
    语料里两种混着存(plain 4893 块 + unicode 5528 块),用户键盘只打得出 plain,
    所以统一成 plain。这样"120 g·kg⁻¹"和用户查的"120 g·kg-1"才能对上。
    """
    plain = tokenize("速效钾 120 g·kg-1").split()
    uni = tokenize("速效钾 120 g·kg⁻¹").split()
    assert plain == uni  # 两种写法切出来必须一样
    assert "1" in uni  # ⁻¹ 里的 ¹ 变成了普通的 1
    assert "²" not in tokenize("施氮量 300 kg/hm²")  # 上标 ² 不残留


def test_empty_and_punctuation():
    assert tokenize("") == ""
    assert tokenize("。,、;：!") == ""
