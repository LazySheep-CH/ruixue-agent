from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 中文 embedding 模型
_model = None  # 懒加载单例的占位


def get_model() -> SentenceTransformer:
    """懒加载单例：第一次用才加载模型（贵，~100MB），之后复用同一个。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# 一次喂给 GPU 多少条。
# sentence-transformers 默认 32 —— 太小,GPU 大部分时间在等数据而不是算。
# 实测(RTX 3090,真实子块,平均 171 字):
#     batch_size= 32(默认):   874 条/秒
#     batch_size=256       :  1247 条/秒   ← 快 43%
#     batch_size=512       :  1186 条/秒   ← 反而慢了,不是越大越好
# 256 是实测出来的,不是拍的。换了卡或换了模型要重新量。
_BATCH_SIZE = 256


def embed(texts: list[str]):
    """把一批文本编码成向量（已归一化，可直接用点积算余弦相似度）。"""
    return get_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=_BATCH_SIZE,
        show_progress_bar=False,
    )


if __name__ == "__main__":
    import numpy as np

    texts = ["PBAT地膜的水蒸气阻隔性", "生物降解薄膜的透水性", "今天天气很好"]
    vecs = embed(texts)
    print("句1 vs 句2(都讲膜):", float(np.dot(vecs[0], vecs[1])))  # 应该【高】
    print("句1 vs 句3(天气) :", float(np.dot(vecs[0], vecs[2])))  # 应该【低】
