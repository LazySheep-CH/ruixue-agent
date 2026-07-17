from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 中文 embedding 模型
_model = None  # 懒加载单例的占位


def get_model() -> SentenceTransformer:
    """懒加载单例：第一次用才加载模型（贵，~100MB），之后复用同一个。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]):
    """把一批文本编码成向量（已归一化，可直接用点积算余弦相似度）。"""
    return get_model().encode(texts, normalize_embeddings=True)


if __name__ == "__main__":
    import numpy as np

    texts = ["PBAT地膜的水蒸气阻隔性", "生物降解薄膜的透水性", "今天天气很好"]
    vecs = embed(texts)
    print("句1 vs 句2(都讲膜):", float(np.dot(vecs[0], vecs[1])))  # 应该【高】
    print("句1 vs 句3(天气) :", float(np.dot(vecs[0], vecs[2])))  # 应该【低】
