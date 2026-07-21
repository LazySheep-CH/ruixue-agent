"""文本向量化。模型:BAAI/bge-small-zh-v1.5(中文,512 维)。"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_model = None


def get_model() -> SentenceTransformer:
    """懒加载单例:模型约 100MB,首次调用加载,进程内复用。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# 批大小。sentence-transformers 默认 32,GPU 利用率不足。
# 实测(RTX 3090,平均 171 字的真实语料):32 → 874 条/秒,
# 256 → 1247 条/秒,512 → 1186 条/秒(过大反而下降)。
# 换硬件或换模型后需重新标定。
_BATCH_SIZE = 256


def embed(texts: list[str]):
    """批量编码。输出已 L2 归一化,点积即余弦相似度。

    下游依赖此契约:Milvus 用 COSINE 度量,评测与融合按点积比较。
    """
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
    print("相关句对:", float(np.dot(vecs[0], vecs[1])))
    print("无关句对:", float(np.dot(vecs[0], vecs[2])))
