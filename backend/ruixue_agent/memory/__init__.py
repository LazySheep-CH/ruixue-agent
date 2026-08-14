"""长期记忆:跨会话记住用户的事实。

## 和短期记忆的分工

    短期记忆(已有)  一次会话内的滑动窗口 + 摘要压缩,会话结束就没了
    长期记忆(本模块) 跨会话,记住"这个用户是谁、他的地在哪、种什么"

## 三段式

    抽取(extract) → 存储(store) → 召回注入(recall + 中间件)

写入是【运行结束后异步做】的,不阻塞用户;抽取失败不影响回答 ——
记忆是锦上添花,不能因为它拖垮主流程。

设计取舍(存事实而非原始对话、为什么两边都存)写在
persistence/models.py 的 MemoryRow 文档里。
"""

from ruixue_agent.memory.store import (
    MEMORY_COLLECTION,
    delete_memory,
    list_memories,
    recall,
    remember,
)

__all__ = ["MEMORY_COLLECTION", "delete_memory", "list_memories", "recall", "remember"]
