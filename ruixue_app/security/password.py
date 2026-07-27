"""密码哈希。

两个要点(都是真实踩过的坑):

1. **绝不存明文,也绝不用 md5/sha1 直接存**。密码哈希要【慢】才安全 ——
   bcrypt 每次校验故意耗时几十毫秒,让暴力破解不可行;而 sha256 快到一秒能算几亿次。

2. **bcrypt 有 72 字节上限**:超过部分被【静默截断】,即"密码前72字节相同就算同一个密码"。
   对策(同 deer-flow):先用 sha256 把任意长度压成固定 32 字节、再 base64 → 永远不超限,
   且整个密码都参与运算。

"盐"由 bcrypt 自动生成并存进哈希串里,所以同一个密码每次哈希结果都不同 —— 防彩虹表。
"""

from __future__ import annotations

import base64
import hashlib

import bcrypt


def _prehash(password: str) -> bytes:
    """sha256 预哈希 → base64,固定 44 字节,绕开 bcrypt 的 72 字节截断。"""
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    """把明文密码变成可入库的哈希串(含盐)。"""
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """校验密码。用 bcrypt.checkpw —— 它是【常数时间比较】,防时序攻击。"""
    try:
        return bcrypt.checkpw(_prehash(password), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # 哈希串损坏/格式不对时判为失败,而不是抛异常暴露内部细节
        return False
