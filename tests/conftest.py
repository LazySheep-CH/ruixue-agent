"""pytest 公共夹具(fixture)。

核心问题:测数据库的测试,跑完得【干净】—— 不能往库里留垃圾,
更不能让第 1 个测试插的数据影响第 2 个测试(那样测试就有了顺序依赖,
时好时坏,最后没人信它)。

做法:每个测试跑在一个【事务】里,测完直接 rollback。
      数据库像什么都没发生过一样。这比"测完手动 DELETE"可靠得多 ——
      测试中途抛异常时,DELETE 那行根本执行不到,而 rollback 由 fixture 兜底。
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine


def _pg_alive() -> bool:
    try:
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def pg_available() -> bool:
    return _pg_alive()


@pytest.fixture
def session(pg_available):
    """一个测完自动回滚的 Session。

    没起 PG 就跳过 —— 而不是报错。别人 clone 下来只想跑单测,
    不该被"你没装数据库"卡住。但【跳过】会明晃晃打印出来,不是静默通过。
    """
    if not pg_available:
        pytest.skip("PostgreSQL 没起(docker compose up -d postgres)")

    conn = get_engine().connect()
    trans = conn.begin()
    sess = Session(bind=conn)
    try:
        yield sess
    finally:
        sess.close()
        trans.rollback()  # ← 关键:无论测试成功失败,数据都不落库
        conn.close()
