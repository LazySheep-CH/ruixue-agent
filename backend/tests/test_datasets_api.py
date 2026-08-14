"""数据集上传/查询/删除接口的测试。

渲染层和解析层已经在 test_analysis_loader.py 里测过了。这里测的是
只有走真接口才能验证的那些:

- 归属校验:猜到别人的 dataset_id 能不能读/删 —— 错了是数据泄露。
- 体积闸门:声明的 Content-Length 不可信,读完必须复核。
- 契约错误的状态码:用户数据不合格是 422(你的数据有问题),不是 500(我们坏了)。
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

_KA, _KB = "ds-test-alice", "ds-test-bob"
_GOOD = "PLA含量,PBAT含量,厚度,覆膜天数,降解率\n30,70,12,120,28.5\n50,50,10,120,41.2\n"


def _deps_up() -> bool:
    try:
        from sqlalchemy import text

        from ruixue_agent.persistence.engine import get_engine

        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _deps_up(), reason="需要 PostgreSQL")


@pytest.fixture
def api(monkeypatch):
    import ruixue_app.auth as auth
    import ruixue_app.main as main_mod

    monkeypatch.setattr(auth, "API_KEYS", {_KA: "alice-dsapi", _KB: "bob-dsapi"})
    return TestClient(main_mod.app)


def _upload(api, key: str, text: str = _GOOD, name: str = "田间记录.csv"):
    return api.post(
        "/datasets",
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/csv")},
        headers={"X-API-Key": key},
    )


# ── 正常路径 ──────────────────────────────────────────────────


def test_upload_returns_id_and_summary(api):
    r = _upload(api, _KA)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset_id"]
    assert body["n_rows"] == 2
    assert body["targets"] == ["DR"]
    assert "PLA_pct" in body["features"]


def test_uploaded_dataset_appears_in_my_list(api):
    ds_id = _upload(api, _KA).json()["dataset_id"]
    ids = [
        d["dataset_id"] for d in api.get("/datasets", headers={"X-API-Key": _KA}).json()["datasets"]
    ]
    assert ds_id in ids


# ── 归属校验:错了是数据泄露 ────────────────────────────────────


def test_another_user_cannot_see_my_dataset_in_their_list(api):
    ds_id = _upload(api, _KA).json()["dataset_id"]
    ids = [
        d["dataset_id"] for d in api.get("/datasets", headers={"X-API-Key": _KB}).json()["datasets"]
    ]
    assert ds_id not in ids, "别人的数据集出现在了我的列表里"


def test_another_user_cannot_delete_my_dataset(api):
    """返回 404 而不是 403 —— 403 等于确认"这个 id 存在",是可枚举的信息泄露。"""
    ds_id = _upload(api, _KA).json()["dataset_id"]
    assert api.delete(f"/datasets/{ds_id}", headers={"X-API-Key": _KB}).status_code == 404
    # 我自己还删得掉,证明上一步不是"根本没删成"
    assert api.delete(f"/datasets/{ds_id}", headers={"X-API-Key": _KA}).status_code == 200


def test_no_credential_is_rejected(api):
    assert api.get("/datasets").status_code == 401
    assert _upload(api, "wrong-key").status_code == 401


# ── 契约错误:是 422 不是 500 ──────────────────────────────────


def test_missing_target_column_is_422_with_actionable_message(api):
    """用户的数据不合契约 → 422(你的数据有问题);500 会让人以为是我们坏了。"""
    r = _upload(api, _KA, "PLA含量,厚度\n30,12\n")
    assert r.status_code == 422
    assert "实测值" in r.json()["detail"], "错误消息要说清缺什么、怎么改"


def test_empty_file_is_422(api):
    assert _upload(api, _KA, "").status_code == 422


def test_header_only_is_422(api):
    assert _upload(api, _KA, "PLA含量,降解率\n").status_code == 422


# ── 体积闸门 ──────────────────────────────────────────────────


def test_oversized_upload_is_413(api):
    """声明的 Content-Length 不可信,读完还要复核一次 ——
    只信任声明值等于让客户端自己决定能塞多少进来。"""
    from ruixue_agent.analysis.loader import MAX_BYTES

    r = _upload(api, _KA, "x" * (MAX_BYTES + 1024))
    assert r.status_code == 413


# ── 文件名:不可信输入不进路径也不进响应头 ──────────────────────


def test_filename_with_traversal_is_stored_as_plain_text(api):
    """我们根本不落盘,所以路径穿越无处可去 —— 但仍要确认它没被当路径用,
    而且不影响正常上传(不是靠"拒绝奇怪文件名"来防,是靠"根本不用它拼路径")。"""
    r = _upload(api, _KA, name="../../etc/passwd.csv")
    assert r.status_code == 200
    assert r.json()["n_rows"] == 2
