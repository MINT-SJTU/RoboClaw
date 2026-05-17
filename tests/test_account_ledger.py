from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.account import AccountLedger
from roboclaw.http.routes.account import register_account_routes, set_ledger_for_tests


def test_account_ledger_recharge_freeze_settle_release(tmp_path) -> None:
    ledger = AccountLedger(tmp_path / "ledger.json")

    wallet, recharge = ledger.admin_recharge("pearl", 10_000)
    assert wallet.balance_cents == 10_000
    assert wallet.available_cents == 10_000
    assert recharge.kind == "admin_recharge"

    wallet, frozen = ledger.freeze("pearl", 4_000, task_name="train-1", job_id="job-1")
    assert wallet.balance_cents == 10_000
    assert wallet.frozen_cents == 4_000
    assert wallet.available_cents == 6_000
    assert frozen.kind == "freeze"

    wallet, released = ledger.release("pearl", 1_000)
    assert wallet.balance_cents == 10_000
    assert wallet.frozen_cents == 3_000
    assert wallet.available_cents == 7_000
    assert released.kind == "release"

    wallet, settled = ledger.settle("pearl", 3_000)
    assert wallet.balance_cents == 7_000
    assert wallet.frozen_cents == 0
    assert wallet.available_cents == 7_000
    assert settled.kind == "settle"
    assert settled.amount_cents == -3_000

    records = ledger.records("pearl")
    assert [record.kind for record in records] == ["settle", "release", "freeze", "admin_recharge"]


def test_account_ledger_topup_order_auto_recharges_once(tmp_path) -> None:
    ledger = AccountLedger(tmp_path / "ledger.json")

    order = ledger.create_topup_order(
        "pearl",
        5_000,
        bonus_points=5,
        provider="mockpay",
        payee_name="Evo Studio",
        payee_account="merchant-001",
    )

    assert order.status == "pending"
    assert order.bonus_points == 5
    assert order.payee_name == "Evo Studio"
    assert order.payee_account == "merchant-001"
    assert order.pay_url == f"roboclaw://pay/mockpay/{order.order_id}"
    assert ledger.wallet("pearl").balance_cents == 0

    paid_order, wallet, record = ledger.complete_topup_order(order.order_id, provider_order_id="txn-1")

    assert paid_order.status == "paid"
    assert paid_order.payee_account == "merchant-001"
    assert paid_order.provider_order_id == "txn-1"
    assert wallet.balance_cents == 5_000
    assert wallet.reward_points == 5
    assert record is not None
    assert record.kind == "payment_recharge"
    assert record.job_id == order.order_id

    paid_order_2, wallet_2, record_2 = ledger.complete_topup_order(order.order_id)
    assert paid_order_2.status == "paid"
    assert wallet_2.balance_cents == 5_000
    assert wallet_2.reward_points == 5
    assert record_2 is None


def test_account_ledger_dataset_reward_is_idempotent(tmp_path) -> None:
    ledger = AccountLedger(tmp_path / "ledger.json")

    wallet, record, granted = ledger.grant_dataset_reward("pearl", "dataset-1", 15)

    assert granted is True
    assert wallet.available_cents == 0
    assert wallet.reward_points == 15
    assert record.kind == "dataset_reward"
    assert record.amount_cents == 15
    assert record.reward_points_after == 15
    assert record.job_id == "dataset-1"

    wallet_2, record_2, granted_2 = ledger.grant_dataset_reward("pearl", "dataset-1", 15)
    assert granted_2 is False
    assert wallet_2.available_cents == 0
    assert wallet_2.reward_points == 15
    assert record_2.record_id == record.record_id


def test_account_ledger_reassigns_pending_training_hold(tmp_path) -> None:
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("pearl", 10_000)
    _wallet, hold = ledger.freeze("pearl", 990, job_id="pending-cloud-train")

    updated = ledger.reassign_job_hold("pearl", "pending-cloud-train", "cloud-job-1")

    assert updated.record_id == hold.record_id
    assert updated.job_id == "cloud-job-1"
    records = ledger.records("pearl")
    assert records[0].kind == "freeze"
    assert records[0].job_id == "cloud-job-1"
    assert ledger.wallet("pearl").frozen_cents == 990


def test_account_ledger_releases_only_matching_job_hold(tmp_path) -> None:
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("pearl", 10_000)
    ledger.freeze("pearl", 990, job_id="job-a")
    ledger.freeze("pearl", 990, job_id="job-b")

    wallet, release = ledger.release_job_hold("pearl", "job-a")

    assert release.amount_cents == 990
    assert release.job_id == "job-a"
    assert wallet.frozen_cents == 990
    records = ledger.records("pearl")
    assert [record.job_id for record in records if record.kind == "freeze"] == ["job-b", "job-a"]


def test_account_ledger_rejects_insufficient_balance(tmp_path) -> None:
    ledger = AccountLedger(tmp_path / "ledger.json")
    ledger.admin_recharge("pearl", 100)

    try:
        ledger.freeze("pearl", 200)
    except ValueError as exc:
        assert "insufficient" in str(exc)
    else:
        raise AssertionError("freeze should fail")


def test_account_routes_flow(tmp_path) -> None:
    headers = {"X-Roboclaw-Admin-Token": "admin-test"}
    from roboclaw.http.routes import account as account_routes

    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    with patch.dict(account_routes.os.environ, {"EVO_STUDIO_ADMIN_TOKEN": "admin-test"}):
        recharge = client.post(
            "/api/admin/account/recharge",
            json={"username": "pearl", "amount_cents": 10_000, "reason": "test topup"},
            headers=headers,
        )
        assert recharge.status_code == 200
        assert recharge.json()["wallet"]["availableBalanceCents"] == 10_000
        assert recharge.json()["wallet"]["availableCents"] == 10_000

        freeze = client.post(
            "/api/billing/freeze",
            json={"username": "pearl", "amount_cents": 4_000, "task_name": "train-1"},
            headers=headers,
        )
        assert freeze.status_code == 200
        assert freeze.json()["wallet"]["frozenCents"] == 4_000

        settle = client.post(
            "/api/billing/settle",
            json={"username": "pearl", "amount_cents": 2_500, "task_name": "train-1"},
            headers=headers,
        )
        assert settle.status_code == 200
        assert settle.json()["wallet"]["balanceCents"] == 7_500
        assert settle.json()["wallet"]["frozenCents"] == 1_500

    balance = client.get("/api/account/balance", params={"username": "pearl"})
    assert balance.status_code == 200
    assert balance.json()["wallet"]["availableCents"] == 6_000

    records = client.get("/api/account/billing-records", params={"username": "pearl"})
    assert records.status_code == 200
    assert [record["kind"] for record in records.json()["records"]] == ["settle", "freeze", "admin_recharge"]

    set_ledger_for_tests(None)


def test_account_routes_topup_order_flow(tmp_path) -> None:
    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    order_response = client.post(
        "/api/account/topup-orders",
        json={"username": "pearl", "amount_cents": 8_000, "bonus_points": 8, "provider": "mockpay"},
    )
    assert order_response.status_code == 200
    order = order_response.json()["order"]
    assert order["status"] == "pending"
    assert "paymentConfig" in order_response.json()

    balance = client.get("/api/account/balance", params={"username": "pearl"})
    assert balance.json()["wallet"]["availableCents"] == 0

    complete_response = client.post(
        "/api/account/topup-orders/complete",
        json={"order_id": order["orderId"], "provider_order_id": "txn-2"},
    )
    assert complete_response.status_code == 200
    assert complete_response.json()["order"]["status"] == "paid"
    assert complete_response.json()["wallet"]["availableBalanceCents"] == 8_000
    assert complete_response.json()["wallet"]["creditPoints"] == 8
    assert complete_response.json()["record"]["kind"] == "payment_recharge"

    orders = client.get("/api/account/topup-orders", params={"username": "pearl"})
    assert orders.status_code == 200
    assert orders.json()["orders"][0]["providerOrderId"] == "txn-2"

    set_ledger_for_tests(None)


def test_account_routes_payment_config_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVO_STUDIO_PAYMENT_PROVIDER", "alipay")
    monkeypatch.setenv("EVO_STUDIO_PAYEE_NAME", "Evo Studio")
    monkeypatch.setenv("EVO_STUDIO_PAYEE_ACCOUNT", "merchant-001")
    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    config = client.get("/api/account/payment-config")
    assert config.status_code == 200
    assert config.json()["configured"] is True
    assert config.json()["payeeAccount"] == "merchant-001"

    order = client.post(
        "/api/account/topup-orders",
        json={"username": "pearl", "amount_cents": 1_000, "provider": "alipay"},
    )
    assert order.status_code == 200
    assert order.json()["order"]["payeeName"] == "Evo Studio"
    assert order.json()["order"]["payeeAccount"] == "merchant-001"

    set_ledger_for_tests(None)


def test_account_routes_dataset_reward_flow(tmp_path) -> None:
    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    reward = client.post(
        "/api/account/rewards/dataset-upload",
        json={"username": "pearl", "dataset_id": "cloud/verify-so101", "reward_points": 20},
    )

    assert reward.status_code == 200
    assert reward.json()["granted"] is True
    assert reward.json()["wallet"]["availableBalanceCents"] == 0
    assert reward.json()["wallet"]["creditPoints"] == 20
    assert reward.json()["record"]["kind"] == "dataset_reward"

    duplicate = client.post(
        "/api/account/rewards/dataset-upload",
        json={"username": "pearl", "dataset_id": "cloud/verify-so101", "reward_points": 20},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["granted"] is False
    assert duplicate.json()["wallet"]["availableBalanceCents"] == 0
    assert duplicate.json()["wallet"]["creditPoints"] == 20

    set_ledger_for_tests(None)


def test_account_routes_reject_insufficient_balance(tmp_path) -> None:
    from roboclaw.http.routes import account as account_routes

    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    with patch.dict(account_routes.os.environ, {"EVO_STUDIO_ADMIN_TOKEN": "admin-test"}):
        response = client.post(
            "/api/billing/freeze",
            json={"username": "pearl", "amount_cents": 1},
            headers={"X-Roboclaw-Admin-Token": "admin-test"},
        )

    assert response.status_code == 409
    assert "insufficient" in response.json()["detail"]
    set_ledger_for_tests(None)


def test_account_admin_routes_require_token(tmp_path) -> None:
    from roboclaw.http.routes import account as account_routes

    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    with patch.dict(account_routes.os.environ, {"EVO_STUDIO_ADMIN_TOKEN": "admin-test"}):
        missing = client.post(
            "/api/admin/account/recharge",
            json={"username": "pearl", "amount_cents": 10_000},
        )
        wrong = client.post(
            "/api/admin/account/recharge",
            json={"username": "pearl", "amount_cents": 10_000},
            headers={"X-Roboclaw-Admin-Token": "wrong"},
        )

    assert missing.status_code == 403
    assert wrong.status_code == 403
    set_ledger_for_tests(None)
