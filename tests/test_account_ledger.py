from __future__ import annotations

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

    order = ledger.create_topup_order("pearl", 5_000, provider="mockpay")

    assert order.status == "pending"
    assert order.pay_url == f"roboclaw://pay/mockpay/{order.order_id}"
    assert ledger.wallet("pearl").balance_cents == 0

    paid_order, wallet, record = ledger.complete_topup_order(order.order_id, provider_order_id="txn-1")

    assert paid_order.status == "paid"
    assert paid_order.provider_order_id == "txn-1"
    assert wallet.balance_cents == 5_000
    assert record is not None
    assert record.kind == "payment_recharge"
    assert record.job_id == order.order_id

    paid_order_2, wallet_2, record_2 = ledger.complete_topup_order(order.order_id)
    assert paid_order_2.status == "paid"
    assert wallet_2.balance_cents == 5_000
    assert record_2 is None


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
    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    recharge = client.post(
        "/api/admin/account/recharge",
        json={"username": "pearl", "amount_cents": 10_000, "reason": "test topup"},
    )
    assert recharge.status_code == 200
    assert recharge.json()["wallet"]["availableCents"] == 10_000

    freeze = client.post(
        "/api/billing/freeze",
        json={"username": "pearl", "amount_cents": 4_000, "task_name": "train-1"},
    )
    assert freeze.status_code == 200
    assert freeze.json()["wallet"]["frozenCents"] == 4_000

    settle = client.post(
        "/api/billing/settle",
        json={"username": "pearl", "amount_cents": 2_500, "task_name": "train-1"},
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
        json={"username": "pearl", "amount_cents": 8_000, "provider": "mockpay"},
    )
    assert order_response.status_code == 200
    order = order_response.json()["order"]
    assert order["status"] == "pending"

    balance = client.get("/api/account/balance", params={"username": "pearl"})
    assert balance.json()["wallet"]["availableCents"] == 0

    complete_response = client.post(
        "/api/account/topup-orders/complete",
        json={"order_id": order["orderId"], "provider_order_id": "txn-2"},
    )
    assert complete_response.status_code == 200
    assert complete_response.json()["order"]["status"] == "paid"
    assert complete_response.json()["wallet"]["availableCents"] == 8_000
    assert complete_response.json()["record"]["kind"] == "payment_recharge"

    orders = client.get("/api/account/topup-orders", params={"username": "pearl"})
    assert orders.status_code == 200
    assert orders.json()["orders"][0]["providerOrderId"] == "txn-2"

    set_ledger_for_tests(None)


def test_account_routes_reject_insufficient_balance(tmp_path) -> None:
    set_ledger_for_tests(AccountLedger(tmp_path / "ledger.json"))
    app = FastAPI()
    register_account_routes(app)
    client = TestClient(app)

    response = client.post("/api/billing/freeze", json={"username": "pearl", "amount_cents": 1})

    assert response.status_code == 409
    assert "insufficient" in response.json()["detail"]
    set_ledger_for_tests(None)
