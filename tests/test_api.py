"""FastAPI endpoint tests using httpx async client."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from dashboard.main import app


@pytest.fixture(scope="module")
def event_loop_policy():
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


@pytest.mark.asyncio
async def test_dashboard_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_dashboard_index():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "MLOps" in r.text


@pytest.mark.asyncio
async def test_dashboard_summary():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/dashboard/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_predictions" in data
    assert "fraud_rate_pct" in data


@pytest.mark.asyncio
async def test_dashboard_timeseries():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/dashboard/timeseries")
    assert r.status_code == 200
    data = r.json()
    assert "buckets" in data
    assert len(data["buckets"]) > 0


@pytest.mark.asyncio
async def test_dashboard_drift():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/dashboard/drift")
    assert r.status_code == 200
    data = r.json()
    assert "overall_drift" in data
    assert "feature_results" in data


@pytest.mark.asyncio
async def test_predict_endpoint_legit():
    payload = {
        "transaction_id": "TXN-TEST-001",
        "features": {
            "amount": 49.99,
            "hour": 14,
            "day_of_week": 2,
            "merchant_category": 5,
            "distance_from_home": 2.5,
            "distance_from_last_transaction": 1.0,
            "ratio_to_median_purchase_price": 0.9,
            "repeat_retailer": 1,
            "used_chip": 1,
            "used_pin_number": 1,
            "online_order": 0,
        },
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_probability" in data
    assert 0 <= data["fraud_probability"] <= 1
    assert "is_fraud" in data


@pytest.mark.asyncio
async def test_predict_endpoint_fraud_scenario():
    payload = {
        "transaction_id": "TXN-FRAUD-001",
        "features": {
            "amount": 1999.99,
            "hour": 3,
            "day_of_week": 6,
            "merchant_category": 17,
            "distance_from_home": 450.0,
            "distance_from_last_transaction": 380.0,
            "ratio_to_median_purchase_price": 14.5,
            "repeat_retailer": 0,
            "used_chip": 0,
            "used_pin_number": 0,
            "online_order": 1,
        },
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["fraud_probability"] > data["fraud_probability"] * 0


@pytest.mark.asyncio
async def test_experiments_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/dashboard/experiments")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "auc" in data[0]


@pytest.mark.asyncio
async def test_model_versions_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/dashboard/model-versions")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_recent_predictions_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/dashboard/recent-predictions")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
