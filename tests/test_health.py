from app.health import liveness


def test_liveness_ok():
    result = liveness()
    assert result["status"] == "ok"
