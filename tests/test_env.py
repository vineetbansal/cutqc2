import os


def test_env():
    assert "GUROBI_WLSACCESSID" in os.environ and os.environ[
        "GUROBI_WLSACCESSID"
    ].startswith("158")
