"""Unit tests for TritonClient without a live server."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_get_client_singleton():
    from app.triton_client import get_client
    with patch("app.triton_client._client", None):
        with patch("app.triton_client.grpcclient.InferenceServerClient"):
            c1 = get_client()
            c2 = get_client()
            assert c1 is c2


def test_is_server_live_false_on_connection_error():
    from app.triton_client import TritonClient
    with patch("app.triton_client.grpcclient.InferenceServerClient") as MockClient:
        MockClient.return_value.is_server_live.side_effect = Exception("connection refused")
        client = TritonClient()
        assert client.is_server_live() is False


def test_is_model_ready_false_on_error():
    from app.triton_client import TritonClient
    with patch("app.triton_client.grpcclient.InferenceServerClient") as MockClient:
        MockClient.return_value.is_model_ready.side_effect = Exception("not found")
        client = TritonClient()
        assert client.is_model_ready() is False
