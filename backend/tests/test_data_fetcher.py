"""Tests for the football-data.org API client."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.data_fetcher import FootballDataFetcher


def _response(status_code: int, headers: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None
        response.json.return_value = {"matches": []}
    return response


class TestRateLimitedGet:
    def test_persistent_429_raises_after_bounded_retries(self):
        """A permanently throttled key must not retry forever (old code recursed unbounded)."""
        fetcher = FootballDataFetcher(api_key="test")
        fetcher.session.get = MagicMock(return_value=_response(429, {"Retry-After": "1"}))

        with patch("app.services.data_fetcher.time.sleep"):
            with pytest.raises(requests.HTTPError):
                fetcher._rate_limited_get("https://example.test/x")

        assert fetcher.session.get.call_count == fetcher.MAX_RATE_LIMIT_RETRIES + 1

    def test_recovers_after_transient_429(self):
        fetcher = FootballDataFetcher(api_key="test")
        fetcher.session.get = MagicMock(
            side_effect=[_response(429, {"Retry-After": "1"}), _response(200)]
        )

        with patch("app.services.data_fetcher.time.sleep"):
            data = fetcher._rate_limited_get("https://example.test/x")

        assert data == {"matches": []}
        assert fetcher.session.get.call_count == 2

    def test_non_429_error_raises_immediately(self):
        fetcher = FootballDataFetcher(api_key="test")
        fetcher.session.get = MagicMock(return_value=_response(500))

        with patch("app.services.data_fetcher.time.sleep"):
            with pytest.raises(requests.HTTPError):
                fetcher._rate_limited_get("https://example.test/x")

        assert fetcher.session.get.call_count == 1
