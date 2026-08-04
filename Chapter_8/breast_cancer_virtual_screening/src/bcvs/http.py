from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class HTTPConfig:
    timeout_seconds: float = 60
    max_retries: int = 6
    requests_per_second: float | None = None


class ResilientSession:
    def __init__(self, config: HTTPConfig, user_agent: str = "bcvs/0.1"):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self._last_request = 0.0

    def _throttle(self) -> None:
        rps = self.config.requests_per_second
        if not rps:
            return
        minimum_interval = 1.0 / rps
        elapsed = time.monotonic() - self._last_request
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=self.config.timeout_seconds)
                self._last_request = time.monotonic()
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Transient HTTP {response.status_code}: {response.text[:200]}", response=response
                    )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                delay = min(60.0, (2**attempt) + random.random())
                LOGGER.warning("Request failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError(f"Request failed after retries: {url}") from last_error
