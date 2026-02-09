# ingestion_utils.py
# Shared utilities for external API ingestion: safe HTTP, retry/backoff, JSON parsing, schema normalization.
# Used by notebooks 09–13 (World Bank WDI/WGI, IMF commodity prices, NY Fed GSCPI, WTO barometer).

from __future__ import annotations

import json
import time
from typing import Any, Optional

try:
    import requests
except ImportError:
    requests = None


def safe_get(
    url: str,
    *,
    timeout: int = 60,
    retries: int = 3,
    backoff: float = 2.0,
    headers: Optional[dict] = None,
) -> requests.Response:
    """
    Perform HTTP GET with retries and exponential backoff.
    Raises on final failure; caller should catch and fail gracefully.
    """
    if requests is None:
        raise RuntimeError("requests is required; install with: pip install requests")
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {})
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last_exc


def parse_json(text: str) -> Any:
    """Parse JSON string; returns Python object. Raises on invalid JSON."""
    return json.loads(text)


def normalize_indicator_row(
    *,
    source: str,
    ingested_at: str,
    as_of_date: str,
    country_code: Optional[str],
    indicator_code: str,
    indicator_name: str,
    value: Optional[float],
    unit: Optional[str],
    frequency: str,
    raw_payload: Optional[str] = None,
) -> dict:
    """
    Build a row dict conforming to the standard indicator schema:
    source, ingested_at, as_of_date, country_code, indicator_code, indicator_name, value, unit, frequency, raw_payload.
    """
    return {
        "source": source,
        "ingested_at": ingested_at,
        "as_of_date": as_of_date,
        "country_code": country_code if country_code is not None else "",
        "indicator_code": indicator_code,
        "indicator_name": indicator_name,
        "value": float(value) if value is not None else None,
        "unit": unit,
        "frequency": frequency,
        "raw_payload": raw_payload,
    }
