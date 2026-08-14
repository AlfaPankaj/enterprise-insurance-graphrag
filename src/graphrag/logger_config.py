"""Structured JSON logging (Phase 5).

One JSON object per log line — grep-able, parseable by any log aggregator
(ELK/Loki/Datadog). ``setup_logging`` installs the formatter on the app
loggers; every module that does ``logging.getLogger(\"graphrag...\")`` inherits
it without touching its code.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """stdlib logging formatter that emits one JSON object per record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                   + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # extra={"request_id": ...} style context lands as top-level fields
        for key in ("request_id", "doc_id", "query", "latency_ms", "savings_pct"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO, force: bool = False) -> None:
    """Attach the JSON formatter to the root ``graphrag`` logger once."""
    logger = logging.getLogger("graphrag")
    if logger.handlers and not force:
        return
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers[:] = [handler]
    logger.propagate = False  # don't double-log via the root handler
