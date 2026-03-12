"""Sends logs to Elasticsearch. View in Kibana."""
import logging
import sys
import threading
import atexit
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .config import ElasticLoggerConfig
from .formatter import ECSFormatter


class ElasticsearchHandler(logging.Handler):
    def __init__(self, config: Optional[ElasticLoggerConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = ElasticLoggerConfig(**kwargs) if kwargs else ElasticLoggerConfig()
        elif isinstance(config, dict):
            self.config = ElasticLoggerConfig(**config)
        else:
            self.config = config
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._client = None
        self._timer: Optional[threading.Timer] = None
        self._start_flush_timer()
        atexit.register(self._flush_on_exit)

    def _get_client(self):
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
                self._client = Elasticsearch(**self.config.to_connection_params())
            except ImportError as e:
                raise ImportError("pip install elasticsearch") from e
        return self._client

    def _get_index_name(self) -> str:
        p = self.config.index_pattern
        now = datetime.now(timezone.utc)
        if "{month}" in p:
            # benchmark-03-26 format (MM-YY)
            return p.replace("{month}", now.strftime("%m-%y"))
        if "{date}" in p:
            return p.replace("{date}", now.strftime("%Y.%m.%d"))
        return p

    def _build_document(self, record: logging.LogRecord) -> Dict[str, Any]:
        record.service = self.config.service_name
        record.environment = self.config.environment or "development"
        record.project_name = self.config.project_name or self.config.service_name
        if isinstance(self.formatter, ECSFormatter):
            doc = self.formatter._record_to_client_format(record)
        else:
            doc = {
                "severity": record.levelname,
                "message": self.format(record),
                "@timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "service": {"name": self.config.service_name},
                "environment": self.config.environment or "development",
            }
        return doc

    def emit(self, record: logging.LogRecord) -> None:
        try:
            doc = self._build_document(record)
            with self._lock:
                self._buffer.append(doc)
                if len(self._buffer) >= self.config.bulk_size:
                    self._flush_buffer()
        except Exception as e:
            self.handleError(record)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        bulk_body = []
        idx = self._get_index_name()
        for doc in self._buffer:
            bulk_body.append({"index": {"_index": idx}})
            bulk_body.append(doc)
        self._buffer = []
        try:
            resp = self._get_client().bulk(body=bulk_body, refresh=True)
            if resp.get("errors"):
                for item in resp.get("items", []):
                    idx = item.get("index", {})
                    if "error" in idx:
                        print(f"[elastic_logger] Index error: {idx['error']}", file=sys.stderr)
        except Exception as e:
            print(f"[elastic_logger] Error: {e}", file=sys.stderr)

    def _start_flush_timer(self) -> None:
        def flush():
            with self._lock:
                self._flush_buffer()
            self._timer = threading.Timer(self.config.flush_interval, flush)
            self._timer.daemon = True
            self._timer.start()
        self._timer = threading.Timer(self.config.flush_interval, flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush_on_exit(self) -> None:
        if self._timer:
            self._timer.cancel()
        with self._lock:
            self._flush_buffer()
