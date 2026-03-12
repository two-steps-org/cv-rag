"""
Elastic Logger - CustomLogger inherits logging.Logger.
Logs show in Elasticsearch and Kibana.
"""
import logging
import sys
from typing import Dict, List, Any, Optional

from .handler import ElasticsearchHandler
from .formatter import ECSFormatter
from .query import query_logs, list_indices

__version__ = "1.0.0"
__all__ = ["CustomLogger", "query_logs", "list_indices"]

# LogRecord reserved - cannot use in extra
_RESERVED = {
    "name", "msg", "args", "message", "pathname", "filename", "module", "lineno",
    "funcName", "created", "msecs", "levelname", "levelno", "process", "processName",
    "thread", "threadName", "exc_info", "exc_text", "stack_info", "relativeCreated",
    "taskName", "asctime",
}


class _ConsoleFormatter(logging.Formatter):
    """Console formatter - no traceback (traceback goes to ES only)."""

    def formatException(self, exc_info):
        return ""


def _filter_extra(extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Remove reserved keys, map 'name' -> 'user_name' for auth."""
    if not extra:
        return {}
    out = {}
    for k, v in extra.items():
        if k in _RESERVED:
            if k == "name" and v is not None:
                out["user_name"] = v
            continue
        out[k] = v
    return out


class CustomLogger(logging.Logger):
    """
    Custom logger (inherits logging.Logger).
    Logs go to console + Elasticsearch. View in Kibana.
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        elastic_hosts: Optional[List[Dict[str, Any]]] = None,
        index_name: str = "python-logs",
        index_pattern: Optional[str] = None,
        service_name: str = "api",
        project_name: Optional[str] = None,
        environment: str = "development",  # Required: development, staging, production
        console: bool = True,
        **kwargs,
    ):
        super().__init__(name, level)

        if console:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(_ConsoleFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
            self.addHandler(h)

        pattern = index_pattern or index_name
        proj = project_name or service_name
        es_handler = ElasticsearchHandler(
            hosts=elastic_hosts or [{"host": "localhost", "port": 9200}],
            index_name=index_name,
            index_pattern=pattern,
            service_name=service_name,
            project_name=proj,
            environment=environment,
            **kwargs,
        )
        es_handler.setFormatter(ECSFormatter())
        self.addHandler(es_handler)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        extra = _filter_extra(extra)
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)
