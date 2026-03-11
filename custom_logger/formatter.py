"""
Client Core Fields Specification formatter.
Output: severity, message, timestamp, auth, error, stack_error, request_context, service, environment, custom_fields
"""
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Set

# Client spec: severity values
SEVERITY_MAP = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

# Use user_name (not "name") - LogRecord reserves "name" for logger
AUTH_KEYS: Set[str] = {"email", "user_name", "user_id", "session_id", "ip_address"}
REQUEST_CONTEXT_KEYS: Set[str] = {"request_id", "method", "endpoint", "duration_ms", "status_code"}
ERROR_KEYS: Set[str] = {"error_code", "error_type", "error_message", "http_status"}
SKIP_KEYS: Set[str] = {
    "name", "msg", "args", "created", "filename", "funcName", "levelname", "levelno",
    "lineno", "module", "msecs", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "exc_info", "exc_text", "thread", "threadName", "message", "taskName",
} | AUTH_KEYS | REQUEST_CONTEXT_KEYS | ERROR_KEYS


class ECSFormatter(logging.Formatter):
    """Formats logs per client Core Fields Specification."""

    MAX_MESSAGE_LENGTH = 2048

    def format(self, record: logging.LogRecord) -> str:
        import json
        return json.dumps(self._record_to_client_format(record), default=str)

    def _record_to_client_format(self, record: logging.LogRecord) -> Dict[str, Any]:
        severity = SEVERITY_MAP.get(record.levelno, "INFO")
        message = record.getMessage()
        if len(message) > self.MAX_MESSAGE_LENGTH:
            message = message[: self.MAX_MESSAGE_LENGTH - 3] + "..."

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        svc = getattr(record, "service", "api")
        env = getattr(record, "environment", "development")
        project = getattr(record, "project_name", None) or (svc if isinstance(svc, str) else svc.get("name", "api"))
        doc = {
            "@timestamp": ts,  # Kibana default time field
            "severity": severity,
            "message": message,
            "timestamp": ts,
            "project": project,  # Project name - clearly visible in Kibana
            "service": {"name": svc} if isinstance(svc, str) else svc,  # ES mapping expects object
            "environment": env,
        }

        auth = self._extract_auth(record)
        if auth:
            doc["auth"] = auth

        if severity in ("ERROR", "CRITICAL"):
            error_data = self._extract_error(record)
            if error_data:
                # ES index has "error" as text; store object as JSON string
                doc["error"] = json.dumps(error_data, default=str)
            stack = self._extract_stack(record)
            if stack:
                doc["stack_error"] = stack

        request_ctx = self._extract_request_context(record)
        if request_ctx:
            doc["request_context"] = request_ctx

        custom = self._extract_custom_fields(record)
        if custom:
            doc["custom_fields"] = custom

        return doc

    def _extract_auth(self, record: logging.LogRecord) -> Dict[str, Any] | None:
        auth = getattr(record, "auth", None)
        if isinstance(auth, dict):
            return auth
        out = {}
        mapping = {"user_name": "name"}  # client spec uses "name", extra uses "user_name"
        for k in AUTH_KEYS:
            if hasattr(record, k):
                v = getattr(record, k)
                if v is not None:
                    out[mapping.get(k, k)] = v
        return out if out else None

    def _extract_request_context(self, record: logging.LogRecord) -> Dict[str, Any] | None:
        ctx = getattr(record, "request_context", None)
        if isinstance(ctx, dict):
            return ctx
        out = {}
        for k in REQUEST_CONTEXT_KEYS:
            if hasattr(record, k):
                v = getattr(record, k)
                if v is not None:
                    out[k] = v
        return out if out else None

    def _extract_error(self, record: logging.LogRecord) -> Dict[str, Any] | None:
        out = {}
        if record.exc_info and record.exc_info[0]:
            out["error_type"] = record.exc_info[0].__name__
        if record.exc_info and record.exc_info[1]:
            out["error_message"] = str(record.exc_info[1])
        for k in ERROR_KEYS:
            if hasattr(record, k):
                v = getattr(record, k)
                if v is not None:
                    out[k] = v
        return out if out else None

    def _extract_stack(self, record: logging.LogRecord) -> str | None:
        if record.exc_info:
            return "".join(traceback.format_exception(*record.exc_info))
        return getattr(record, "stack_error", None)

    def _extract_custom_fields(self, record: logging.LogRecord) -> Dict[str, Any] | None:
        out = {}
        for k, v in record.__dict__.items():
            if k not in SKIP_KEYS and v is not None:
                if k in ("auth", "request_context"):
                    continue
                out[k] = v
        return out if out else None
