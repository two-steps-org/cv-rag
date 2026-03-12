"""Centralized logger instance using custom_logger.CustomLogger.

All modules should import `get_logger` from here instead of using
`logging.getLogger`. This ensures a single Elasticsearch handler
is shared across the application.
"""

import logging

from app.config import settings
from custom_logger import CustomLogger


def _create_root_logger() -> CustomLogger:
    """Create and configure the application-wide CustomLogger instance."""
    return CustomLogger(
        name="cv-rag",
        level=logging.DEBUG if settings.debug else logging.INFO,
        elastic_hosts=[
            {
                "host": settings.elasticsearch_host,
                "port": settings.elasticsearch_port,
            }
        ],
        index_name=settings.elasticsearch_index,
        service_name="cv-rag",
        project_name="cv-rag",
        environment=settings.environment,
        console=True,
    )


_root_logger = _create_root_logger()


def get_logger(name: str | None = None) -> CustomLogger:
    """Return the application logger.

    Args:
        name: Optional module name for log identification. The name is
              stored but the same root logger instance is returned so
              that all modules share the single Elasticsearch handler.

    Returns:
        The configured CustomLogger instance.
    """
    if name:
        _root_logger.name = name
    return _root_logger
