"""Query logs from Elasticsearch with filters (index name, date, month)."""
from typing import List, Dict, Any, Optional
from datetime import datetime


def get_es_client(hosts: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
    """Get Elasticsearch client."""
    try:
        from elasticsearch import Elasticsearch
    except ImportError as e:
        raise ImportError("pip install elasticsearch") from e

    conn = hosts or [{"host": "localhost", "port": 9200}]
    return Elasticsearch(hosts=conn, **kwargs)


def query_logs(
    index_pattern: str = "python-logs*",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    severity: Optional[str] = None,
    size: int = 100,
    from_: int = 0,
    hosts: Optional[List[Dict[str, Any]]] = None,
    **es_kwargs,
) -> Dict[str, Any]:
    """
    Query logs from Elasticsearch with filters.

    Args:
        index_pattern: Index name or pattern, e.g. "python-logs", "python-logs*", "python-logs-2025.03.11"
        from_date: Start date (YYYY-MM-DD or ISO datetime)
        to_date: End date (YYYY-MM-DD or ISO datetime)
        year: Filter by year (e.g. 2025)
        month: Filter by month 1-12 (use with year)
        severity: Filter by severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        size: Max hits to return (default 100)
        from_: Offset for pagination (default 0)
        hosts: ES connection hosts
        **es_kwargs: Additional ES client options

    Returns:
        Elasticsearch search response with hits, total, etc.
    """
    client = get_es_client(hosts=hosts, **es_kwargs)

    must = []

    # Date range filter
    if year is not None and month is not None:
        # Filter by specific month
        start = f"{year}-{month:02d}-01T00:00:00.000Z"
        if month == 12:
            end = f"{year + 1}-01-01T00:00:00.000Z"
        else:
            end = f"{year}-{month + 1:02d}-01T00:00:00.000Z"
        must.append({"range": {"@timestamp": {"gte": start, "lt": end}}})
    elif from_date or to_date:
        range_query: Dict[str, Any] = {"@timestamp": {}}
        if from_date:
            range_query["@timestamp"]["gte"] = from_date
        if to_date:
            range_query["@timestamp"]["lte"] = to_date
        must.append({"range": range_query})

    # Severity filter
    if severity:
        must.append({"term": {"severity.keyword": severity}})

    body: Dict[str, Any] = {
        "size": size,
        "from": from_,
        "track_total_hits": True,
        "sort": [{"@timestamp": {"order": "desc"}}],
    }
    if must:
        body["query"] = {"bool": {"must": must}}
    else:
        body["query"] = {"match_all": {}}

    try:
        return client.search(
            index=index_pattern,
            body=body,
            ignore_unavailable=True,
            allow_no_indices=True,
        )
    except Exception:
        return {"hits": {"total": 0, "hits": []}}


def list_indices(hosts: Optional[List[Dict[str, Any]]] = None, pattern: str = "python-logs*") -> List[str]:
    """List indices matching pattern (useful for index name filter dropdown)."""
    try:
        client = get_es_client(hosts=hosts)
        resp = client.cat.indices(index=pattern, format="json")
        return sorted([idx["index"] for idx in resp if not idx["index"].startswith(".")])
    except Exception:
        return []
