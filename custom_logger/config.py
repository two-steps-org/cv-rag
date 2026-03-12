"""Configuration for Elasticsearch handler."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ElasticLoggerConfig:
    hosts: List[Dict[str, Any]] = field(default_factory=lambda: [{"host": "localhost", "port": 9200}])
    index_name: str = "python-logs"
    index_pattern: str = "python-logs"
    use_ssl: bool = False
    verify_certs: bool = True
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    cloud_id: Optional[str] = None
    service_name: str = "python-app"
    project_name: Optional[str] = None
    environment: Optional[str] = None
    flush_interval: float = 1.0
    bulk_size: int = 100

    def to_connection_params(self) -> Dict[str, Any]:
        hosts = []
        for h in self.hosts:
            if isinstance(h, dict) and "scheme" not in h:
                scheme = "https" if self.use_ssl else "http"
                hosts.append({**h, "scheme": scheme})
            else:
                hosts.append(h)
        params = {"hosts": hosts or [{"scheme": "http", "host": "localhost", "port": 9200}], "verify_certs": self.verify_certs}
        if self.username and self.password:
            params["basic_auth"] = (self.username, self.password)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.cloud_id:
            params["cloud_id"] = self.cloud_id
        return params
