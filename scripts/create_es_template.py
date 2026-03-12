#!/usr/bin/env python3
"""
Create Elasticsearch index template for benchmark-* (or custom prefix).
Run once before logger starts. Example: python scripts/create_es_template.py --prefix benchmark
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Create ES index template for monthly logs")
    parser.add_argument("--prefix", default="benchmark", help="Index prefix (default: benchmark)")
    parser.add_argument("--priority", type=int, default=0, help="Template priority (use 10+ if conflict with broader pattern like benchmark-*)")
    parser.add_argument("--host", default="localhost", help="ES host")
    parser.add_argument("--port", type=int, default=9200, help="ES port")
    args = parser.parse_args()

    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        print("pip install elasticsearch", file=sys.stderr)
        sys.exit(1)

    client = Elasticsearch(hosts=[{"scheme": "http", "host": args.host, "port": args.port}])
    template_name = f"{args.prefix}-template"
    index_pattern = [f"{args.prefix}-*"]

    body = {
        "index_patterns": index_pattern,
        "priority": args.priority,
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "timestamp": {"type": "date"},
                    "severity": {"type": "keyword"},
                    "message": {"type": "text"},
                    "project": {"type": "keyword"},
                    "service": {"properties": {"name": {"type": "keyword"}}},
                    "environment": {"type": "keyword"},
                    "auth": {"type": "object", "enabled": True},
                    "request_context": {"type": "object", "enabled": True},
                    "custom_fields": {"type": "object", "enabled": True},
                }
            },
        },
    }

    try:
        client.indices.put_index_template(name=template_name, body=body)
        print(f"Template '{template_name}' created successfully.")
        print(f"Logger should use index_pattern='{args.prefix}-{{month}}' (e.g. {args.prefix}-03-26 for March 2026)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
