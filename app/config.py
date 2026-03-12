from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "HR Resume Agent"
    port: int = 8000
    debug: bool = False
    database_name: str = "hr_agent"
    uploads_dir: str = "uploads"

    voyage_api_key: str
    anthropic_api_key: str
    atlas_connection_string: str

    # Elasticsearch (for custom_logger)
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index: str = "cv-rag-logs"
    environment: str = "development"


settings = Settings()
