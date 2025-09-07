"""Global configurations."""
from pydantic_settings import BaseSettings, SettingsConfigDict

from constants import ModelProvider


class Settings(BaseSettings):
    """Global settings overridable by .env file and environment variables."""

    port: int = 8080

    model_provider: ModelProvider = "AzureOpenAI"

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_embed_model_deployment_name: str = ""
    azure_openai_llm_deployment_name: str = ""

    vertexai_llm: str = "gemini-1.0-pro-001"
    vertexai_embed_model: str = "textembedding-gecko@003"

    postgres_conn_str: str = "postgresql+psycopg2://pgadmin:pgadmin@localhost:5432/pgvectordb"  # following RFC-1738
    postgres_table: str = (
        "llamaindex"  # the actual table name will be data_{postgres_table}
    )

    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_endpoint: str = "http://localhost:9000"

    gcp_project: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),  # do not protect the "model_" namespace
    )

    hf_embedding : str = ""
    groq_model_name : str = ""
    groq_api_key: str = ""
    groq_endpoint: str = ""

settings = Settings()  # singleton
