import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parents[2]))

import os
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from src.utils.logger import logger

_PROJECT_ROOT = Path(__file__).parents[2]
_SETTING_FILE = _PROJECT_ROOT / "config/setting.yaml"


class LLMConfig(BaseModel):
    model_path: str
    temperature: float


class EmbeddingConfig(BaseModel):
    model_name: str
    base_url: str


class RagConfig(BaseModel):
    reranker: str


class RetrievalConfig(BaseModel):
    threshold: float
    top_k: int


class StorageConfig(BaseModel):
    data_dir: str
    vector_db: str
    collection_name: str


class LangsmithConfig(BaseModel):
    """LangSmith tracing project settings."""

    project: str
    endpoint: str


class AppConfig(BaseModel):

    project_name: str
    version: str
    debug: bool


class Settings(BaseSettings):
    langsmith_api_key: str | None = Field(None, alias="LANGSMITH_API_KEY")

    app: AppConfig
    llm: LLMConfig
    rag: RagConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    langsmith: LangsmithConfig

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )


def load_setting() -> Settings:
    if not _SETTING_FILE.exists():
        raise FileNotFoundError("setting.yaml not found")

    with open(_SETTING_FILE, "r", encoding="utf-8") as f:
        setting_config = yaml.safe_load(f)

    return Settings(**setting_config)


try:
    settings = load_setting()
    logger.info("Setting load successfully")

    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith.endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project
        logger.info(
            f"LangSmith tracing enabled for project: {settings.langsmith.project}"
        )

except Exception as e:
    logger.error(f"Error while loading settings: {e}")
    raise
