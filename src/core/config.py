"""Application config — loads config/setting.yaml + .env."""

import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[2]))

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import logger

_PROJECT_ROOT = Path(__file__).parents[2]
_SETTING_FILE = _PROJECT_ROOT / "config/setting.yaml"


# ── LLM config (nested: server + 2 role) ─────────────────────────────


class LLMServerConfig(BaseModel):


    binary: str
    host: str = "127.0.0.1"
    port: int = 8001
    base_url: str
    api_key: str = "not-needed"
    startup_timeout_s: int = 60
    shutdown_timeout_s: int = 5
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    extra_args: list[str] = Field(default_factory=list)


class LLMRoleConfig(BaseModel):


    model_name: str
    model_path: str
    temperature: float = 0.0
    max_tokens: int = 1024


class LLMConfig(BaseModel):
    """Bundle: server config + 2 role (coder, instruct)."""

    server: LLMServerConfig
    coder: LLMRoleConfig
    instruct: LLMRoleConfig





class EmbeddingConfig(BaseModel):
    model_name: str
    base_url: str | None = None


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


class APIConfig(BaseModel):
    """HTTP layer config."""
    request_budget_seconds: int = 58


class SolverConfig(BaseModel):
    timeout_s: int = 20
    max_retries: int = 1


class AppConfig(BaseModel):
    project_name: str
    version: str
    debug: bool


# ── Distillation config (offline pipeline, optional) ─────────────────


class DistillationTeacherConfig(BaseModel):
    """Optional offline teacher config. Closed-source LLM APIs are prohibited."""
    provider: str = "none"
    model_name: str = ""
    temperature: float = 0.1
    max_output_tokens: int = 1024
    api_key_env: str = ""


class DistillationPipelineConfig(BaseModel):

    mode: str = "extract"
    concurrency: int = 8
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    timeout_s: int = 60


class DistillationPathsConfig(BaseModel):
    raw_output: str = "data/distilled/physics_kb.raw.jsonl"
    verified_output: str = "data/distilled/physics_kb.verified.jsonl"
    cost_log: str = "data/distilled/cost_log.jsonl"


class DistillationConfig(BaseModel):
    teacher: DistillationTeacherConfig = Field(default_factory=DistillationTeacherConfig)
    pipeline: DistillationPipelineConfig = Field(default_factory=DistillationPipelineConfig)
    paths: DistillationPathsConfig = Field(default_factory=DistillationPathsConfig)


class Settings(BaseSettings):
    langsmith_api_key: str | None = Field(None, alias="LANGSMITH_API_KEY")

    app: AppConfig
    llm: LLMConfig
    api: APIConfig
    solver: SolverConfig = Field(default_factory=SolverConfig)
    rag: RagConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    langsmith: LangsmithConfig
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )


def load_setting() -> Settings:
    if not _SETTING_FILE.exists():
        raise FileNotFoundError(f"setting.yaml not found at {_SETTING_FILE}")

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
