
"""Central application settings via Pydantic Settings."""

from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Preserve compatibility with model_config.py and other modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "HR-Policy-RAG"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    SECRET_KEY: str = "default-insecure-key-change-me"

    BACKEND_API_URL: str = "http://localhost:8000/api/v1"

    PROJECT_ROOT: Path = PROJECT_ROOT

    DATA_RAW_DIR: Path | None = None
    DATA_PROCESSED_DIR: Path | None = None
    ARTIFACTS_DIR: Path | None = None
    LOGS_DIR: Path | None = None
    VECTOR_INDEX_PATH: Path | None = None
    METADATA_STORE_PATH: Path | None = None

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @model_validator(mode="after")
    def resolve_project_paths(self):
        root = self.PROJECT_ROOT.expanduser().resolve()

        self.PROJECT_ROOT = root
        self.DATA_RAW_DIR = root / "data" / "raw"
        self.DATA_PROCESSED_DIR = root / "data" / "processed"
        self.ARTIFACTS_DIR = root / "artifacts"
        self.LOGS_DIR = root / "logs"
        self.VECTOR_INDEX_PATH = self.ARTIFACTS_DIR / "faiss_index.bin"
        self.METADATA_STORE_PATH = (
            self.ARTIFACTS_DIR / "chunks_metadata.json"
        )

        return self


settings = Settings()
