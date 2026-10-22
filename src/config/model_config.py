"""Model parameters, retrieval thresholds, and flexible LLM configurations (API vs. Local Transformers)."""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.config.settings import PROJECT_ROOT


class LLMConfig(BaseModel):
    """Configuration contract for LLM generation layer."""
    api_based: bool = Field(
        default=True,
        description="True for cloud API providers (OpenAI, Groq), False for local HuggingFace transformers.",
    )
    provider: str = Field(default="openai", description="API provider name (e.g. openai, groq)")
    api_key: Optional[str] = Field(default="", description="API key (required if api_based=True)")
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible providers like Groq or local vLLM.",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Model identifier (OpenAI/Groq model tag or HuggingFace repo id)",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, gt=0)
    request_timeout: float = Field(default=45.0, gt=0.0)


class EmbeddingConfig(BaseModel):
    provider: str = Field(default="huggingface", description="Embedding model provider")
    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Pretrained embedding model",
    )
    dimension: int = Field(default=384, description="Vector dimension size")
    device: str = Field(default="cpu", description="Execution device (cpu or cuda)")
    batch_size: int = Field(default=32, gt=0)


class RerankerConfig(BaseModel):
    enabled: bool = Field(default=True)
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking",
    )
    top_k: int = Field(default=3, description="Number of passages to return after reranking")


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=500, description="Target chunk size in characters/tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between consecutive chunks")
    separators: list[str] = Field(
        default=["\n\n", "\n", " ", ""],
        description="Separators ordered by splitting priority",
    )


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=10, description="Initial candidates retrieved from vector index")
    score_threshold: float = Field(default=0.3, description="Minimum relevance similarity threshold")


class ModelSettings(BaseSettings):
    # Dynamic LLM Settings
    API_BASED: bool = True
    API_PROVIDER: str = "openai"
    API_BASE_URL: Optional[str] = None
    MODEL_API: str = ""
    MODEL_NAME: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"

    # Reranker Settings
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_llm_config(self) -> LLMConfig:
        return LLMConfig(
            api_based=self.API_BASED,
            provider=self.API_PROVIDER.lower(),
            api_key=self.MODEL_API,
            base_url=self.API_BASE_URL if self.API_BASE_URL else None,
            model_name=self.MODEL_NAME,
            temperature=self.LLM_TEMPERATURE,
            max_tokens=self.LLM_MAX_TOKENS,
        )

    def get_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=self.EMBEDDING_PROVIDER,
            model_name=self.EMBEDDING_MODEL,
            dimension=self.EMBEDDING_DIMENSION,
            device=self.EMBEDDING_DEVICE,
        )

    def get_reranker_config(self) -> RerankerConfig:
        return RerankerConfig(
            enabled=self.RERANKER_ENABLED,
            model_name=self.RERANKER_MODEL,
        )


model_settings = ModelSettings()
chunking_config = ChunkingConfig()
retrieval_config = RetrievalConfig()