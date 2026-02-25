"""Application configuration using Pydantic Settings v2."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "RagLaw API"
    API_STR: str = "/api"

    # === Google Gemini ===
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # === Anthropic (fallback) ===
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"

    # === Qdrant ===
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "laws"

    # === Embedding ===
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding_v2"
    EMBEDDING_DIM: int = 1024
    DEVICE: str = "cpu"

    # === Retrieval ===
    RETRIEVAL_MODE: str = "dense"  # "hybrid", "dense", or "sparse"
    RETRIEVAL_K: int = 10
    TOP_K: int = 5

    # === Database ===
    DATABASE_URL: str = "sqlite:///./raglaw.db"

    @property
    def COLLECTION_NAME(self) -> str:
        return self.QDRANT_COLLECTION


config = Config()
