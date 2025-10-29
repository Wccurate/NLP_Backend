"""Configuration utilities for the Simple RAG + Agent backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_name: str = Field("gpt-4o-mini", env="MODEL_NAME")
    temperature: float = Field(0.3, env="TEMPERATURE")
    history_window: int = Field(10, env="HISTORY_WINDOW")
    chroma_dir: Path = Field(Path("./data/chroma"), env="CHROMA_DIR")
    primary_intent_mode: str = Field("openai", env="PRIMARY_INTENT_MODE")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Load and cache application settings."""

    load_dotenv()
    return Settings()

