from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


API_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ADR Reformulation API"
    app_version: str = "1.0.0"
    max_input_tokens: int = 256
    model_dir: Path = API_ROOT / "modele_final"


@lru_cache
def get_settings() -> Settings:
    return Settings()
