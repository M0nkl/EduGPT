from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    GROQ_API_URL: str
    GROQ_API_KEY: str

    DATABASE_URL: str = "sqlite:///./data/methodics.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()