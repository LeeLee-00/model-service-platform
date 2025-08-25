from pydantic import  Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HF_TOKEN: str = Field("", env="HF_TOKEN")
    MINIO_SVC_USER: str = Field("admin", env="MINIO_SVC_USER")
    MINIO_BUCKET: str = Field("models", env="MINIO_BUCKET")
    MINIO_ENDPOINT: str = Field("localhost:9000", env="MINIO_ENDPOINT")
    MINIO_SVC_PASSWORD: str = Field("password", env="MINIO_SVC_PASSWORD")

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
