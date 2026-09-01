from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
 app_name:str="Adaptive AI Evaluation Lab"; database_url:str="sqlite:///./adaptive_lab.db"; ollama_base_url:str="http://localhost:11434"; cors_origins:str="http://localhost:5173,http://localhost:3000"; auto_seed:bool=True
 model_config=SettingsConfigDict(env_file=".env",extra="ignore")
 @property
 def cors_list(self): return [x.strip() for x in self.cors_origins.split(',') if x.strip()]
settings=Settings()
