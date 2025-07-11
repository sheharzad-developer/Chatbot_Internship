import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # MongoDB Configuration
    mongodb_url: str = os.getenv("MONGODB_URL", "")
    mongodb_db_name: str = "chatbot_db"
    
    # Tavily API Configuration
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    
    # Langfuse Configuration
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Google AI Configuration
    google_ai_generative: str = os.getenv("GOOGLE_AI_GENERATIVE", "")
    
    # XAI Configuration
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    
    # Application Configuration
    app_title: str = "Reflect Agent Chatbot"
    app_version: str = "1.0.0"
    debug: bool = True
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields without validation errors

# Create global settings instance
settings = Settings() 