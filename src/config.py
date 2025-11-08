import os
from pathlib import Path
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

env_path = project_root / '.env'

load_dotenv(dotenv_path=env_path)

def get_env_variable(var_name: str, required: bool = True) -> str:
    """Safely retrieves environment variable with validation."""
    value = os.getenv(var_name)
    if required and not value:
        raise ValueError(
            f"❌ Environment variable '{var_name}' is not set!\n"
            f"Please add it to your .env file.\n"
            f"See .env.example for guidance."
        )
    return value

class Config:

    OPENWEATHERMAP_API_KEY = get_env_variable('OPENWEATHERMAP_API_KEY')
    NEWS_API_KEY = get_env_variable('NEWS_API_KEY')
    HF_TOKEN_MODEL = get_env_variable('HF_TOKEN_MODEL')

print(f"   Configuration loaded successfully!")
print(f"   OPENWEATHERMAP_API_KEY: {Config.OPENWEATHERMAP_API_KEY[:5]}...")
print(f"   NEWS_API_KEY: {Config.NEWS_API_KEY[:5]}...")
print(f"   HF_TOKEN_MODEL: {Config.HF_TOKEN_MODEL[:10]}...")