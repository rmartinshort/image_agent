from typing import Dict
from dotenv import load_dotenv
import os


def load_secrets(env_path: str = ".env") -> Dict[str, str]:
    """
    Load API keys from the specified environment file.

    This function loads environment variables from the default `.env` file
    and the specified `env_path`. It retrieves the OpenAI and Tavily API keys.

    Args:
        env_path (str): The path to the environment file. Defaults to ".env".

    Returns:
        Dict[str, str]: A dictionary containing the loaded API keys.
    """
    # both calls are needed here
    load_dotenv()
    load_dotenv(dotenv_path=env_path)

    return {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
