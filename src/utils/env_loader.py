"""
Utility to load environment variables from a .env file.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Union

def load_environment(
    env_file: Union[str, Path] = Path(".env")
) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_file (Union[str, Path]): Path to the .env file. Defaults to ".env".
    """
    # Check if the file exists
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"The specified .env file does not exist: {env_file}")

    # Load the environment variables from the .env file
    load_dotenv(dotenv_path=env_file)
    
def get_env_variable(
    var_name: str,
    default: Union[str, None] = None,
    required: bool = False    
) -> str:   
    """
    Get an environment variable.

    Args:
        var_name (str): Name of the environment variable.
        default (Union[str, None]): Default value if the variable is not found. Defaults to None.
        required (bool): If True, raises an error if the variable is not found. Defaults to False.

    Returns:
        str: Value of the environment variable or default value.
    """
    
    value = os.getenv(var_name, default)
    
    if required and value is None:
        raise EnvironmentError(f"Environment variable '{var_name}' is required but not set.")
    return value