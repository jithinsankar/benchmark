
import requests
import logging
from typing import Optional
from importlib import metadata
import sys

def get_current_version() -> Optional[str]:
    """
    Gets the currently installed version of the ollamabench package.
    """
    try:
        return metadata.version("ollamabench")
    except metadata.PackageNotFoundError:
        logging.warning("Could not determine the current version of the package.")
        return None

def get_latest_version(repo_url: str) -> Optional[str]:
    """
    Fetches the latest release version from a GitHub repository.
    """
    try:
        api_url = f"https://api.github.com/repos/{repo_url}/releases/latest"
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()["tag_name"]
    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not check for new version: {e}")
        return None

def check_for_updates_and_exit(repo_url: str):
    """
    Checks for updates and exits if the current version is not the latest.
    """
    current_version = get_current_version()
    latest_version = get_latest_version(repo_url)

    if current_version and latest_version and current_version != latest_version:
        print(f"Your version {current_version} is outdated. The latest version is {latest_version}.")
        print("Please update to the latest version to continue.")
        sys.exit(1)
