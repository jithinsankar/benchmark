# ollamabenchmark/ollama_manager.py
import platform
import subprocess
import os
import requests
import stat
import time
import logging
import json
import re
from typing import Optional, Tuple, Dict, Any

import tarfile
import zipfile
import shutil
import tempfile

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

OLLAMA_DEFAULT_HOST = "http://localhost:11434"

def _get_os_info() -> Tuple[str, str]:
    """Returns (os_type, arch_type) e.g., ('windows', 'amd64')"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Normalize OS names
    os_map = {
        'windows': 'windows',
        'linux': 'linux',
        'darwin': 'darwin'  # macOS
    }
    
    # Normalize architecture names
    arch_map = {
        'x86_64': 'amd64',
        'amd64': 'amd64',
        'aarch64': 'arm64',
        'arm64': 'arm64',
        'armv7l': 'arm64',  # Raspberry Pi, etc.
        'i386': '386',
        'i686': '386',
    }
    
    os_type = os_map.get(system, system)
    arch_type = arch_map.get(machine, machine)
    
    if os_type not in ['windows', 'linux', 'darwin']:
        raise NotImplementedError(f"Unsupported OS: {system}")
    
    logging.info(f"Detected system: {os_type}-{arch_type} (raw: {system}-{machine})")
    return os_type, arch_type

def get_ollama_executable_path() -> Optional[str]:
    """
    Attempts to find the Ollama executable in common locations.
    Returns the path if found, None otherwise.
    """
    os_type, _ = _get_os_info()
    if os_type == "windows":
        # Check program files and PATH
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            path = os.path.join(program_files, "Ollama", "ollama.exe")
            if os.path.exists(path):
                return path
        # Check local app data
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            path = os.path.join(local_app_data, "Ollama", "ollama.exe")
            if os.path.exists(path):
                return path
        # Search PATH
        for path_dir in os.environ["PATH"].split(os.pathsep):
            exe_path = os.path.join(path_dir, "ollama.exe")
            if os.path.exists(exe_path):
                return exe_path

    elif os_type == "darwin" or os_type == "linux":
        # Check common /usr/local/bin, /usr/bin, and user's local bin
        for path_dir in ["/usr/local/bin", "/usr/bin", os.path.expanduser("~/.local/bin")]:
            exe_path = os.path.join(path_dir, "ollama")
            if os.path.exists(exe_path) and os.access(exe_path, os.X_OK):
                return exe_path
    return None

def get_ollama_version() -> Optional[str]:
    """
    Gets the installed Ollama CLI version.
    Returns the version string or None if not found.
    """
    ollama_exe_path = get_ollama_executable_path()
    if not ollama_exe_path:
        logging.warning("Ollama executable not found, cannot get version.")
        return None
    
    try:
        # The command is just the executable path plus "--version"
        command = [ollama_exe_path, "--version"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Example output: "ollama version is 0.1.30"
        output = result.stdout.strip()
        
        # Extract version number using regex
        match = re.search(r"(\d+\.\d+\.\d+)", output)
        if match:
            version = match.group(1)
            logging.info(f"Found Ollama version: {version}")
            return version
        else:
            logging.warning(f"Could not parse Ollama version from output: {output}")
            return None
            
    except FileNotFoundError:
        logging.error("Ollama executable not found at the path returned by get_ollama_executable_path.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting Ollama version: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting Ollama version: {e}")
        return None

def check_ollama_server_status(host: str = OLLAMA_DEFAULT_HOST) -> bool:
    """Checks if the Ollama server is running and accessible."""
    try:
        logging.info(f"Checking Ollama server at {host}...")
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        logging.info("Ollama server is running.")
        return True
    except requests.exceptions.ConnectionError:
        logging.warning(f"Ollama server not reachable at {host}.")
        return False
    except requests.exceptions.Timeout:
        logging.warning(f"Connection to Ollama server at {host} timed out.")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(f"An unexpected error occurred while connecting to Ollama: {e}")
        return False

def get_latest_ollama_release() -> Optional[Dict]:
    """Fetch the latest Ollama release information from GitHub API."""
    try:
        api_url = "https://api.github.com/repos/ollama/ollama/releases/latest"
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'OllamaManager/1.0'
        }
        
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        release_data = response.json()
        logging.info(f"Latest Ollama version: {release_data.get('tag_name', 'Unknown')}")
        return release_data
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch latest Ollama release: {e}")
        return None

def find_matching_asset(assets: list, os_type: str, arch_type: str) -> Optional[Dict]:
    """Find the matching asset from GitHub releases for the given OS and architecture."""
    
    # MODIFIED: Refined patterns to be more specific and avoid incorrect matches like ROCm versions.
    # We prioritize standalone binaries, then generic tarballs, then more general patterns.
    patterns = {
        ('windows', 'amd64'): [
            r'OllamaSetup\.exe$',
            r'.*windows.*amd64.*\.exe$',
        ],
        ('linux', 'amd64'): [
            r'^ollama-linux-amd64$',  # Prefer the standalone binary
            r'^ollama-linux-amd64\.tar\.gz$', # Then the generic tarball
            r'.*linux.*amd64.*\.tar\.gz$',
            r'.*linux.*amd64.*(?<!-rocm)\.tgz$', # Avoid rocm tgz if possible
            r'.*linux.*amd64.*',
        ],
        ('linux', 'arm64'): [
            r'^ollama-linux-arm64$',
            r'^ollama-linux-arm64\.tar\.gz$',
            r'.*linux.*arm64.*\.tar\.gz$',
            r'.*linux.*arm64.*',
        ],
        ('darwin', 'amd64'): [
            r'^ollama-darwin$',
            r'.*darwin.*\.zip$',
        ],
        ('darwin', 'arm64'): [
            r'.*darwin-arm64\.zip$',
            r'.*darwin.*arm64.*\.zip$',
        ],
    }
    
    key = (os_type, arch_type)
    if key not in patterns:
        logging.warning(f"No patterns defined for {os_type}-{arch_type}")
        return None
    
    # Try to find matching asset
    for pattern in patterns[key]:
        for asset in assets:
            asset_name = asset.get('name', '').lower()
            if re.match(pattern.lower(), asset_name):
                logging.info(f"Found matching asset: {asset['name']}")
                return asset
    
    logging.warning(f"Could not find a preferred asset for {os_type}-{arch_type}. Falling back to any match.")
    # Fallback: look for any asset containing OS or arch info
    fallback_terms = [os_type, arch_type]
    for asset in assets:
        asset_name = asset.get('name', '').lower()
        if any(term in asset_name for term in fallback_terms):
            logging.warning(f"Using fallback match: {asset['name']}")
            return asset
            
    return None

def get_file_extension(os_type: str, asset_name: str = None) -> str:
    """Determine the appropriate file extension based on OS and asset name."""
    if asset_name:
        if asset_name.endswith('.tar.gz'):
            return '.tar.gz'
        # Use the actual extension from the asset
        if '.' in asset_name:
            # Handle cases like .exe, .zip, .tgz
            return os.path.splitext(asset_name)[1]
    
    # Default extensions by OS
    extensions = {
        'windows': '.exe',
        'linux': '',  # Usually no extension for Linux binaries
        'darwin': '.zip',  # macOS often uses .zip or .pkg
    }
    
    return extensions.get(os_type, '')

def download_with_progress(url: str, local_filename: str) -> bool:
    """Download a file with progress indication."""
    try:
        headers = {
            'User-Agent': 'OllamaManager/1.0'
        }
        
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            mb_downloaded = downloaded_size / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"  Downloading: {mb_downloaded:.2f}MB / {mb_total:.2f}MB ({progress:.1f}%)", end='\r')
                        else:
                            mb_downloaded = downloaded_size / (1024 * 1024)
                            print(f"  Downloaded: {mb_downloaded:.2f}MB", end='\r')
        
        print()  # New line after progress
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed: {e}")
        return False

def download_ollama_installer(os_type: str = None, arch_type: str = None, 
                            download_dir: str = None, force_version: str = None) -> Optional[str]:
    """
    Downloads the Ollama installer/binary for the specified or detected platform.
    
    Args:
        os_type: Target OS ('windows', 'linux', 'darwin'). Auto-detected if None.
        arch_type: Target architecture ('amd64', 'arm64', '386'). Auto-detected if None.
        download_dir: Directory to save the file. Uses current directory if None.
        force_version: Specific version to download (e.g., 'v0.9.1'). Uses latest if None.
    
    Returns:
        Path to downloaded file or None if failed.
    """
    
    # Auto-detect system info if not provided
    if os_type is None or arch_type is None:
        detected_os, detected_arch = _get_os_info()
        os_type = os_type or detected_os
        arch_type = arch_type or detected_arch
    
    # Set download directory
    if download_dir is None:
        download_dir = os.getcwd()
    
    logging.info(f"Preparing to download Ollama for {os_type}-{arch_type}...")
    
    # Get release information
    if force_version:
        # Download specific version
        api_url = f"https://api.github.com/repos/ollama/ollama/releases/tags/{force_version}"
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            release_data = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch version {force_version}: {e}")
            return None
    else:
        # Get latest release
        release_data = get_latest_ollama_release()
        if not release_data:
            return None
    
    # Find matching asset
    assets = release_data.get('assets', [])
    if not assets:
        logging.error("No assets found in release")
        return None
    
    matching_asset = find_matching_asset(assets, os_type, arch_type)
    if not matching_asset:
        logging.error(f"No matching asset found for {os_type}-{arch_type}")
        logging.info("Available assets:")
        for asset in assets:
            logging.info(f"  - {asset['name']}")
        return None
    
    # Prepare download
    download_url = matching_asset['browser_download_url']
    asset_name = matching_asset['name']
    file_extension = get_file_extension(os_type, asset_name)
    
    # Create local filename
    base_name = f"ollama-{release_data.get('tag_name', 'unknown')}-{os_type}-{arch_type}"
    local_filename = os.path.join(download_dir, base_name + file_extension)
    
    # Check if file already exists
    if os.path.exists(local_filename):
        logging.info(f"File already exists: {local_filename}")
        return local_filename
    
    # Download the file
    print(f"  Downloading: {asset_name}")
    print(f"  From: {download_url}")
    print(f"  To: {local_filename}")
    
    if download_with_progress(download_url, local_filename):
        logging.info(f"Successfully downloaded Ollama to: {local_filename}")
        
        # Make executable on Unix-like systems
        if os_type in ['linux', 'darwin'] and not file_extension:
            try:
                os.chmod(local_filename, 0o755)
                logging.info("Made file executable")
            except OSError as e:
                logging.warning(f"Could not make file executable: {e}")
        
        return local_filename
    else:
        # Clean up partial download
        if os.path.exists(local_filename):
            try:
                os.remove(local_filename)
            except OSError:
                pass
        return None

def list_available_versions(limit: int = 10) -> Optional[list]:
    """List available Ollama versions from GitHub releases."""
    try:
        api_url = f"https://api.github.com/repos/ollama/ollama/releases?per_page={limit}"
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'OllamaManager/1.0'
        }
        
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        releases = response.json()
        versions = []
        
        for release in releases:
            versions.append({
                'tag': release.get('tag_name', 'Unknown'),
                'name': release.get('name', 'Unnamed'),
                'published_at': release.get('published_at', ''),
                'prerelease': release.get('prerelease', False),
                'draft': release.get('draft', False)
            })
        
        return versions
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch releases: {e}")
        return None

def install_ollama_cli(installer_path: str) -> bool:
    """
    Installs Ollama. For archives (.zip, .tgz), it extracts the binary.
    For Linux/macOS, it moves the binary to a PATH directory.
    For Windows, it launches the installer.
    """
    os_type, _ = _get_os_info()
    logging.info(f"Attempting to install Ollama from: {installer_path}")

    if os_type == "windows":
        logging.info("Windows detected. Please run the downloaded installer manually if it doesn't auto-launch.")
        logging.info(f"Executing: {installer_path}")
        try:
            subprocess.Popen([installer_path], shell=True)
            logging.info("Ollama installer launched. Please follow the on-screen instructions.")
            return True
        except Exception as e:
            logging.error(f"Failed to launch Windows installer: {e}")
            return False

    # --- COMPLETELY REWRITTEN LOGIC FOR LINUX/MACOS ---
    elif os_type in ["darwin", "linux"]:
        target_dir = os.path.expanduser("~/.local/bin")
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "ollama")

        try:
            # Check if the downloaded file is an archive and needs extraction
            if installer_path.endswith((".tar.gz", ".tgz")):
                logging.info(f"Extracting archive: {installer_path}")
                with tempfile.TemporaryDirectory() as extract_dir:
                    with tarfile.open(installer_path, "r:gz") as tar:
                        tar.extractall(path=extract_dir)
                        # The actual executable is expected to be named 'ollama' inside
                        extracted_binary = os.path.join(extract_dir, "ollama")
                        if not os.path.exists(extracted_binary):
                             # Sometimes it's in a subdirectory
                            for root, _, files in os.walk(extract_dir):
                                if "ollama" in files:
                                    extracted_binary = os.path.join(root, "ollama")
                                    break
                        
                        if os.path.exists(extracted_binary):
                             logging.info(f"Found executable in archive at {extracted_binary}")
                             # Move the extracted binary to the target path
                             shutil.move(extracted_binary, target_path)
                        else:
                             logging.error("Could not find 'ollama' executable inside the archive.")
                             return False
            elif installer_path.endswith(".zip"): # For macOS zip files
                 logging.info(f"Extracting zip archive: {installer_path}")
                 with tempfile.TemporaryDirectory() as extract_dir:
                    with zipfile.ZipFile(installer_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                        extracted_binary = os.path.join(extract_dir, "ollama")
                        if not os.path.exists(extracted_binary):
                            for root, _, files in os.walk(extract_dir):
                                if "ollama" in files:
                                    extracted_binary = os.path.join(root, "ollama")
                                    break

                        if os.path.exists(extracted_binary):
                            logging.info(f"Found executable in archive at {extracted_binary}")
                            shutil.move(extracted_binary, target_path)
                        else:
                            logging.error("Could not find 'ollama' executable inside the zip file.")
                            return False
            else:
                # The downloaded file is the binary itself
                logging.info("Downloaded file is a standalone binary.")
                shutil.move(installer_path, target_path)

            # Make the final binary executable
            logging.info(f"Setting execute permissions for {target_path}")
            os.chmod(target_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            
            logging.info(f"Ollama binary installed to: {target_path}")
            logging.info(f"Please ensure '{target_dir}' is in your system's PATH.")
            
            # Clean up the downloaded installer file if it wasn't moved
            if os.path.exists(installer_path) and installer_path != target_path:
                 os.remove(installer_path)

            return True
        except Exception as e:
            logging.error(f"Failed to install Ollama binary: {e}", exc_info=True)
            return False

    return False


def start_ollama_server() -> bool:
    """Attempts to start the Ollama server by running 'ollama serve'."""
    if check_ollama_server_status():
        logging.info("Ollama server is already running.")
        return True

    ollama_exe_path = get_ollama_executable_path()
    if not ollama_exe_path:
        logging.error("Ollama executable not found. Cannot start server.")
        return False

    logging.info(f"Attempting to start Ollama server using: '{ollama_exe_path} serve'")
    try:
        command = [ollama_exe_path, "serve"]
        # Run the server as a detached background process
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        logging.info("Server process launched. Waiting for it to become available...")
        time.sleep(8)  # Give the server time to initialize
        
        if check_ollama_server_status():
            logging.info("Ollama server started successfully.")
            return True
        else:
            logging.error("Ollama server did not start or is not reachable after launch attempt.")
            return False
    except Exception as e:
        logging.error(f"Error starting Ollama server: {e}")
        return False


def ensure_ollama_ready() -> bool:
    """
    Main function to ensure Ollama is installed and running.
    Guides the user through installation if necessary.
    """
    if check_ollama_server_status():
        logging.info("Ollama is already running and ready.")
        return True

    logging.info("Ollama server not found or not running.")
    ollama_exe_path = get_ollama_executable_path()

    if ollama_exe_path:
        logging.info(f"Ollama executable found at: {ollama_exe_path}")
        if start_ollama_server():
            return True
        else:
            logging.error("Failed to start Ollama server. Please start it manually.")
            return False
    else:
        logging.info("Ollama executable not found. Attempting to download and install.")
        os_type, arch_type = _get_os_info()
        download_path = download_ollama_installer(os_type, arch_type)
        if download_path:
            if install_ollama_cli(download_path):
                logging.info("Ollama installation process initiated. Please follow its instructions.")
                logging.info("Attempting to start Ollama server now...")
                # Give installer time to finish if it's a GUI installer
                time.sleep(10)
                if start_ollama_server():
                    logging.info("Ollama is now installed and running.")
                    return True
                else:
                    logging.error("Ollama server did not start automatically after installation. Please start it manually.")
                    return False
            else:
                logging.error("Ollama installation failed. Please install manually from https://ollama.com/download")
                return False
        else:
            logging.error("Failed to download Ollama installer. Please check your internet connection or install manually from https://ollama.com/download")
            return False

def pull_ollama_model(model_name: str) -> bool:
    """Pulls a specified Ollama model."""
    import ollama # Import here to avoid circular dependency or if ollama not installed yet

    try:
        # Check if model exists
        available_models = [m['model'] for m in ollama.list()['models']]
        if model_name in available_models:
            logging.info(f"Model '{model_name}' is already available locally.")
            return True

        logging.info(f"Pulling model: {model_name}...")
        # Use ollama.pull with streaming for progress
        for progress in ollama.pull(model_name, stream=True):
            if 'total' in progress and 'completed' in progress:
                percentage = (progress['completed'] / progress['total']) * 100 if progress['total'] else 0
                print(f"  Pulling {model_name}: {progress['completed'] / 1024 / 1024:.2f}MB / {progress['total'] / 1024 / 1024:.2f}MB ({percentage:.1f}%)", end='\r')
            elif 'status' in progress:
                print(f"  Status: {progress['status']}", end='\r')
        print("\n") # Newline after progress
        logging.info(f"Model '{model_name}' pulled successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to pull model '{model_name}': {e}")
        logging.info(f"You may need to pull it manually using: ollama pull {model_name}")
        return False

if __name__ == "__main__":
    logging.info("--- Enhanced Ollama Manager Test ---")
    
    # Show available versions
    print("Recent Ollama versions:")
    versions = list_available_versions(5)
    if versions:
        for v in versions:
            status = " (prerelease)" if v['prerelease'] else ""
            print(f"  - {v['tag']}: {v['name']}{status}")
    print()
    
    if ensure_ollama_ready():
        logging.info("Ollama is ready. Attempting to pull a test model (e.g., 'gemma3:1b').")
        pull_ollama_model("gemma3:1b") # Example model
    else:
        logging.error("Ollama is not ready. Please resolve issues above.")
