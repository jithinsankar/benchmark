# benchmark/sys_info.py
import platform
import psutil
import logging
import subprocess
import re
from typing import Dict, Any, Optional

# For NVIDIA GPU info (optional, requires `pynvml` and NVIDIA driver)
try:
    # Try the newer import structure first
    try:
        import pynvml
        nvmlInit = pynvml.nvmlInit
        nvmlDeviceGetCount = pynvml.nvmlDeviceGetCount
        nvmlDeviceGetHandleByIndex = pynvml.nvmlDeviceGetHandleByIndex
        nvmlDeviceGetName = pynvml.nvmlDeviceGetName
        nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo
        nvmlDeviceGetUtilizationRates = pynvml.nvmlDeviceGetUtilizationRates
        nvmlShutdown = pynvml.nvmlShutdown
        NVMLError = pynvml.NVMLError
        _HAS_NVML = True
    except AttributeError:
        # Fallback to the older import structure
        from pynvml.pynvml import *
        _HAS_NVML = True
except ImportError:
    _HAS_NVML = False
    logging.warning("pynvml not found. NVIDIA GPU details will be limited.")
except Exception as error:
    _HAS_NVML = False
    logging.warning(f"pynvml initialization error: {error}. NVIDIA GPU details will be limited.")


def _get_windows_system_info() -> Dict[str, Any]:
    """Get Windows-specific system information using WMI."""
    info = {}
    
    try:
        # Get computer system info
        result = subprocess.run([
            "wmic", "computersystem", "get", 
            "Manufacturer,Model,TotalPhysicalMemory,SystemType", "/format:csv"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
            if len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(',')]
                data = [d.strip() for d in lines[1].split(',')]
                
                for i, header in enumerate(headers):
                    if i < len(data) and header:
                        if header == "Manufacturer":
                            info["manufacturer"] = data[i]
                        elif header == "Model":
                            info["model"] = data[i]
                        elif header == "SystemType":
                            info["system_type"] = data[i]
    except Exception as e:
        logging.debug(f"Could not get Windows system info: {e}")
    
    # Get BIOS/UEFI info
    try:
        result = subprocess.run([
            "wmic", "bios", "get", "Manufacturer,SMBIOSBIOSVersion", "/format:csv"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
            if len(lines) > 1:
                data = [d.strip() for d in lines[1].split(',')]
                if len(data) >= 3:
                    info["bios_manufacturer"] = data[1]
                    info["bios_version"] = data[2]
    except Exception as e:
        logging.debug(f"Could not get BIOS info: {e}")
    
    # Get motherboard info
    try:
        result = subprocess.run([
            "wmic", "baseboard", "get", "Manufacturer,Product,Version", "/format:csv"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
            if len(lines) > 1:
                data = [d.strip() for d in lines[1].split(',')]
                if len(data) >= 4:
                    info["motherboard_manufacturer"] = data[1]
                    info["motherboard_model"] = data[2]
                    info["motherboard_version"] = data[3]
    except Exception as e:
        logging.debug(f"Could not get motherboard info: {e}")
    
    return info


def _get_linux_system_info() -> Dict[str, Any]:
    """Get Linux-specific system information using DMI."""
    info = {}
    
    dmi_mappings = {
        "sys_vendor": "manufacturer",
        "product_name": "model",
        "product_version": "model_version",
        "bios_vendor": "bios_manufacturer",
        "bios_version": "bios_version",
        "board_vendor": "motherboard_manufacturer",
        "board_name": "motherboard_model",
        "board_version": "motherboard_version"
    }
    
    for dmi_key, info_key in dmi_mappings.items():
        try:
            with open(f"/sys/class/dmi/id/{dmi_key}", "r") as f:
                value = f.read().strip()
                if value and value != "Unknown" and value != "Not Specified":
                    info[info_key] = value
        except (FileNotFoundError, PermissionError):
            continue
        except Exception as e:
            logging.debug(f"Could not read DMI {dmi_key}: {e}")
    
    # Try lscpu for additional CPU info
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Model name:' in line:
                    cpu_model = line.split(':', 1)[1].strip()
                    info["cpu_model_detailed"] = cpu_model
                    break
    except Exception as e:
        logging.debug(f"Could not get detailed CPU info: {e}")
    
    return info


def _get_macos_system_info() -> Dict[str, Any]:
    """Get macOS-specific system information using system_profiler."""
    info = {}
    
    try:
        result = subprocess.run([
            "system_profiler", "SPHardwareDataType", "-json"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            hardware_info = data.get("SPHardwareDataType", [{}])[0]
            
            info["manufacturer"] = "Apple"
            info["model"] = hardware_info.get("machine_name", "Unknown Mac")
            info["model_identifier"] = hardware_info.get("machine_model", "")
            info["serial_number"] = hardware_info.get("serial_number", "")
            info["cpu_model_detailed"] = hardware_info.get("chip_type", 
                                                         hardware_info.get("cpu_type", ""))
            
    except Exception as e:
        logging.debug(f"Could not get macOS system info: {e}")
        # Fallback to basic commands
        try:
            result = subprocess.run(["sysctl", "-n", "hw.model"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info["model_identifier"] = result.stdout.strip()
                info["manufacturer"] = "Apple"
        except Exception:
            pass
    
    return info


def _classify_device_category(system_info: Dict[str, Any]) -> str:
    """Classify the device into categories for benchmarking grouping."""
    manufacturer = system_info.get("manufacturer", "").lower()
    model = system_info.get("model", "").lower()
    
    # Apple devices
    if "apple" in manufacturer or "mac" in model:
        if "macbook" in model:
            if "pro" in model:
                return "MacBook Pro"
            elif "air" in model:
                return "MacBook Air"
            else:
                return "MacBook"
        elif "imac" in model:
            if "pro" in model:
                return "iMac Pro"
            else:
                return "iMac"
        elif "mac" in model:
            if "pro" in model:
                return "Mac Pro"
            elif "mini" in model:
                return "Mac Mini"
            elif "studio" in model:
                return "Mac Studio"
        return "Apple Device"
    
    # Common laptop/desktop patterns
    laptop_indicators = ["laptop", "notebook", "portable", "book", "zenbook", 
                        "thinkpad", "inspiron", "pavilion", "envy", "spectre", 
                        "xps", "surface", "yoga", "flex"]
    
    desktop_indicators = ["desktop", "tower", "workstation", "optiplex", 
                         "precision", "vostro", "alienware"]
    
    model_lower = model.lower()
    
    # Check for specific laptop series
    if any(indicator in model_lower for indicator in laptop_indicators):
        if "zenbook" in model_lower:
            return "ASUS ZenBook"
        elif "thinkpad" in model_lower:
            return "Lenovo ThinkPad"
        elif "xps" in model_lower:
            return "Dell XPS"
        elif "surface" in model_lower:
            return "Microsoft Surface"
        elif "pavilion" in model_lower:
            return "HP Pavilion"
        elif "envy" in model_lower:
            return "HP Envy"
        elif "spectre" in model_lower:
            return "HP Spectre"
        elif "inspiron" in model_lower:
            return "Dell Inspiron"
        elif "yoga" in model_lower:
            return "Lenovo Yoga"
        else:
            return f"{manufacturer.title()} Laptop"
    
    # Check for desktop patterns
    elif any(indicator in model_lower for indicator in desktop_indicators):
        return f"{manufacturer.title()} Desktop"
    
    # Manufacturer-based classification
    elif "asus" in manufacturer:
        return "ASUS Computer"
    elif "dell" in manufacturer:
        return "Dell Computer"
    elif "hp" in manufacturer or "hewlett" in manufacturer:
        return "HP Computer"
    elif "lenovo" in manufacturer:
        return "Lenovo Computer"
    elif "microsoft" in manufacturer:
        return "Microsoft Device"
    elif "acer" in manufacturer:
        return "Acer Computer"
    elif "msi" in manufacturer:
        return "MSI Computer"
    
    return "Generic Computer"


def get_system_info() -> Dict[str, Any]:
    """Collects detailed system hardware information for benchmarking."""
    
    # Get platform-specific system info
    if platform.system() == "Windows":
        platform_info = _get_windows_system_info()
    elif platform.system() == "Linux":
        platform_info = _get_linux_system_info()
    elif platform.system() == "Darwin":  # macOS
        platform_info = _get_macos_system_info()
    else:
        platform_info = {}
    
    # Build comprehensive system info
    info = {
        "device": {
            "manufacturer": platform_info.get("manufacturer", "Unknown"),
            "model": platform_info.get("model", "Unknown"),
            "model_version": platform_info.get("model_version", ""),
            "model_identifier": platform_info.get("model_identifier", ""),
            "category": "",  # Will be filled below
            "bios_manufacturer": platform_info.get("bios_manufacturer", ""),
            "bios_version": platform_info.get("bios_version", ""),
            "motherboard_manufacturer": platform_info.get("motherboard_manufacturer", ""),
            "motherboard_model": platform_info.get("motherboard_model", ""),
            "system_type": platform_info.get("system_type", ""),
        },
        "os": {
            "name": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
        },
        "cpu": {
            "name": platform.processor(),
            "model_detailed": platform_info.get("cpu_model_detailed", platform.processor()),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency_ghz": round(psutil.cpu_freq().max / 1000, 2) if psutil.cpu_freq() else None,
            "architecture": platform.machine(),
        },
        "ram": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "usage_percent": psutil.virtual_memory().percent,
        },
        "storage": _get_storage_info(),
        "gpu": []
    }
    
    # Classify device category for benchmarking grouping
    info["device"]["category"] = _classify_device_category(info["device"])
    
    # NVIDIA GPU Info
    if _HAS_NVML:
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                gpu_name = nvmlDeviceGetName(handle)
                # Handle both string and bytes return types
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
                
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                total_memory_mb = memory_info.total / (1024**2)
                
                gpu_info = {
                    "name": gpu_name,
                    "vendor": "NVIDIA",
                    "vram_mb": round(total_memory_mb, 2),
                    "vram_gb": round(total_memory_mb / 1024, 2),
                    "index": i,
                    # Uncomment if you want live usage metrics
                    # "usage_percent": nvmlDeviceGetUtilizationRates(handle).gpu,
                    # "memory_usage_mb": round(memory_info.used / (1024**2), 2)
                }
                info["gpu"].append(gpu_info)
            nvmlShutdown()
        except Exception as error:
            logging.warning(f"Could not query NVIDIA GPUs: {error}. Skipping NVIDIA info.")

    # Generic GPU Info (for AMD/Intel or if NVML fails)
    if not info["gpu"]: # If no NVIDIA GPUs were found/queried
        generic_gpu_info = _get_generic_gpu_info()
        if generic_gpu_info:
            info["gpu"].extend(generic_gpu_info)
        else:
            # Placeholder for unknown GPU
            info["gpu"].append({
                "name": "Unknown/Integrated GPU",
                "vendor": "N/A",
                "vram_mb": None,
                "vram_gb": None
            })

    return info


def _get_storage_info() -> list:
    """Get storage device information."""
    storage_devices = []
    
    try:
        partitions = psutil.disk_partitions()
        processed_devices = set()
        
        for partition in partitions:
            try:
                if partition.device not in processed_devices:
                    usage = psutil.disk_usage(partition.mountpoint)
                    storage_devices.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "filesystem": partition.fstype,
                        "total_gb": round(usage.total / (1024**3), 2),
                        "used_gb": round(usage.used / (1024**3), 2),
                        "free_gb": round(usage.free / (1024**3), 2),
                        "usage_percent": round((usage.used / usage.total) * 100, 1)
                    })
                    processed_devices.add(partition.device)
            except (PermissionError, FileNotFoundError):
                continue
    except Exception as e:
        logging.debug(f"Could not get storage info: {e}")
    
    return storage_devices


def _get_generic_gpu_info():
    """Attempt to get basic GPU info for non-NVIDIA GPUs."""
    gpu_info = []
    
    if platform.system() == "Windows":
        try:
            import subprocess
            # Try to get GPU info using wmic
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:csv"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip() and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            name = parts[2].strip()
                            ram = parts[1].strip()
                            if name and name != "Name":
                                vram_mb = None
                                vram_gb = None
                                if ram and ram.isdigit():
                                    vram_mb = round(int(ram) / (1024**2), 2)
                                    vram_gb = round(vram_mb / 1024, 2)
                                
                                vendor = "Unknown"
                                if "NVIDIA" in name.upper():
                                    vendor = "NVIDIA"
                                elif "AMD" in name.upper() or "ATI" in name.upper():
                                    vendor = "AMD"
                                elif "INTEL" in name.upper():
                                    vendor = "Intel"
                                
                                gpu_info.append({
                                    "name": name,
                                    "vendor": vendor,
                                    "vram_mb": vram_mb,
                                    "vram_gb": vram_gb
                                })
        except Exception as e:
            logging.debug(f"Could not get Windows GPU info: {e}")
    
    elif platform.system() == "Linux":
        try:
            import subprocess
            # Try to get GPU info using lspci
            result = subprocess.run(
                ["lspci", "-nn"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA compatible controller' in line or '3D controller' in line:
                        # Extract GPU name (simplified)
                        if ':' in line:
                            gpu_name = line.split(':', 2)[-1].strip()
                            vendor = "Unknown"
                            if "NVIDIA" in gpu_name.upper():
                                vendor = "NVIDIA"
                            elif "AMD" in gpu_name.upper() or "ATI" in gpu_name.upper():
                                vendor = "AMD"
                            elif "INTEL" in gpu_name.upper():
                                vendor = "Intel"
                            
                            gpu_info.append({
                                "name": gpu_name,
                                "vendor": vendor,
                                "vram_mb": None,  # Hard to get without specific tools
                                "vram_gb": None
                            })
        except Exception as e:
            logging.debug(f"Could not get Linux GPU info: {e}")
    
    elif platform.system() == "Darwin":  # macOS
        try:
            result = subprocess.run([
                "system_profiler", "SPDisplaysDataType", "-json"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", [])
                
                for display in displays:
                    gpu_name = display.get("sppci_model", "Unknown GPU")
                    vram_mb = display.get("sppci_vram", "")
                    
                    # Parse VRAM
                    vram_value = None
                    vram_gb = None
                    if vram_mb:
                        # Handle formats like "8 GB", "1024 MB"
                        vram_str = str(vram_mb).upper()
                        if "GB" in vram_str:
                            vram_value = float(re.findall(r'[\d.]+', vram_str)[0]) * 1024
                        elif "MB" in vram_str:
                            vram_value = float(re.findall(r'[\d.]+', vram_str)[0])
                        
                        if vram_value:
                            vram_gb = round(vram_value / 1024, 2)
                    
                    vendor = "Unknown"
                    if "NVIDIA" in gpu_name.upper():
                        vendor = "NVIDIA"
                    elif "AMD" in gpu_name.upper() or "ATI" in gpu_name.upper():
                        vendor = "AMD"
                    elif "INTEL" in gpu_name.upper():
                        vendor = "Intel"
                    elif "APPLE" in gpu_name.upper() or "M1" in gpu_name.upper() or "M2" in gpu_name.upper():
                        vendor = "Apple"
                    
                    gpu_info.append({
                        "name": gpu_name,
                        "vendor": vendor,
                        "vram_mb": vram_value,
                        "vram_gb": vram_gb
                    })
        except Exception as e:
            logging.debug(f"Could not get macOS GPU info: {e}")
    
    return gpu_info


def get_benchmark_identifier() -> str:
    """Generate a unique identifier for benchmarking results."""
    info = get_system_info()
    
    # Create a readable benchmark identifier
    category = info["device"]["category"]
    cpu = info["cpu"]["name"][:50]  # Truncate long CPU names
    ram_gb = info["ram"]["total_gb"]
    gpu_name = info["gpu"][0]["name"][:50] if info["gpu"] else "No GPU"
    
    return f"{category}_{cpu}_{ram_gb}GB_{gpu_name}".replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== System Information for Benchmarking ===")
    system_details = get_system_info()
    
    import json
    print(json.dumps(system_details, indent=2))
    
    print(f"\n=== Benchmark Identifier ===")
    print(get_benchmark_identifier())
    
    print(f"\n=== Device Category for Grouping ===")
    print(f"Category: {system_details['device']['category']}")
    print(f"Full Model: {system_details['device']['manufacturer']} {system_details['device']['model']}")