"""Environment check for CUDA-GDB-CLI."""

import os
import re
import shutil
import subprocess
import platform
from typing import Dict, Any, Optional


def check_environment() -> Dict[str, Any]:
    """Check the environment for CUDA debugging requirements.

    Returns:
        Dict with check results
    """
    results = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
    }

    # Check cuda-gdb
    cuda_gdb_path = shutil.which("cuda-gdb")
    if cuda_gdb_path:
        results["cuda_gdb_path"] = cuda_gdb_path
        results["cuda_gdb_python"] = _check_cuda_gdb_python(cuda_gdb_path)
    else:
        results["cuda_gdb_path"] = None
        results["cuda_gdb_error"] = "cuda-gdb not found. Install CUDA Toolkit."

    # Check CUDA version
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        results["nvcc_path"] = nvcc_path
        results["cuda_version"] = _get_cuda_version(nvcc_path)
    else:
        results["nvcc_path"] = None

    # Check GPU driver
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        results["nvidia_smi"] = nvidia_smi
        results.update(_get_gpu_info(nvidia_smi))
    else:
        results["nvidia_smi"] = None

    # Check GDB
    gdb_path = shutil.which("gdb")
    if gdb_path:
        results["gdb_path"] = gdb_path

    return results


def _check_cuda_gdb_python(cuda_gdb_path: str) -> bool:
    """Check if cuda-gdb has Python support."""
    try:
        result = subprocess.run(
            [cuda_gdb_path, "-nx", "-q", "-batch", "-ex", "python print('OK')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return "OK" in result.stdout
    except Exception:
        return False


def _get_cuda_version(nvcc_path: str) -> Optional[str]:
    """Get CUDA version from nvcc."""
    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def _get_gpu_info(nvidia_smi: str) -> Dict[str, Any]:
    """Get GPU information from nvidia-smi."""
    info = {}
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version,name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(",")
                info["gpu_driver"] = parts[0].strip() if len(parts) > 0 else None
                info["gpu_name"] = parts[1].strip() if len(parts) > 1 else None
                info["compute_capability"] = parts[2].strip() if len(parts) > 2 else None
    except Exception:
        pass
    return info


def check_core_file(core_path: str) -> Dict[str, Any]:
    """Check if a core dump file is valid.

    Args:
        core_path: Path to core dump file

    Returns:
        Dict with check results
    """
    results = {
        "path": core_path,
        "exists": os.path.exists(core_path),
    }

    if results["exists"]:
        results["size"] = os.path.getsize(core_path)
        # Could add more checks here (file type, etc.)

    return results


def check_binary(binary_path: str) -> Dict[str, Any]:
    """Check if a binary file exists and is executable.

    Args:
        binary_path: Path to binary file

    Returns:
        Dict with check results
    """
    results = {
        "path": binary_path,
        "exists": os.path.exists(binary_path),
    }

    if results["exists"]:
        results["is_file"] = os.path.isfile(binary_path)
        results["is_executable"] = os.access(binary_path, os.X_OK)

    return results
