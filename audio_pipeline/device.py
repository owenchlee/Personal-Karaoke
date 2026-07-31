"""Shared GPU/CPU device selection for the audio pipeline's ML steps."""
import os
import sys

import torch

if sys.platform == "win32" and torch.cuda.is_available():
    # ctranslate2 (faster-whisper's backend, used by lyrics_extraction.py) links
    # its own CUDA 12 cuBLAS at runtime, separate from whatever CUDA version
    # torch itself bundles -- without this, the first real GPU transcription
    # raises "cublas64_12.dll is not found or cannot be loaded" even though
    # torch.cuda.is_available() is True and torch-based stages (Demucs,
    # forced_alignment) work fine. The nvidia-cublas-cu12 pip package ships
    # just that DLL; wire it up once, here, since every GPU-capable stage in
    # this pipeline already goes through get_device().
    try:
        import nvidia.cublas

        _cublas_bin = os.path.join(nvidia.cublas.__path__[0], "bin")
        os.add_dll_directory(_cublas_bin)
        os.environ["PATH"] = _cublas_bin + os.pathsep + os.environ["PATH"]
    except ImportError:
        pass


def get_device() -> str:
    """"cuda" if an NVIDIA/CUDA GPU is available, else "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_info() -> dict:
    """Display-friendly detail on the device `get_device()` would select --
    lets the frontend show "NVIDIA GPU detected" instead of leaving users to
    guess whether processing is actually using their GPU.
    """
    available = torch.cuda.is_available()
    return {
        "available": available,
        "name": torch.cuda.get_device_name(0) if available else None,
    }
