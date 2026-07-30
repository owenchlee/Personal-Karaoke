"""Shared GPU/CPU device selection for the audio pipeline's ML steps."""
import torch


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
