"""Shared GPU/CPU device selection for the audio pipeline's ML steps."""
import torch


def get_device() -> str:
    """"cuda" if an NVIDIA/CUDA GPU is available, else "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"
