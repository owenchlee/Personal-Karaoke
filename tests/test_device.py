from unittest.mock import patch

from audio_pipeline.device import get_device


def test_get_device_returns_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert get_device() == "cuda"


def test_get_device_returns_cpu_when_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        assert get_device() == "cpu"
