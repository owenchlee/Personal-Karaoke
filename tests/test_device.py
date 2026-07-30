from unittest.mock import patch

from audio_pipeline.device import get_device, get_device_info


def test_get_device_returns_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert get_device() == "cuda"


def test_get_device_returns_cpu_when_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        assert get_device() == "cpu"


def test_get_device_info_reports_the_gpu_name_when_available():
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3060"
    ):
        assert get_device_info() == {"available": True, "name": "NVIDIA GeForce RTX 3060"}


def test_get_device_info_reports_unavailable_with_no_name():
    with patch("torch.cuda.is_available", return_value=False):
        assert get_device_info() == {"available": False, "name": None}
