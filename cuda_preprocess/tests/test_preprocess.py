"""Tests for CUDA preprocessing extension (skip gracefully without GPU)."""
import numpy as np
import pytest

try:
    import cuda_preprocess as _ext
    HAS_EXT = True
except ImportError:
    HAS_EXT = False


pytestmark = pytest.mark.skipif(not HAS_EXT, reason="cuda_preprocess extension not built")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _reference_preprocess(images: np.ndarray) -> np.ndarray:
    """NumPy reference: NHWC uint8 → NCHW float32 (ImageNet normalized)."""
    arr = images.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD            # broadcast over HW
    return arr.transpose(0, 3, 1, 2)      # NHWC → NCHW


def make_random_images(N=2, H=224, W=224, C=3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(N, H, W, C), dtype=np.uint8)


def test_output_shape():
    images = make_random_images(N=4)
    out = _ext.preprocess(images)
    assert out.shape == (4, 3, 224, 224)


def test_output_dtype():
    images = make_random_images(N=1)
    out = _ext.preprocess(images)
    assert out.dtype == np.float32


def test_numerical_correctness():
    images = make_random_images(N=2)
    cuda_out = _ext.preprocess(images)
    ref_out  = _reference_preprocess(images)
    max_diff = np.abs(cuda_out - ref_out).max()
    assert max_diff < 1e-4, f"Max difference too large: {max_diff}"


def test_single_image():
    images = make_random_images(N=1)
    out = _ext.preprocess(images)
    assert out.shape == (1, 3, 224, 224)


def test_invalid_channels():
    bad = np.zeros((2, 224, 224, 1), dtype=np.uint8)
    with pytest.raises(Exception):
        _ext.preprocess(bad)


def test_invalid_dims():
    bad = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        _ext.preprocess(bad)
