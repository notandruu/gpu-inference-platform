"""
cuda_preprocess — CUDA-accelerated image preprocessing extension.

Build with:
    cd cuda_preprocess && python setup.py build_ext --inplace

Or from repo root:
    make build-cuda

Falls back gracefully if the compiled extension is unavailable (no GPU / not built).
"""
try:
    from cuda_preprocess import preprocess as cuda_preprocess  # compiled extension
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cuda_preprocess = None  # type: ignore

__all__ = ["cuda_preprocess", "HAS_CUDA"]
