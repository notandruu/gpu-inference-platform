import io
import numpy as np
import pytest
from PIL import Image

from app.preprocessing import preprocess_image, preprocess_batch


def _make_image(width=224, height=224) -> bytes:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_shape():
    img_bytes = _make_image()
    tensor = preprocess_image(img_bytes)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_image_dtype():
    img_bytes = _make_image()
    tensor = preprocess_image(img_bytes)
    assert tensor.dtype == np.float32


def test_preprocess_image_normalized():
    # Check values are not in [0,255] range after normalization
    img_bytes = _make_image()
    tensor = preprocess_image(img_bytes)
    assert tensor.min() < 0 or tensor.max() < 1.0


def test_preprocess_batch_shape():
    images = [_make_image() for _ in range(4)]
    batch = preprocess_batch(images)
    assert batch.shape == (4, 3, 224, 224)


def test_preprocess_batch_single():
    images = [_make_image()]
    batch = preprocess_batch(images)
    assert batch.shape == (1, 3, 224, 224)
