import io
import numpy as np
from PIL import Image

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes → float32 NCHW tensor (1, 3, 224, 224)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
    arr = (arr - _MEAN) / _STD  # normalize
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis]  # → (1, 3, 224, 224)


def preprocess_batch(images_bytes: list[bytes]) -> np.ndarray:
    """Decode a list of image bytes → float32 NCHW tensor (N, 3, 224, 224)."""
    tensors = [preprocess_image(b)[0] for b in images_bytes]
    return np.stack(tensors, axis=0)
