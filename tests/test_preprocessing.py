import numpy as np
import cv2
from scripts.main import preprocess_image


def test_preprocess_image_shape_and_type():
    dummy_img = np.ones((300, 600), dtype=np.uint8) * 255
    path = "tests/sample_dummy.png"
    cv2.imwrite(path, dummy_img)

    img = preprocess_image(path)
    assert img is not None
    assert img.shape == (128, 256)
    assert img.dtype == np.uint8
