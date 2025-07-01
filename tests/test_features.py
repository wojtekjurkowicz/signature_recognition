import numpy as np
from scripts.main import extract_features


def test_extract_features_output():
    dummy = np.zeros((128, 256), dtype=np.uint8)
    feats = extract_features(dummy)
    assert isinstance(feats, list)
    assert len(feats) == 25
    assert all(np.isfinite(feats))
