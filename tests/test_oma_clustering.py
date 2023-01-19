"""
Test of the oma_clustering module.
"""
from dataclasses import asdict
import pytest
import pandas as pd
from sklearn.datasets import make_blobs
from oma_tracking.oma_clustering import ModeClusterer


def test_fit():
    """Test of the fit method of the ModeClusterer class."""
    data, y = make_blobs(n_samples=100, n_features=3, centers=3, random_state=0)
    data = pd.DataFrame(data, columns=["frequency", "size", "damping"])
    modeclusterer = ModeClusterer()
    modeclusterer.fit(data)

    data_size = data[
        (data["size"] > modeclusterer.min_modal_size)
        & (data["damping"] < modeclusterer.max_modal_damping)
    ]
    assert modeclusterer.dbsc is not None
    assert modeclusterer.dbscan_data is not None
    assert modeclusterer.dbscan_data.shape[0] == data_size.shape[0]


def test_fit_raise_error():
    """Test of the error raised during the fit method of the ModeClusterer class."""
    data, y = make_blobs(n_samples=100, centers=3, random_state=0)
    data = pd.DataFrame(data, columns=["frequency", "size"])
    modeclusterer = ModeClusterer()
    with pytest.raises(ValueError):
        modeclusterer.fit(data)


def test_post_init():
    """Test of the post_init method of the ModeClusterer class."""
    modeclusterer = ModeClusterer()
    assert modeclusterer.dbsc is not None
    assert modeclusterer.dbscan_data is not None


def test_dataclass_attributes():
    """Test of the dataclass attributes of the ModeClusterer class."""
    modeclusterer = ModeClusterer()
    assert asdict(modeclusterer) == asdict(ModeClusterer())
