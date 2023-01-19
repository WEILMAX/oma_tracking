#!/usr/bin/env python

"""Tests for `oma_tracking` package."""

import pytest


from oma_tracking import oma_tracking
from oma_tracking.oma_tracking import ModeTracking
import pandas as pd
import pytest


@pytest.fixture
def clustered_modes():
    data = {
        "mean_frequency": [1, 2, 3],
        "labels": ["A", "B", "C"],
        "max_distance": [0.5, 0.3, 0.2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def oma_modes():
    data = {"frequency": [1.1, 2.1, 3.1, 4.1]}
    return pd.DataFrame(data)


def test_classify_indices(clustered_modes, oma_modes):
    mode_tracking = ModeTracking(clustered_modes, oma_modes)
    oma_modes_labeled = mode_tracking.classify_indices()

    # Test that the correct label is assigned to the first OMA mode
    assert oma_modes_labeled.loc[0, "labels"] == "A"
    # Test that the correct label is assigned to the second OMA mode
    assert oma_modes_labeled.loc[1, "labels"] == "B"
    # Test that the correct label is assigned to the third OMA mode
    assert oma_modes_labeled.loc[2, "labels"] == "C"
    # Test that the correct label is assigned to the fourth OMA mode
    assert oma_modes_labeled.loc[3, "labels"] == "undefined"


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
