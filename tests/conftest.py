import numpy as np
import pytest


@pytest.fixture
def np_rng() -> np.random.Generator:
    return np.random.default_rng(1234)
