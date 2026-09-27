import numpy as np
import pytest

from dcp.synthetic import make_scene


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_scene():
    """Haze-free scene satisfying the prior, hazed with KNOWN t and A."""
    return make_scene(seed=0)
