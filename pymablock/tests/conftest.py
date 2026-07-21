"""Shared test fixtures."""

import hashlib

import numpy as np
import pytest


def _rng_for_node(request: pytest.FixtureRequest) -> np.random.Generator:
    """Return a modern NumPy generator controlled by pytest-randomly."""
    randomly_seed = request.config.getoption("randomly_seed")
    node_digest = hashlib.sha256(request.node.nodeid.encode()).digest()
    node_seed = int.from_bytes(node_digest[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([randomly_seed, node_seed]))


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Return a reproducible generator unique to the current test."""
    return _rng_for_node(request)


@pytest.fixture(scope="module")
def module_rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Return a reproducible generator shared by module-scoped fixtures."""
    return _rng_for_node(request)
