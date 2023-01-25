from itertools import product

import pytest
import numpy as np

from codes.poly_kpm import SumOfOperatorProducts


def random_term(n, m, length, start, end, rng=None):
    """Generate a random term.

    Parameters
    ----------
    n : int
        Size of "A" space
    m : int
        Size of "B" space
    length : int
        Number of operators in the term
    start, end : str
        Start and end spaces of the term (A or B)
    rng : np.random.Generator
        Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()
    spaces = "".join(np.random.choice(a=["A", "B"], size=length - 1))
    spaces = start + spaces + end
    op_spaces = ["".join(s) for s in zip(spaces[:-1], spaces[1:])]
    op_dims = [
        (n if dim[0] == "A" else m, m if dim[1] == "B" else n) for dim in op_spaces
    ]
    ops = [rng.random(size=dim) for dim in op_dims]
    return SumOfOperatorProducts([[(op, space) for op, space in zip(ops, op_spaces)]])


def test_shape_validation():
    """Test that only terms of compatible shapes are accepted.

    Instead of providing terms manually we rely on SumOfOperatorProducts
    creating new instances of itself on addition and multiplication.
    """
    n, m = 4, 10
    terms = {
        "AA": random_term(n, m, 1, "A", "A"),
        "AB": random_term(n, m, 1, "A", "B"),
        "BA": random_term(n, m, 1, "B", "A"),
        "BB": random_term(n, m, 1, "B", "B"),
    }
    for (space1, term1), (space2, term2) in product(terms.items(), repeat=2):
        # Sums should work if the spaces are the same
        if space1 == space2:
            # no error, moreover the result should simplify to a single term
            term1 + term2
            assert len(term1.terms) == 1
        else:
            with pytest.raises(ValueError):
                term1 + term2

        # Matmuls should work if start space of term2 matches end space of term1
        if space1[1] == space2[0]:
            term1 @ term2
        else:
            with pytest.raises(ValueError):
                term1 @ term2


def test_neg():
    """Test that negation works."""
    n, m = 4, 10
    term = random_term(n, m, 1, "A", "A")
    zero = term + -term
    # Should have one term with all zeros
    assert len(zero.terms) == 1
    np.testing.assert_allclose(zero.terms[0][0][0], 0)