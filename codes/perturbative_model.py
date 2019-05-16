from itertools import product
from collections import UserDict
from numbers import Number
from copy import copy
import numpy as np
import scipy.sparse
import sympy
from sympy.core.basic import Basic

from .qsymm.model import Model, allclose, _find_shape, _find_momenta


class PerturbativeModel(Model):

    def __init__(self, hamiltonian={}, locals=None, interesting_keys=None, momenta=[]):
        """
        General class to efficiently store any matrix valued function.
        The internal structure is a dict with {symbol: value}, where
        symbol is a sympy expression, the object representing sum(symbol * value).
        The values can be scalars, arrays (both dense and sparse) or LinearOperators.
        Implements many sympy and numpy methods and arithmetic operators.
        Multiplication is distributed over the sum, `*` is passed down to
        both symbols and values, `@` is passed to symbols as `*` and to values
        as `@`. Assumes that symbols form a commutative group.
        Enhances the functionality of Model by allowing `interesting_keys` to be
        specified, symbols not listed there are discarded.

        Parameters
        ----------
        hamiltonian : str, SymPy expression or dict
            Symbolic representation of a Hamiltonian.  It is
            converted to a SymPy expression using `kwant_continuum.sympify`.
            If a dict is provided, it should have the form
            `{expression: array}` with all arrays the same size (dense or sparse).
            `expression` will be passed through sympy.sympify, and should consist
            purely of symbolic coefficients, no constant factors other than 1.
        locals : dict or ``None`` (default)
            Additional namespace entries for `~kwant_continuum.sympify`.  May be
            used to simplify input of matrices or modify input before proceeding
            further. For example:
            ``locals={'k': 'k_x + I * k_y'}`` or
            ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
        interesting_keys : iterable of sympy expressions or None (default)
            Set of symbolic coefficients that are kept, anything that does not
            appear here is discarded. Useful for perturbative calculations where
            only terms to a given order are needed. By default all keys are kept.
        momenta : list of int or list of Sympy objects
            Indices of momentum variables from ['k_x', 'k_y', 'k_z']
            or a list of names for the momentum variables as sympy symbols.
            Momenta are treated the same as other keys for the purpose of
            `interesting_keys`, need to list interesting powers of momenta.
        """
        # Usual case is initializing with a dict,
        # want to bypass cleanup mechanism in Model
        if isinstance(hamiltonian, dict):
            UserDict.__init__(self, {sympy.sympify(k): v for k, v in hamiltonian.items()
                              if not interesting_keys or sympy.sympify(k) in interesting_keys})
            self.shape = _find_shape(hamiltonian)
            self.momenta = _find_momenta(momenta)
        # Otherwise try to parse the input with Model's machinery.
        # This will always result in a dense PerturbativeModel.
        else:
            super().__init__(hamiltonian, locals, momenta=momenta,
                             interesting_keys=interesting_keys)

        if interesting_keys is not None:
            self.interesting_keys = set(interesting_keys)
        else:
            self.interesting_keys = set()

        # Keep track of whether this is a dense array
        self._isarray = any(isinstance(val, np.ndarray) for val in self.values())

    # This only differs in not deleting small values
    def __add__(self, other):
        # Addition of Models. It is assumed that both Models are
        # structured correctly, every key is in standard form.
        # Define addition of 0 and {}
        result = self.zeros_like()
        if not other:
            result.data = self.data.copy()
        # If self is empty return other
        elif not self and isinstance(other, type(self)):
            result = other.zeros_like()
            result.data = other.data.copy()
        elif isinstance(other, type(self)):
            for key in self.keys() & other.keys():
                total = self[key] + other[key]
                # If only one is sparse matrix, the result is np.matrix, recast it to np.ndarray
                if isinstance(total, np.matrix):
                    total = total.A
                result[key] = total
            for key in self.keys() - other.keys():
                result[key] = copy(self[key])
            for key in other.keys() - self.keys():
                result[key] = copy(other[key])
        else:
            raise NotImplementedError('Addition of {} with {} not supported'.format(type(self), type(other)))
        return result
