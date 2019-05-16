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
        interesting_keys : iterable of sympy expressions
            Set of symbolic coefficients that are kept, anything that does not
            appear here is discarded. Useful for perturbative calculations where
            only terms to a given order are needed.
        momenta : list of int or list of Sympy objects
            Indices of momentum variables from ['k_x', 'k_y', 'k_z']
            or a list of names for the momentum variables as sympy symbols.
            Momenta are treated the same as other keys for the purpose of
            `interesting_keys`, need to list interesting powers of momenta.
        """
        # Usual case is initializing with a dict,
        # want to bypass cleanup mechanism in Model
        if isinstance(hamiltonian, dict):
            UserDict.__init__(self, {sympy.sympify(k): v for k, v in hamiltonian.items()})
            self.shape = _find_shape(hamiltonian)
            self.momenta = _find_momenta(momenta)
        # Otherwise try to parse the input with Model's machinery.
        # This will always result in a dense PerturbativeModel.
        else:
            super().__init__(hamiltonian, locals, momenta)

        if interesting_keys is not None:
            self.interesting_keys = set(interesting_keys)
        else:
            self.interesting_keys = set()
        # Removing all keys that are not interesting.
        if self.interesting_keys:
            for key in self.keys():
                if key not in self.interesting_keys:
                    del self[key]
        # Keep track of whether this is a dense array
        self._isarray = any(isinstance(val, np.ndarray) for val in self.values())

    # Defaultdict functionality
    def __missing__(self, key):
        if self.shape is not None:
            if self.shape == ():
                #scalar
                return 0
            elif self._isarray:
                # Return dense zero array if dense
                return np.zeros(self.shape, dtype=complex)
            else:
                # Otherwise return a csr_matrix
                return scipy.sparse.csr_matrix(self.shape, dtype=complex)
        else:
            return None

    def tosympy(self, digits=12):
        """Convert into sympy matrix."""
        result = self.toarray()
        result = [(key * np.round(val, digits)) for key, val in result.items()]
        result = sympy.Matrix(sum(result))
        result.simplify()
        return result

    def tocsr(self):
        result = self.zeros_like()
        result.data = {key: scipy.sparse.csr_matrix(val, dtype=complex)
                       for key, val in self.items()}
        result._isarray = False
        return result

    def toarray(self):
        result = self.zeros_like()
        result.data = {key : (val if isinstance(val, np.ndarray) else val.toarray())
                        for key, val in self.items()}
        result._isarray = True
        return result

    def lambdify(self, *gens):
        """Lambdify using gens as variables."""
        result = self.tosympy()
        result = sympy.lambdify(gens, result, 'numpy')
        return result

    def evalf(self, subs=None):
        result = []
        for key, val in self.items():
            key = float(key.evalf(subs=subs))
            result.append(key * val)
        return sum(result)

    def __eq__(self, other):
        a = self.toarray()
        b = other.toarray()
        for key in a.keys() | b.keys():
            if not allclose(a[key], b[key]):
                return False
        return True

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

    def __mul__(self, other):
        # Multiplication by numbers, sympy symbols, arrays and Model
        result = self.zeros_like()
        if isinstance(other, Number):
            result.data = {key: val * other for key, val in self.items()}
        elif isinstance(other, Basic):
            result.data = {key * other: val for key, val in self.items()
                           if (key * other in interesting_keys or not interesting_keys)}
        elif isinstance(other, PerturbativeModel):
            interesting_keys = self.interesting_keys | other.interesting_keys
            result = sum(PerturbativeModel({k1 * k2: v1 * v2},
                                           interesting_keys=interesting_keys)
                      for (k1, v1), (k2, v2) in product(self.items(), other.items())
                      if (k1 * k2 in interesting_keys or not interesting_keys))
            result.momenta = list(set(self.momenta) | set(other.momenta))
            # Find out the shape of the result even if it is empty
            result.shape = _find_shape(result.data) if result.data else (self[1] * other[1]).shape
        else:
            # Otherwise try to multiply every value with other
            result.data = {key: val * other for key, val in self.items()}
            result.shape = _find_shape(result.data) if result.data else (self[1] * other).shape
        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            result = self.zeros_like()
            result.data = {other * key: val for key, val in self.items()}
        else:
            # Otherwise try to multiply every value with other
            result = self.zeros_like()
            result.data = {key: other * val for key, val in self.items()}
            result.shape = _find_shape(result.data) if result.data else (other * self[1]).shape
        return result

    def __matmul__(self, other):
        # Multiplication by arrays and PerturbativeModel
        if isinstance(other, PerturbativeModel):
            interesting_keys = self.interesting_keys | other.interesting_keys
            result = sum(PerturbativeModel({k1 * k2: v1 @ v2},
                                           interesting_keys=interesting_keys)
                      for (k1, v1), (k2, v2) in product(self.items(), other.items())
                      if (k1 * k2 in interesting_keys or not interesting_keys))
            result.momenta = list(set(self.momenta) | set(other.momenta))
            result.shape = _find_shape(result.data) if result.data else (self[1] @ other[1]).shape
        else:
            # Otherwise try to multiply every value with other
            result = self.zeros_like()
            result.data = {key: val @ other for key, val in self.items()}
            result.shape = _find_shape(result.data) if result.data else (self[1] @ other).shape
        return result

    def __rmatmul__(self, other):
        # Left multiplication by arrays
        result = self.zeros_like()
        result.data = {key: other @ val for key, val in self.items()}
        result.shape = _find_shape(result.data) if result.data else (other @ self[1]).shape
        return result

    def around(self, decimals=3):
        raise NotImplementedError()

    def zeros_like(self):
        """Return an empty model object that inherits the other properties"""
        result = PerturbativeModel()
        result.interesting_keys = self.interesting_keys.copy()
        result.momenta = self.momenta.copy()
        result.shape = self.shape
        result._isarray = self._isarray
        return result
