from itertools import product
from collections import UserDict
from numbers import Number
from copy import copy
import numpy as np
import scipy.sparse
import sympy
from sympy.core.basic import Basic

from .qsymm.model import Model, allclose, _find_shape, _find_momenta, _mul_shape

# *********************** POLYNOMIAL CLASS ************************************

# Functions to handle different types of arrays
# If either of them are dense, the result is dense.
# LinearOperator only works if it is on the left
def _smart_dot(a, b):
    if isinstance(a, scipy.sparse.linalg.LinearOperator):
        return a.dot(b)
    elif isinstance(a, scipy.sparse.spmatrix) or isinstance(b, scipy.sparse.spmatrix):
        return scipy.sparse.csr_matrix.dot(a, b)
    else:
        return np.dot(a, b)

def _smart_add(a, b):
    if isinstance(a, scipy.sparse.spmatrix) and isinstance(b, scipy.sparse.spmatrix):
        return a + b
    elif isinstance(a, scipy.sparse.spmatrix) and isinstance(b, np.ndarray):
        return a.A + b
    elif isinstance(b, scipy.sparse.spmatrix) and isinstance(a, np.ndarray):
        return a + b.A
    else:
        return a + b


class PerturbativeModel(Model):

    def __init__(self, hamiltonian={}, locals=None, interesting_keys=None, momenta=[]):
        """General class to store Hamiltonian families.
        Can be used to efficiently store any matrix valued function.
        Implements many sympy and numpy methods. Arithmetic operators are overloaded,
        such that `*` corresponds to matrix multiplication.
        Enhances the functionality of Model by allowing interesting_keys to be
        specified, terms with different coefficients are discarded.
        Allows the use of scipy sparse matrices besides numpy arrays.

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

    # Defaultdict functionality
    def __missing__(self, key):
        if self.shape is not None:
            if self.issparse():
                return scipy.sparse.csr_matrix(self.shape, dtype=complex)
            else:
                return np.zeros(self.shape, dtype=complex)
        else:
            return None

    def tosympy(self, digits=12):
        """Convert into sympy matrix."""
        result = self.todense()
        result = [(key * np.round(val, digits)) for key, val in result.items()]
        result = sympy.Matrix(sum(result))
        result.simplify()
        return result

    def tosparse(self):
        output = self.zeros_like()
        output.data = {key: scipy.sparse.csr_matrix(val, dtype=complex)
                       for key, val in self.items()}
        return output

    def issparse(self):
        for key, val in self.items():
            if isinstance(val, scipy.sparse.spmatrix):
                return True
        return False

    def todense(self):
        output = self.zeros_like()
        output.data = {key : (val.A if isinstance(val, scipy.sparse.spmatrix) else val)
                        for key, val in self.items()}
        return output

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
        a = self.todense()
        b = other.todense()
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
                result[key] = _smart_add(self[key], other[key])
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
            result.data = {key * other: val for key, val in self.items()}
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.zeros_like()
            result.data = {key: _smart_dot(val, other) for key, val in self.items()}
            result.shape = _mul_shape(self.shape, other.shape)
        elif isinstance(other, PerturbativeModel):
            interesting_keys = self.interesting_keys | other.interesting_keys
            result = sum(PerturbativeModel({k1 * k2: _smart_dot(v1, v2)},
                                           interesting_keys=interesting_keys)
                      for (k1, v1), (k2, v2) in product(self.items(), other.items())
                      if (k1 * k2 in interesting_keys or not interesting_keys))
            result.momenta = list(set(self.momenta) | set(other.momenta))
            # need to set in case one of them is empty
            result.shape = _mul_shape(self.shape, other.shape)
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            result = self.zeros_like()
            result.data = {other * key: val for key, val in self.items()}
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.zeros_like()
            result.data = {key: _smart_dot(other, val) for key, val in self.items()}
            result.shape = _mul_shape(other.shape, self.shape)
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def around(self, decimals=3):
        raise NotImplementedError()

    def zeros_like(self):
        """Return an empty model object that inherits the other properties"""
        result = PerturbativeModel()
        result.interesting_keys = self.interesting_keys.copy()
        result.momenta = self.momenta.copy()
        result.shape = self.shape
        return result
