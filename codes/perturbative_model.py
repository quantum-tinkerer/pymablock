from itertools import product
from collections import UserDict
from numbers import Number
import numpy as np
import scipy.sparse
import sympy
from sympy.core.basic import Basic
from .qsymm.model import Model, allclose, _find_shape, _find_momenta

# *********************** POLYNOMIAL CLASS ************************************

# Functions to handle different types of arrays
# If either of them are dense, the result is dense.
def _smart_dot(a, b):
    if isinstance(a, scipy.sparse.spmatrix) or isinstance(b, scipy.sparse.spmatrix):
        return scipy.sparse.csr_matrix.dot(a, b)
    else:
        return np.dot(a, b)

def _smart_add(a, b):
    if isinstance(a, scipy.sparse.spmatrix) and isinstance(b, scipy.sparse.spmatrix):
        return a + b
    elif isinstance(a, scipy.sparse.spmatrix):
        return (a + b).A
    elif isinstance(b, scipy.sparse.spmatrix):
        return (b + a).A
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
            {sympy expression: array} with all arrays either dense or sparse, with
            the same size and sympy expressions consisting purely of symbolic
            coefficients, no constant factors.
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
        # Usual case is initializing with a dict
        if isinstance(hamiltonian, dict):
            UserDict.__init__(self, hamiltonian)
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
        # Removing all key that are not interesting.
        if self.interesting_keys:
            for key in self.keys():
                if key not in self.interesting_keys:
                    del self[key]

    def tosympy(self, digits=12):
        """Convert into sympy matrix."""
        result = self.todense()
        result = [(key * np.round(val, digits)) for key, val in result.items()]
        result = sympy.Matrix(sum(result))
        result.simplify()
        return result

    def tosparse(self):
        output = self.copy()
        for key, val in output.items():
            output[key] = scipy.sparse.csr_matrix(val, dtype=complex)
        return output

    def issparse(self):
        for key, val in self.items():
            if isinstance(val, scipy.sparse.spmatrix):
                return True
        return False

    def todense(self):
        output = self.copy()
        for key, val in output.items():
            if isinstance(val, scipy.sparse.spmatrix):
                output[key] = val.A
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
        if not other:
            result = self.copy()
        # If self is empty return other
        elif not self and isinstance(other, type(self)):
            result = other.copy()
        elif isinstance(other, type(self)):
            result = self.copy()
            for key, val in other.items():
                result[key] = _smart_add(result[key], val)
        else:
            raise NotImplementedError('Addition of {} with {} not supported'.format(type(self), type(other)))
        return result

    def __mul__(self, other):
        # Multiplication by numbers, sympy symbols, arrays and Model
        if isinstance(other, Number):
            result = self.copy()
            for key, val in result.items():
                result[key] *= other
        elif isinstance(other, Basic):
            result = PerturbativeModel({key * other: val for key, val in self.items()},
                                        interesting_keys=interesting_keys)
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = _smart_dot(val, other)
            result.shape = (self.shape[0], other.shape[1])
        elif isinstance(other, PerturbativeModel):
            interesting_keys = self.interesting_keys | other.interesting_keys
            result = sum([PerturbativeModel({k1 * k2: _smart_dot(v1, v2)}, interesting_keys=interesting_keys)
                      for (k1, v1), (k2, v2) in product(self.items(), other.items())
                      if (k1 * k2 in interesting_keys or not interesting_keys)])
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def __truediv__(self, B):
        result = self.copy()

        if isinstance(B, PerturbativeModel):
            raise TypeError(
                "unsupported operand type(s) for /: 'PerturbativeModel' and "
                "'PerturbativeModel'"
            )
        else:
            for key in result:
                result[key] /= B

        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            result = PerturbativeModel({other * key: val for key, val in self.items()},
                                        interesting_keys=interesting_keys)
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = _smart_dot(other, val)
            result.shape = (other.shape[0], self.shape[1])
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def around(self, decimals=3):
        raise NotImplementedError()
