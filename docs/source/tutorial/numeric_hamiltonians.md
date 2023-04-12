---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Numeric

Here we demonstrate how to use Lowdin on numerical inputs.
We start with the necessary imports.

```{code-cell} ipython3
import numpy as np
import lowdin
```

We initialize a random Hermitian matrix of dimensions and a random perturbation
```{code-cell} ipython3
# Define a random Hamiltonian
N = 4
H_0 = np.random.random((N, N)) + 1j * np.random.random((N, N))
H_0 += H_0.conj().T

# Define a random.random perturbation
H_p = np.random.random((N, N)) + 1j * np.random.random((N, N))
H_p += H_p.conj().T
H_p = 0.01 * H_p
```

For simplicity, let's bring the unperturbed Hamiltonian to its eigenbasis
```{code-cell} ipython3
eigvals, eigvecs = np.linalg.eigh(H_0)
H_0_AA = np.diag(eigvals[: N // 2]) # occupied subspace
H_0_BB = np.diag(eigvals[N // 2:]) # unoccupied subspace
```
where we see the eigenvalues of the occupied subspace
```{code-cell} ipython3
H_0_AA
```

We bring the perturbation to the same occupied/unoccupied basis.
```{code-cell} ipython3
H_p = eigvecs.conj().T @ H_p @ eigvecs

H_p_AA = {(1, ): H_p[: N//2, : N // 2]}
H_p_BB = {(1, ): H_p[N//2:, N // 2:]}
H_p_AB = {(1, ): H_p[: N//2, N // 2:]}
```

Now we can write the total Hamiltonian as a Block Series,
```{code-cell} ipython3
H = lowdin.to_BlockSeries(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB)
```

We can access the occupied block of the {math}`0`-th order Hamiltonian as
```{code-cell} ipython3
H.evaluated[0, 0, 0]
```
which, as expected, corresponds to `H_0_AA`.
Similarly, the occupied block of the {math}`1`-st order Hamiltonian is

```{code-cell} ipython3
H.evaluated[0, 0, 1]
```
which corresponds to the original perturbation.

We can now set up Lowdin by defining

```{code-cell} ipython3
H_tilde, U, U_adjoint = lowdin.general(H)
```
and obtain a second order correction to the occupied subspace as

```{code-cell} ipython3
H_tilde.evaluated[0, 0, 2]
```

