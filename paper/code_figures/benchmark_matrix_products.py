# %%
from collections import Counter
import sympy

from IPython.display import display

from pymablock.block_diagonalization import block_diagonalize
from pymablock.series import BlockSeries, AlgebraElement, zero

# %%
solves = []


def solve_sylvester(A):
    solves.append(A)
    return AlgebraElement(f"G_{len(solves)}")


def eval_dense_first_order(*index):
    if index[2] != 1:
        return zero
    return AlgebraElement(f"H_{[['AA', 'AB'], ['BA', 'BB']][index[0]][index[1]]}")


# %%
H = BlockSeries(
    eval=eval_dense_first_order,
    shape=(2, 2),
    n_infinite=1,
)

order = 3
AlgebraElement.log = []
solves = []
H_tilde, *_ = block_diagonalize(
    H,
    solve_sylvester=solve_sylvester,
)
H_tilde[0, 0, order]

# %%
for series in H_tilde.details.values():
    print(series.name)
    for key, value in series._data.items():
        if not isinstance(value, AlgebraElement):
            continue
        print(key)
        display(value.to_sympy())
    print()

# %%
for product in AlgebraElement.log:
    if product[1] != "__mul__":
        continue
    display(sympy.Matrix([[product[0].to_sympy(), product[2][0].to_sympy()]]))
# %%
