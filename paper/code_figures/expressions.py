# %%
import sympy
import pymablock
from pymablock.series import AlgebraElement
from IPython.display import Latex

H = pymablock.series.BlockSeries(
    data={
        (0, 0, 0): AlgebraElement("H_0"),
        (1, 1, 0): AlgebraElement("H_1"),
        (0, 0, 1): AlgebraElement("H_2"),
        (1, 1, 1): AlgebraElement("H_3"),
        (0, 1, 1): AlgebraElement("H_4"),
        (1, 0, 1): AlgebraElement("H_5"),
    },
    shape=(2, 2),
    n_infinite=1,
)

Ys = {}
def solve_sylvester(Y, index):
    Ys[index[2]] = Y
    return AlgebraElement(f"Y_{index[2]}")


H_tilde, *_ = pymablock.block_diagonalize(H, solve_sylvester=solve_sylvester)
# %%
results = []
for order in range(1, 5):
    H_tilde[0, 0, order]

for order in range(1, 4):
    results.append(sympy.Eq(sympy.Symbol(f"Y_{order}"), Ys[order].to_sympy()))
    results.append(sympy.Eq(sympy.Symbol(f"\\tilde{{H}}_{order+1}"), H_tilde[0, 0, order+1].to_sympy()))

result = "\\\\\n".join(sympy.latex(i) for i in results)
result = r"\begin{gather}" + result + r"\end{gather}"
for original, replaced in (
    ("H_{0}", "H_{0,AA}"),
    ("H_{1}", "H_{0,BB}"),
    ("H_{2}", "H_{1,AA}"),
    ("H_{3}", "H_{1,BB}"),
    ("H_{4}", "H_{1,AB}"),
    ("H_{5}", "H_{1,BA}"),
):
    result = result.replace(original, replaced)
# %%
display(Latex(result))
# %%
