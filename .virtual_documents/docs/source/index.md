
import sympy

from lowdin.block_diagonalization import BlockSeries, symbolic

H = BlockSeries(
    data={
        (0, 0, 0): [1],
        (1, 1, 0): [3],
        (0, 0, 1): [2],
        (0, 1, 1): [3],
        (1, 1, 1): [4],
    },
    shape=(2, 2),
    n_infinite=1,
)

H_p_AA = sympy.Symbol('{H_{p}^{AA}}')
H_p_BB = sympy.Symbol('{H_{p}^{BB}}')
H_p_AB = sympy.Symbol('{H_{p}^{AB}}')

H_p_AA_rep = sympy.Symbol('H_{(0, 0, 1)}')
H_p_BB_rep = sympy.Symbol('H_{(1, 1, 1)}')
H_p_AB_rep = sympy.Symbol('H_{(0, 1, 1)}')

hamiltonians = {H_p_AA_rep: H_p_AA, H_p_BB_rep: H_p_BB, H_p_AB_rep: H_p_AB}
offdiagonals = {sympy.Symbol(f'V_{{({i},)}}'): sympy.Symbol(f'V_{i}') for i in range(5)}

H_tilde, U, U_adjoint, Y, subs = symbolic(H)

H_tilde[0, 0, 4].subs({**hamiltonians, **offdiagonals})
