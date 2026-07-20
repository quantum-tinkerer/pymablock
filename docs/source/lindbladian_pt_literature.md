# Lindbladian perturbation theory in the literature

This page maps the main perturbative approaches to Lindblad generators,
identifies their applications, and compares them with Pymablock. It is a
working review rather than an exhaustive bibliography. We include a paper
when it introduces a distinct reduction principle, establishes a structural
guarantee, or supplies a useful benchmark.

## Main approaches

| Approach | Object being expanded | Main guarantee | Typical applications |
| --- | --- | --- | --- |
| [Effective operators](https://arxiv.org/abs/1112.2806) | A decaying excited Hilbert-space sector is eliminated through a non-Hermitian Hamiltonian. | The second-order result is given directly as an effective Hamiltonian and physical jump operators. | Raman transitions, optical pumping, engineered decay, dissipative state preparation. |
| [Dissipative Schrieffer--Wolff transformation](https://arxiv.org/abs/1205.5440) | A nonunitary similarity transformation block diagonalizes the Liouvillian. | Arbitrary perturbative order and agreement with the low-lying Liouvillian spectrum; Lindblad form is proved at second order in the generic ancilla setting. | Elimination of lossy ancillas, cavity-mediated interactions, collective decay and superradiance. |
| [Direct Liouvillian eigenmode perturbation theory](https://www.nature.com/articles/srep04887) | Individual left and right eigenoperators, especially the steady state, are expanded recursively. | Nondegenerate eigenvalue and eigenoperator corrections; an amplitude-matrix variant preserves positivity of the approximate steady state. | Driven spin arrays, coupled cavity--qubit systems and steady-state observables. |
| [Geometric adiabatic elimination](https://arxiv.org/abs/1603.04630) and its [bipartite extension](https://arxiv.org/abs/1704.00785) | The slow invariant manifold and its embedding into the full state space are expanded together. | Trace and complete positivity are built into the second-order reduced generator and embedding. | Reservoir engineering, Zeno dynamics, lossy oscillators and autonomous stabilization. |
| [Metastable spectral reduction](https://arxiv.org/abs/1512.05801) | A cluster of low-lying Liouvillian eigenmodes defines a metastable manifold. | Controlled intermediate-time dynamics with an approximately completely positive effective generator. | Phase coexistence, long-lived classical states, decoherence-free subspaces and noiseless subsystems. |
| [Geometry and response](https://arxiv.org/abs/1512.08079) and [dynamical response theory](https://arxiv.org/abs/1512.07860) | The asymptotic projector, steady states, and observables are differentiated using reduced resolvents. | Linear response and adiabatic transport formulas for steady-state manifolds. | Reservoir-engineered phases, holonomic control, Hall response and sensing near dissipative transitions. |
| [Floquet high-frequency expansion](https://arxiv.org/abs/2107.10054) | A periodic Liouvillian is replaced by a stroboscopic generator and micromotion. | A rotating-frame construction can retain a Floquet Lindbladian at high frequency. | Periodically driven qubits and Floquet reservoir engineering. |
| [Variational perturbation theory](https://arxiv.org/abs/2504.00085) | Perturbative corrections to one steady state form a reusable reduced basis. | A residual-minimizing recombination extends usefulness beyond the Taylor radius; LU reuse and preconditioned Krylov variants avoid an explicit pseudoinverse. | Parameter sweeps, fitting, phase diagrams, and steady-state gradients. |
| [Algebraic center-manifold reduction](https://arxiv.org/abs/2603.11982) | The unperturbed center sector is kept fixed while leakage is treated perturbatively. | A completely positive reduced generator for arbitrary perturbation strength within the construction, together with finite-time leakage bounds. | Dissipative many-body systems with stationary or oscillating long-time sectors. |

These methods share reduced-resolvent algebra but optimize different outputs.
Effective-operator and geometric methods prioritize a physical reduced master
equation. Schrieffer--Wolff methods prioritize spectral decoupling and access
to high orders. Metastability and response theory start from spectral
projectors and focus on time windows or observables rather than an explicit
block-diagonalizing transformation.

### Variational steady-state perturbation theory

Melo, Beugnot, and Minganti address a different computational question from
effective-model perturbation theory. Given

:::{math}
\mathcal L(\varepsilon)=\mathcal L_0+\varepsilon\mathcal L_1,
:::

they compute one stationary right eigenoperator over many parameter values.
Ordinary perturbation theory gives

:::{math}
\mathcal L_0\rho^{(n)}=-\mathcal L_1\rho^{(n-1)},
\qquad
\rho^{(n)}=-\mathcal L_0^+\mathcal L_1\rho^{(n-1)}.
:::

These correction vectors form a Krylov sequence. Instead of retaining the
Taylor weights $\varepsilon^n$, variational perturbation theory orthonormalizes
their span into $Q$ and solves the small projected problem

:::{math}
Q^\dagger\widetilde{\mathcal L}(\varepsilon)Qq
=Q^\dagger b,
\qquad
\rho_{\mathrm{VPT}}=Qq/\operatorname{Tr}(Qq).
:::

The full residual is then checked explicitly. Multipoint VPT pools correction
vectors from several expansion points, allowing the reduced space to cross a
region where no single Taylor series converges.

The overlap with Pymablock is precise but limited. For a unique stationary
state, the vectors $\rho^{(n)}$ describe one column of the perturbative
embedding returned by block diagonalization. Both methods repeatedly solve
with the same singular $\mathcal L_0$. Melo and collaborators replace it by a
rank-one trace-fixed operator and reuse one LU factorization; this is directly
relevant to a specialized stationary-state solver in Pymablock.

The differences matter:

- VPT computes one zero mode, whereas Pymablock transforms an entire spectral
  block and produces its effective generator and embedding.
- VPT's Moore--Penrose convention makes each correction orthogonal to
  $\rho^{(0)}$. This is a normalization gauge, not the partial-trace gauge or
  Pymablock's block-diagonalization gauge.
- Variational recombination is a reduced-basis resummation performed after the
  perturbative vectors are known. Its coefficients are not Taylor
  coefficients and do not define a perturbative effective Liouvillian.
- The rank-one trace fix assumes a one-dimensional stationary kernel. A
  degenerate steady manifold requires a higher-rank constraint and returns to
  the block problem handled by Pymablock.
- Normalization and a small Liouvillian residual do not by themselves ensure
  positivity of the approximate density matrix. The residual must also be
  interpreted relative to the Liouvillian gap, as the paper emphasizes.

The useful integration path is therefore not to merge the algorithms. Let
Pymablock generate stationary-mode correction vectors using its existing
Sylvester backend, then optionally expose their orthonormal span for VPT-style
single- or multipoint recombination. Conversely, the rank-one trace-fixed LU
solve is a practical backend for the unique-steady-state case.

We checked the relation explicitly for a detuned, driven qubit with amplitude
damping. Let $K(\varepsilon)$ be the stationary column of Pymablock's returned
embedding. Its scalar gauge is converted to the Moore--Penrose convention by

:::{math}
\rho_{mathrm{MP}}(\varepsilon)
=\frac{K(\varepsilon)}
{\langle\rho^{(0)},K(\varepsilon)\rangle},
:::

where division denotes formal power-series division. Through fourth order,
these coefficients agree with the recurrence
$\rho^{(n)}=-\mathcal L_0^+\mathcal L_1\rho^{(n-1)}$ to
$2.4\times10^{-16}$. Replacing the pseudoinverse with the paper's rank-one
trace-fixed LU solve gives the same coefficients to $7.9\times10^{-17}$.
Thus standard steady-state PT in this case is exactly one gauge-fixed column
of Pymablock's embedding. VPT begins only when the Taylor weights are replaced
by residual-minimizing coefficients in the span of those columns.

## Applications

The literature uses these reductions most often in four concrete settings.

**Quantum optics and atomic physics.** Eliminating short-lived excited states
produces ac Stark shifts, Raman couplings, optical pumping rates, and engineered
decay channels. The effective-operator formalism packages these processes into
one inverse non-Hermitian Hamiltonian and was developed partly to design
[dissipative state preparation](https://arxiv.org/abs/1112.2806).

**Lossy mediators and reservoir engineering.** A damped cavity, oscillator, or
ancilla mediates coherent interactions and collective jumps between slow
degrees of freedom. Kessler's third-order superradiance calculation shows that
odd orders can materially improve the dynamics even when their Lindblad form
is not manifest. Geometric elimination applies the same idea to strongly
damped oscillators and Zeno manifolds.

**Protected quantum information.** Slow-manifold methods describe gates and
errors inside decoherence-free or metastable subspaces. Recent numerical
adiabatic elimination targets
[autonomous error correction with cat qubits](https://arxiv.org/abs/2303.05089),
where tensor-product simulation quickly becomes prohibitive but local
second-order reductions remain tractable.

**Many-body and driven systems.** Low Liouvillian modes diagnose metastability,
dissipative phase coexistence, and long-lived coherent sectors. Response
theory uses the same reduced resolvents for susceptibilities near dissipative
transitions, while Floquet expansions seek effective generators for periodic
reservoir engineering. These applications need scalable Sylvester solves more
than explicit dense diagonalization, making them the strongest motivation for
Pymablock's implicit and sparse backends.

## Structural limits

Complete positivity is the main dividing line. Kessler proves Lindblad form
for the generic second-order ancilla elimination, while the geometric methods
construct both a Lindbladian slow generator and a completely positive
embedding through second order. Higher order is different. Tokieda, Elouard,
Sarlette, and Rouchon give a
[fourth-order counterexample](https://arxiv.org/abs/2211.11008) in which no
choice of slow-coordinate gauge makes the reduced dynamics completely
positive. The obstruction is physical: a reduced Markovian state cannot
encode all fast--slow correlations accumulated in the invariant manifold.

This result changes the goal of high-order perturbation theory. A high-order
effective Liouvillian may accurately reproduce the slow spectrum and physical
trajectories associated with the embedded manifold without itself generating
a completely positive semigroup on every abstract reduced state. Recovering a
positive Kossakowski series is therefore a test, not an automatic final step.

Exceptional points form a second boundary. The methods above normally assume
an isolated diagonalizable slow spectral subspace. At a Liouvillian
exceptional point, eigenoperators coalesce and ordinary Taylor series may be
replaced by Jordan-chain or Puiseux expansions. Pymablock's present
eigenbasis-based mask rules do not cover this regime.

## How Pymablock fits

Pymablock is closest to the dissipative Schrieffer--Wolff construction. Both
compute a similarity transformation that removes slow--fast couplings and
reproduces the corresponding Liouvillian spectrum. Pymablock adds several
capabilities that are not central in the cited formulations:

- multivariate perturbative series;
- arbitrary collections of blocks rather than one steady and one decaying
  sector;
- selective element masks, including masks inherited from a Hamiltonian
  problem;
- sparse and implicit Sylvester solvers intended for larger operator spaces;
- one recurrence shared with general non-Hermitian problems.

The generality has a cost. Pymablock's transformation is an algebraic
similarity, not a completely positive embedding of reduced density matrices.
Trace- and Hermiticity-compatible masks can preserve those two structures,
but complete positivity must be checked separately. The Kossakowski
reconstruction and Gram-feasibility formulation in
[Lindbladian perturbation theory](lindbladian_perturbation_theory.md) provide
such a check at finite perturbative order.

The most promising positioning is therefore: Pymablock supplies the
high-order, multivariate spectral algebra; specialized physical reductions
supply extra constraints or post-processing when a completely positive
reduced-state interpretation is required. A future interface should expose
both the effective block and the embedding transformation, because the
higher-order no-go result shows that neither object alone captures the full
physical approximation.

## Reproduction: effective operators for a Lambda system

We reproduced the second-order effective-operator result of Reiter and
Sorensen for a three-level Lambda system. Two ground states are weakly coupled
to a detuned excited state, which decays into both ground states. Their
formalism gives

:::{math}
H_{\mathrm{eff}}
=-\frac12V_-\left(H_{\mathrm{NH}}^{-1}
+H_{\mathrm{NH}}^{-1\dagger}\right)V_+,
\qquad
J_{k,\mathrm{eff}}=J_kH_{\mathrm{NH}}^{-1}V_+.
:::

We diagonalized the unperturbed Liouvillian, isolated its four-dimensional
ground-operator kernel, and block diagonalized the full Liouvillian with
Pymablock. For non-symmetric detunings, drives, and decay rates, the
second-order Pymablock coefficient agrees with the Liouvillian constructed
from the published effective operators to

:::{math}
\max|\mathcal L_{2,\mathrm{Pymablock}}
-\mathcal L_{2,\mathrm{effective\ operators}}|
=9.8\times10^{-18}.
:::

The four exact low-lying eigenvalues of the full Liouvillian differ from the
second-order effective eigenvalues by $O(g^4)$; a log--log fit over
$g=0.05$--$0.2$ gives exponent $3.9985$. The missing cubic term follows from
the excitation-parity structure of this model rather than from a generic
fourth-order guarantee.

The executable reproduction is in `docs/reproduce_lindbladian_pt.py` and runs
with

```bash
pixi run -e docs docs-reproduce-lindbladian
```

## Reproduction: a metastable shelving transition

We also reproduced Example I of Macieszczak, Guta, Lesanovsky, and Garrahan.
Their three-level model has

:::{math}
H=\Omega_1(|1\rangle\langle0|+|0\rangle\langle1|)
+\Omega_2(|2\rangle\langle0|+|0\rangle\langle2|),
\qquad
J=\sqrt{\kappa}|0\rangle\langle1|.
:::

At $\Omega_2=0$, the active driven transition and the dark level $|2\rangle$
give two stationary modes. A small $\Omega_2$ couples them and produces one
slow decay mode at second order. At the parameters of the paper,
$\kappa=4\Omega_1$ and $\Omega_2=\Omega_1/10$, Pymablock gives

:::{math}
\lambda_{2,\mathrm{eff}}
=-\frac{20}{3}\frac{\Omega_2^2}{\Omega_1}
=-0.06667\,\Omega_1,
:::

while direct diagonalization gives
$\lambda_2=-0.07455\,\Omega_1$. The next mode has real part
$-0.77486\,\Omega_1$, so the exact slow-to-fast rate ratio is $0.0962$. This
reproduces the detached pair of low modes reported in the paper and explains
the metastable time window perturbatively.

The reported parameter choice also exposes a numerical point relevant to
Pymablock. The fast block has a degenerate optical-Bloch spectrum, making a
full eigenvector decomposition ill-conditioned. We instead construct the
slow subspace from $\ker\mathcal L_0$, the fast invariant subspace from
$\operatorname{range}\mathcal L_0$, and solve the Sylvester equation against
the undiagonalized fast block. This works because the slow spectral projector
is regular even though individual fast eigenvectors are not a useful basis.

## Reproduction: a two-qubit metastable manifold

The second metastability benchmark is a genuinely composite system. Following
the two-qubit model of Macieszczak and collaborators, one qubit controls
whether the other is pumped or damped,

:::{math}
H=\Omega_1\sigma_1^x+\Omega_2\sigma_2^x,
\qquad
J=\sqrt{\gamma_1}n_1\sigma_2^-
+\sqrt{\gamma_2}(1-n_1)\sigma_2^+.
:::

For $\gamma_1=4\gamma_2$, $\Omega_1=2\Omega_2$, and
$\Omega_2=\gamma_2/50$, direct diagonalization finds four slow modes. Their
decay rates in units of $\gamma_2$ are

:::{math}
0,\quad -9.93\times10^{-4},\quad -7.49\times10^{-3},
\quad -8.09\times10^{-3}.
:::

The ratio between the largest slow decay rate and the smallest fast decay
rate is $0.0164$. This is a more demanding reduction target than shelving:
the retained object is a four-dimensional operator manifold of a bipartite
system, not a single effective transition.

## Reproduction: third-order mediated superradiance

Kessler's example eliminates a decaying electron spin coupled to collective
nuclear-spin operators. The perturbation is

:::{math}
V=-ig\left[
\frac12(\sigma^+I^-+\sigma^-I^+)
+\sigma^+\sigma^-I^z,\mathord\cdot\right].
:::

We use the symmetric spin-one representation of two nuclear spins and retain
its complete nine-dimensional operator space. Pymablock reproduces the four
terms in Kessler's explicit third-order generator with maximum coefficient
error below $5\times10^{-16}$. Both the second- and third-order coefficients
are trace and Hermiticity preserving.

There is an apparent factor-of-four inconsistency in the published
second-order result. Direct evaluation of the paper's
$-V^-\mathcal L_0^{-1}V^+$, and Pymablock applied to the stated interaction
with prefactor $g/2$, give

:::{math}
\gamma_{\mathrm{eff}}
=\frac{g^2\gamma}{4[(\gamma/2)^2+\omega^2]},
\qquad
\omega_{\mathrm{eff}}
=-\frac{g^2\omega}{4[(\gamma/2)^2+\omega^2]}.
:::

The text prints the same expressions without the denominator's factor four.
The third-order formula is consistent with the stated $g/2$ interaction and
agrees directly. This benchmark therefore validates the nontrivial odd order
while also recording the convention mismatch rather than silently fitting it.

## Reproduction: fourth-order loss of complete positivity

We also encode the analytic obstruction of Tokieda, Elouard, Sarlette, and
Rouchon. For a thermally damped oscillator weakly coupled to a qubit, their
fourth-order reduced generator contains a dephasing coefficient proportional
to

:::{math}
-\frac{8n_{\mathrm{th}}(n_{\mathrm{th}}+1)
[3-6(2\Delta/\gamma)^2-(2\Delta/\gamma)^4]}
{\gamma^3[1+(2\Delta/\gamma)^2]^3}.
:::

It is negative for nonzero temperature and
$|\Delta|/\gamma<0.3406$. We now derive this term directly with Pymablock,
using the thermal oscillator state tensored with the full qubit operator space
as the slow basis. At $n_{\mathrm{th}}=0.2$ and $\Delta=0$, oscillator cutoffs
$3,4,5,6$ give

:::{math}
\gamma_{\phi,4}=-2.524,-4.510,-5.366,-5.650,
:::

which converges toward the analytic value $-5.760$. At cutoff six the maximum
error among $\omega_{B,4}$, $\gamma_{-,4}$, $\gamma_{+,4}$, and
$\gamma_{\phi,4}$ is $0.111$.
The fourth-order coefficient remains trace and Hermiticity preserving despite
its negative Kossakowski eigenvalue.

At cutoff seven we additionally compare all second- and fourth-order
coefficients at $\Delta/\gamma=0,0.2,0.4$. The maximum absolute coefficient
errors are $0.042$, $0.068$, and $0.050$, respectively. The fitted
$\gamma_{\phi,4}$ is negative at the first two detunings and positive at the
third, reproducing the analytic sign change at
$|\Delta|/\gamma=0.3406$ rather than checking only the resonant point.

This also gives a basis-independent infeasibility certificate. In the
orthonormal traceless basis
$F_i=\sigma_i/\sqrt2$, the second-order Kossakowski matrix $C_2$ has
$F_z\in\ker C_2$. Positivity of
$C(g)=g^2C_2+g^4C_4+O(g^6)$ would require

:::{math}
\langle F_z,C_4F_z\rangle\geq0.
:::

At cutoff six, Pymablock instead gives $-11.299$. This quadratic form is
unchanged by an orthonormal change of traceless operator basis, so the result
is not an artifact of expressing the generator with
$\sigma_\pm$ and $\sigma_z$ jumps.

The returned transformation is also physically useful: its slow columns are
the perturbative embedding of reduced qubit operators into the joint
oscillator--qubit space. Through third order they satisfy
$\mathcal L K=K\mathcal L_{\mathrm{eff}}$ with residual
$3.5\times10^{-13}$. Thus Pymablock already returns both objects needed to
separate accurate embedded dynamics from complete positivity on arbitrary
abstract reduced states.

## Next reproduction targets

The next useful benchmarks test features absent from the Lambda system:

1. A reduced-basis helper that exports selected embedding columns in an
   orthonormal basis would provide the natural interface for VPT-style
   recombination without coupling it to the block-diagonalization recurrence.
2. A multipoint version could pool those bases across expansion points and
   validate candidates using the full Liouvillian residual and spectral gap.
3. A periodically driven qubit in Sambe space tests whether the current
   non-Hermitian algorithm can reproduce a rotating-frame Floquet expansion.
