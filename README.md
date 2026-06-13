# Channel Capacity via the Blahut–Arimoto Algorithm

A detailed derivation, implementation, and **independently verified** numerical
study of S. Arimoto's algorithm for computing the capacity of an arbitrary
discrete memoryless channel (DMC).

The repository pairs a from-scratch presentation of the theory
([`arimoto-practice.tex`](arimoto-practice.tex)) with a small, tested NumPy
implementation ([`simulation-non-symmetric/`](simulation-non-symmetric/)) whose
output is checked against the Karush–Kuhn–Tucker (KKT) optimality conditions for
channel capacity — so the computed number is provably the true capacity, not
merely a fixed point of the iteration.

---

## The problem

The capacity of a DMC with transition matrix $P(y\mid x)$ is

$$
C = \max_{p(x)} I(X;Y),
\qquad
I(X;Y) = \sum_{x,y} p(x)\,p(y\mid x)\,\log\frac{p(y\mid x)}{\sum_{x'} p(x')\,p(y\mid x')}.
$$

The denominator couples every input probability inside one logarithm, so
differentiating $I$ directly gives a nonlinear system with no closed form.
Arimoto's insight is to introduce an auxiliary "backward" distribution
$\phi(x\mid y)$ and write capacity as a **double maximization**

$$
C = \max_{p}\ \max_{\phi}\ \sum_{x,y} p(x)\,p(y\mid x)\,\log\frac{\phi(x\mid y)}{p(x)},
$$

which is solved by alternating maximization — each half-step has a closed form
and never decreases the objective.

## The algorithm

Starting from the uniform input $p^0$, iterate for $t = 0, 1, 2, \dots$:

$$
\phi^t(x_j\mid y_i) = \frac{p(y_i\mid x_j)\,p^t(x_j)}{\sum_k p(y_i\mid x_k)\,p^t(x_k)},
\qquad
r_j^t = \exp\!\Big(\textstyle\sum_i p(y_i\mid x_j)\,\log\phi^t(x_j\mid y_i)\Big),
$$

$$
p^{t+1}(x_j) = \frac{r_j^t}{\sum_k r_k^t},
\qquad
C(t{+}1, t) = \log\sum_j r_j^t .
$$

Capacity increases monotonically along the half-steps,
$C(t,t) \le C(t{+}1,t) \le C(t{+}1,t{+}1)$, and converges to $C$.

## Worked example: a non-symmetric $3\times 3$ channel

$$
P(Y\mid X) =
\begin{bmatrix}
0.60 & 0.70 & 0.50\\
0.30 & 0.10 & 0.05\\
0.10 & 0.20 & 0.45
\end{bmatrix}
\quad(\text{columns are } p(\cdot\mid x_j)).
$$

The columns are not permutations of one another, so the channel is asymmetric
and the uniform input is **not** optimal. Running the algorithm to convergence:

| quantity | value |
|---|---|
| Capacity $C$ | **0.112035 nats/use** = 0.161632 bits/use |
| Optimal input $p^\*$ | $(0.50174,\ \approx 1.5\times10^{-11},\ 0.49826)$ |
| Iterations to converge | 377 (to $\lVert\Delta p\rVert_\infty < 10^{-12}$) |

**Key finding — a symbol is abandoned.** The optimal distribution drives
$p^\*(x_2)\to 0$: the output of $x_2$ is statistically close to a mixture of
$x_1$ and $x_3$, so it carries almost no extra information and the
capacity-optimal code simply does not use it.

### Why this is the *true* capacity (not just a fixed point)

At the capacity-achieving input, the KKT conditions require the relative entropy
$D\big(p(\cdot\mid x_j)\,\Vert\, p(y)\big)$ to equal $C$ on the support and be
$\le C$ off it. The implementation checks exactly this (`--verify`):

```
D( p(.|x_j) || p(y) ) : [0.112034669  0.049134899  0.112034669]   (nats)
capacity C            :  0.112034669
=> support D == C, unused symbol D < C  =>  KKT satisfied : True
```

$D(x_1)=D(x_3)=C$ on the support and $D(x_2)=0.0491 < C$ for the dropped symbol —
a certificate that the result is the global optimum.

## Repository layout

```
.
├── arimoto-practice.tex          # Beamer slides: full derivation + worked example + appendix
├── arimoto-practice.pdf          # compiled slides
├── README.md
└── simulation-non-symmetric/
    ├── simulate_arimoto.py        # NumPy implementation (general DMC) + KKT verifier + table export
    ├── test_arimoto.py            # tests: monotonicity, regression, KKT, BSC closed form
    ├── requirements.txt
    └── arimoto-*results*.{csv,md,tex}   # generated convergence tables
```

The core routines (`mutual_information`, `arimoto`, `run_to_convergence`,
`kkt_gaps`) accept an arbitrary channel matrix; the worked example is just the
default `CHANNEL` constant.

## Usage

```bash
cd simulation-non-symmetric
pip install -r requirements.txt

# Reproduce the convergence tables (CSV / Markdown / LaTeX) and print the KKT certificate
python3 simulate_arimoto.py --verify

# Run the tests (no pytest needed; pytest also works if installed)
python3 test_arimoto.py
```

Build the slides with `pdflatex arimoto-practice.tex` (run twice for the page
counter; the `Images/` logo and bundled `.sty` files are included).

## References

1. S. Arimoto, "An algorithm for computing the capacity of arbitrary discrete
   memoryless channels," *IEEE Trans. Inf. Theory*, vol. 18, no. 1, pp. 14–20, 1972.
2. R. Blahut, "Computation of channel capacity and rate-distortion functions,"
   *IEEE Trans. Inf. Theory*, vol. 18, no. 4, pp. 460–473, 1972.
3. T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006.
