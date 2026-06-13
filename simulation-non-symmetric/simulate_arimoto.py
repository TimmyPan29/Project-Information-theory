#!/usr/bin/env python3
"""Blahut--Arimoto computation of the capacity of a discrete memoryless channel.

The script runs the alternating-maximization (Arimoto, 1972) iteration on the
non-symmetric 3x3 channel used in the course deck, exports the per-iteration
convergence tables (CSV / LaTeX / Markdown), and -- with ``--verify`` -- checks
the converged input distribution against the Karush--Kuhn--Tucker optimality
conditions for channel capacity.

Conventions
-----------
* ``P`` is the channel matrix ``P[i, j] = p(y_i | x_j)`` (outputs index rows,
  inputs index columns), so every column sums to 1.
* All information quantities are in **nats** (natural logarithm).
* For an input distribution ``p`` and iteration ``t``:
    - ``C(t, t)``   is the mutual information ``I(X; Y)`` evaluated at ``p^t``;
    - ``C(t+1, t)`` is ``log sum_j r_j^t`` -- the capacity after the ``p``-update
      but still using the old posterior ``phi^t``.
  These satisfy ``C(t, t) <= C(t+1, t) <= C(t+1, t+1)`` (monotone increase).

The core routines (:func:`mutual_information`, :func:`arimoto`,
:func:`kkt_gaps`) take an arbitrary channel matrix, so the module is reusable
beyond the worked example.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Worked example: the non-symmetric 3x3 channel P[i, j] = p(y_i | x_j).
CHANNEL: NDArray[np.float64] = np.array(
    [
        [0.60, 0.70, 0.50],
        [0.30, 0.10, 0.05],
        [0.10, 0.20, 0.45],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Iteration:
    """One Arimoto iteration ``t``.

    Attributes
    ----------
    t:
        Iteration index.
    p:
        Input distribution ``p^t`` at the start of the iteration.
    capacity_current:
        ``C(t, t) = I(X; Y)`` evaluated at ``p^t``.
    p_next:
        Updated distribution ``p^{t+1}``.
    capacity_updated:
        ``C(t+1, t) = log sum_j r_j^t`` (uses the post-update ``p`` with the old
        posterior ``phi^t``).
    """

    t: int
    p: NDArray[np.float64]
    capacity_current: float
    p_next: NDArray[np.float64]
    capacity_updated: float


def _xlogy_sum_over_outputs(
    weights: NDArray[np.float64], ratio: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``sum_i weights[i, j] * log(ratio[i, j])`` per column ``j``.

    Terms with ``weights == 0`` contribute 0 even where ``ratio`` is 0, matching
    the information-theory convention ``0 * log 0 = 0`` and keeping the result
    finite for channels with zero transition probabilities.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = weights * np.log(ratio)
    return np.where(weights > 0.0, contrib, 0.0).sum(axis=0)


def mutual_information(channel: NDArray[np.float64], p: NDArray[np.float64]) -> float:
    """Mutual information ``I(X; Y)`` in nats for input ``p`` over ``channel``."""
    q = channel @ p  # output marginal p(y_i)
    divergence = _xlogy_sum_over_outputs(channel, channel / q[:, None])
    return float(divergence @ p)


def kkt_gaps(
    channel: NDArray[np.float64], p: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """Return ``(I, D)`` where ``D[j] = D( p(.|x_j) || p(y) )`` in nats.

    At the capacity-achieving input the Kuhn--Tucker conditions read
    ``D[j] == I`` for every ``j`` with ``p[j] > 0`` and ``D[j] <= I`` otherwise;
    ``I`` is then the channel capacity.
    """
    q = channel @ p
    divergence = _xlogy_sum_over_outputs(channel, channel / q[:, None])
    return float(divergence @ p), divergence


def _update(
    channel: NDArray[np.float64], p: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """One Arimoto p-update; returns ``(p_next, C(t+1, t))``."""
    q = channel @ p
    posterior = channel * p / q[:, None]  # phi^t(x_j | y_i)
    r = np.exp(_xlogy_sum_over_outputs(channel, posterior))
    normalization = r.sum()
    return r / normalization, float(np.log(normalization))


def arimoto(channel: NDArray[np.float64], max_t: int) -> list[Iteration]:
    """Run ``max_t + 1`` Arimoto iterations from the uniform input distribution."""
    n = channel.shape[1]
    p = np.full(n, 1.0 / n)
    history: list[Iteration] = []

    for t in range(max_t + 1):
        capacity_current = mutual_information(channel, p)
        p_next, capacity_updated = _update(channel, p)
        history.append(
            Iteration(t, p.copy(), capacity_current, p_next.copy(), capacity_updated)
        )
        p = p_next

    return history


def run_to_convergence(
    channel: NDArray[np.float64], tol: float = 1e-12, max_iter: int = 200_000
) -> tuple[NDArray[np.float64], int]:
    """Iterate until the input distribution stops changing; return ``(p*, iters)``."""
    n = channel.shape[1]
    p = np.full(n, 1.0 / n)
    for iters in range(1, max_iter + 1):
        p_next, _ = _update(channel, p)
        if np.max(np.abs(p_next - p)) < tol:
            return p_next, iters
        p = p_next
    return p, max_iter


def selected_indices(max_t: int, step: int) -> list[int]:
    """Indices ``0, step, 2*step, ...`` always including ``max_t``."""
    indices = list(range(0, max_t + 1, step))
    if indices[-1] != max_t:
        indices.append(max_t)
    return indices


# --------------------------------------------------------------------------- #
# Output writers. ``rows`` is a list of (t, capacity, p-vector) triples.
# --------------------------------------------------------------------------- #
Row = tuple[int, float, NDArray[np.float64]]


def _headers(n: int) -> list[str]:
    return [f"p(x{j + 1})" for j in range(n)]


def write_csv(path: Path, rows: list[Row], capacity_label: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["t", f"{capacity_label} [nats/use]", *_headers(len(rows[0][2]))])
        for t, capacity, p in rows:
            writer.writerow([t, f"{capacity:.12f}", *(f"{v:.12f}" for v in p)])


def write_latex(path: Path, rows: list[Row], capacity_label: str) -> None:
    n = len(rows[0][2])
    header = " & ".join([f"$p(x_{j + 1})$" for j in range(n)])
    lines = [
        r"\begin{tabular}{r" + "c" * (n + 1) + "}",
        r"\toprule",
        f"$t$ & ${capacity_label}$ & {header} \\\\",
        r"\midrule",
    ]
    lines += [
        f"{t} & {capacity:.9f} & " + " & ".join(f"{v:.9f}" for v in p) + r" \\"
        for t, capacity, p in rows
    ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[Row], capacity_label: str) -> None:
    path.write_text(_markdown(rows, capacity_label) + "\n", encoding="utf-8")


def _markdown(rows: list[Row], capacity_label: str) -> str:
    n = len(rows[0][2])
    head = f"| t | {capacity_label} [nats/use] | " + " | ".join(_headers(n)) + " |"
    sep = "|---:" * (n + 2) + "|"
    body = [
        f"| {t} | {capacity:.9f} | " + " | ".join(f"{v:.9f}" for v in p) + " |"
        for t, capacity, p in rows
    ]
    return "\n".join([head, sep, *body])


def export(path_stem_args: argparse.Namespace, history: list[Iteration]) -> None:
    """Write both the ``C(t+1, t)`` and ``C(t, t)`` table families."""
    indices = selected_indices(path_stem_args.max_t, path_stem_args.step)

    updated_rows: list[Row] = [
        (h.t, h.capacity_updated, h.p_next) for h in (history[t] for t in indices)
    ]
    current_rows: list[Row] = [
        (h.t, h.capacity_current, h.p) for h in (history[t] for t in indices)
    ]

    write_csv(path_stem_args.csv, updated_rows, "C(t+1,t)")
    write_latex(path_stem_args.latex, updated_rows, "C(t+1,t)")
    write_markdown(path_stem_args.md, updated_rows, "C(t+1,t)")

    write_csv(path_stem_args.current_csv, current_rows, "C(t,t)")
    write_latex(path_stem_args.current_latex, current_rows, "C(t,t)")
    write_markdown(path_stem_args.current_md, current_rows, "C(t,t)")

    print(_markdown(updated_rows, "C(t+1,t)"))
    print()
    print(_markdown(current_rows, "C(t,t)"))


def verify(channel: NDArray[np.float64], atol: float = 1e-6) -> bool:
    """Iterate to convergence and print the KKT optimality report.

    Returns ``True`` iff the converged input distribution satisfies the
    channel-capacity Kuhn--Tucker conditions to within ``atol``.
    """
    p_star, iters = run_to_convergence(channel)
    capacity, divergence = kkt_gaps(channel, p_star)
    support = p_star > 1e-9

    support_ok = bool(np.allclose(divergence[support], capacity, atol=atol))
    unused_ok = bool(np.all(divergence <= capacity + atol))

    print(f"\n=== KKT optimality check (converged after {iters} iterations) ===")
    print(f"capacity C            : {capacity:.9f} nats = {capacity / np.log(2):.9f} bits")
    print(f"p*                    : {np.array2string(p_star, precision=9)}")
    print("D( p(.|x_j) || p(y) ) : "
          f"{np.array2string(divergence, precision=9)}  (nats)")
    print(f"C - D per input       : {np.array2string(capacity - divergence, precision=9)}")
    print(f"support D == C        : {support_ok}")
    print(f"all D <= C            : {unused_ok}")
    print(f"=> KKT satisfied      : {support_ok and unused_ok}")
    return support_ok and unused_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-t", type=int, default=114, help="last iteration index")
    parser.add_argument("--step", type=int, default=1, help="row stride in the tables")
    parser.add_argument("--verify", action="store_true", help="print the KKT check")
    parser.add_argument("--csv", type=Path, default=Path("arimoto-results.csv"))
    parser.add_argument("--latex", type=Path, default=Path("arimoto-results-table.tex"))
    parser.add_argument("--md", type=Path, default=Path("arimoto-results-table.md"))
    parser.add_argument(
        "--current-csv", type=Path, default=Path("arimoto-current-results.csv")
    )
    parser.add_argument(
        "--current-latex", type=Path, default=Path("arimoto-current-results-table.tex")
    )
    parser.add_argument(
        "--current-md", type=Path, default=Path("arimoto-current-results-table.md")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_t < 0:
        raise ValueError("--max-t must be non-negative")
    if args.step <= 0:
        raise ValueError("--step must be positive")

    history = arimoto(CHANNEL, args.max_t)
    export(args, history)
    if args.verify:
        verify(CHANNEL)


if __name__ == "__main__":
    main()
