# Numerical methods, cost, and reproducibility

This document explains *what* is computed, *why* it is the specified objective,
*how* it is solved, and what it costs. All code is deterministic; `python3 demo.py`
recomputes `artifacts/report.json` from the two fixtures.

## 1. Local objective (client side)

For one client with label list `S` (length `L = |S|`, arbitrary order), a
two-layer tanh classifier

```
H = tanh(X A + b),   Z = H C + c,   P = softmax(Z),      # columns indexed by S
```

labelled training rows `(x_i, y_i)` (global exact label, `y_i ∈ S`), positive
integer weights `w_i ∈ {1..5}`, public rows `public_X[r], public_weights[r]`, and
an **external global teacher** `Q ∈ R^{R×K}`:

```
t[r,l] = Q[r, S[l]] / Σ_{k∈S} Q[r,k]        # renormalised restriction to S
```

The objective is

```
L(θ) = Σ_i w_i · (−log P[i, y_i]) / Σ_i w_i
     + τ · Σ_r pw_r · (− Σ_l t[r,l] log P_public[r,l]) / Σ_r pw_r
     + reg/2 · (‖A‖² + ‖b‖² + ‖C‖² + ‖c‖²).
```

* `t` is a **constant** (no gradient flows into `Q`).
* The teacher term is the cross-entropy against the row-renormalised teacher, so
  truncating `Q` to `S` never leaves a row sum below 1.
* `τ = 0` removes the teacher term exactly: it degenerates to the ordinary
  weighted exact-label cross-entropy. `reg ∈ [0.05, 1]`, `τ ∈ [0, 2]`.
* Softmax and log-sum-exp are evaluated with the shift `Z − logsumexp(Z)`; the
  teacher log uses `log(max(P, 1e-300))`.

### Analytic gradient

With `G_sup[i,:] = (w_i/Σw)·(P[i,:] − onehot(y_i))` and
`G_dis[r,:] = τ·(pw_r/Σpw)·(P_public[r,:] − t[r,:])`, stacked as `G` over the
concatenated training/public rows and hidden activations `H`, the
backpropagation is the usual

```
dH = (G Cᵀ) ∘ (1 − H²),   ∇A = Xᵀ dH,   ∇b = Σ dH,   ∇C = Hᵀ G,   ∇c = Σ G,
```

plus `reg·θ` for every parameter. Cost: one forward + one backward over
`n + R ≤ 38` rows, i.e. `O((n+R)·d·H + (n+R)·H·L)` multiply-adds; the parameter
space has at most `4·3 + 3 + 3·6 + 6 = 39` scalars. Finite-difference checks in
`tests/` and `starter/verify.py` agree to `~1e-10` absolute / `~1e-8` relative.

## 2. Server consensus (aggregation)

Each report `i` supplies `S_i`, a **conditional** distribution `p[i,r,:]`
over `S_i` (strictly positive, row sum 1), weight `a_i ∈ [0.1, 5]`, and a
`present` mask. For each public row `r` independently, solve

```
F_r(z) = Σ_{i : present[i,r]} a_i · ( logsumexp(z[S_i]) − Σ_l p[i,r,l]·z[S_i[l]] )
       + ridge/2 · ‖z‖² ,        ridge ∈ [0.05, 1].
```

Why this is the right aggregate:

* The `logsumexp(z[S_i])` term is the **exact normaliser** of report `i`'s
  conditional distribution under the global logits; without it, the objective
  would treat each report as an unconditional distribution and double-count the
  within-`S_i` normalisation. A missing class is penalised only through
  `logsumexp`, never by treating "no head" as "impossible".
* `ridge > 0` makes `F_r` **strictly convex** (the data terms are convex, the
  quadratic is strongly convex) and selects the common additive shift of `z`
  (the data terms are shift-invariant; the ridge fixes it). Hence the minimiser
  exists and is unique.
* Unreported rows: `F_r(z) = ridge/2‖z‖²` has the unique minimiser `z = 0`, so
  `Q` is uniform. Disconnected or overlapping label subsets are perfectly legal.

Gradient and Hessian on the subset `S_i` (with `q = softmax(z[S_i])`):

```
∇F = Σ_i a_i (q − p_i) + ridge·z
∇²F = Σ_i a_i (diag(q) − q qᵀ) + ridge·I
```

The Hessian is symmetric positive definite for `ridge > 0`.

### Solver

Per row, damped **Newton** with backtracking line search:

```
solve H d = −g            # small K×K system (K ≤ 6)
z ← z + α d,  α halved until Armijo F(z+αd) ≤ F(z) + 1e-4 α gᵀd
```

Start at `z = 0`. Stop when `max|∇F| ≤ 1e-7` (reported as `converged=True`),
otherwise after `max_iter` (default 500, positive integer) iterations or on a
numerical failure (linear solve / non-finite step / failed line search). In every
non-converged case the **best finite iterate** is returned with
`converged=False` and an explicit message; it never claims success or
infeasibility falsely. Per-row iteration counts, gradient norms, messages, and
objective history are returned in `diagnostics`.

**Cost.** Per row: one `K×K` linear solve plus a few matrix–vector products over
reports per Newton step. With `K ≤ 6` and `≤ 4` reports, each step is tens of
flops; normal fixtures converge in 2–4 steps. Cross-row work is fully
independent (no cross-row coupling beyond the identical formulation), so total
cost is `O(R · Newton_steps · (K³ + Σ_i |S_i|²))` — trivially small.

An **independent** solver (`starter/verify.py`, SciPy `trust-ncg` with analytic
gradient/Hessian) reproduces `z` to `~1e-9` (both solvers stop at their own
tolerance); plain-loop reference evaluations match `F` and `∇F` to `1e-8`.

## 3. One collaborative round

Fixed order, executed by `starter.round.run_round`:

1. **Reports.** Each client evaluates its round-initial model on `public_X` and
   emits `P_public` in its own `S` order (plus its report weight and mask).
2. **Global teacher.** The single server entry point `consensus(...)` produces
   `Q = softmax(z)` from those reports.
3. **Local step.** Each client computes one full local gradient *at that `Q`*
   (exact labels + renormalised teacher) and applies **one** clipped step
   `θ_new = clip(θ − η·∇L, −2, 2)`, `η ∈ [0.001, 0.2]`.
4. **Record.** Reports, `Q`, per-client loss-before, gradient, updated model and
   pre/post public probabilities are all returned and serialised by `summary`.

No re-aggregation of updated models and no inner training loop. Next-round models
can be fed in explicitly.

### Intervention (Q is actually consumed)

`run_round` is run, then run again with one client's round-initial model changed.
`intervention_contrast` reports:

* the change in that client's conditional report,
* the change in global `Q`,
* for each *other* client, the local gradient recomputed at the **old** vs **new**
  `Q`, and the resulting update difference.

The consumer's gradient/update provably changes, so the teacher is consumed at
the gradient, not merely printed. With `τ = 0` the gradient is teacher-independent
and the contrast vanishes (as it must).

## 4. Validation

`starter/validation.py` rejects with `InputError` (a `ValueError`): wrong
dimensionality/shapes, empty inputs, non-finite values, classes outside `[3, 6]`,
duplicate or out-of-range labels, report probabilities `< 1e-12` or row sums off
by `> 1e-12`, teacher entries `< 1e-300` or row sums off by `> 1e-12`, weights
outside their allowed ranges, non-boolean masks, `reg`/`τ`/`ridge`/`η`/`max_iter`
outside their ranges, and mismatched `R` across reports/clients.

## 5. Tests

`tests/` (run `python3 -m pytest -q`) covers finite-difference gradients across
input dims, hidden widths and label counts; `τ = 0` degeneracy; renormalised (not
truncated) teacher; global→local label remapping; stable large logits;
server first-order optimality at `1e-7`; independent-solver agreement;
unreported rows → `z = 0`, uniform `Q`; disconnected/overlapping subsets; mixed
`present` masks; `max_iter` capping; weight sensitivity; the zero-padded baseline
failure mode; full-round replay; one-step/clip checking; teacher consumption; and
intervention effects.
