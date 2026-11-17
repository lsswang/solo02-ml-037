# Heterogeneous-label collaborative distillation

Local tanh classifiers with **different label subsets, input dimensions, hidden
widths and label orders** collaborate without ever averaging parameters:
zero-padding would misinterpret "this head does not exist" as "impossible", so
instead the server solves a well-posed convex problem for a global teacher and
each client distils that teacher back into its own labels.

Python 3.11, NumPy + SciPy only. No networking, scheduling, model store or
secure-aggregation protocol.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy pytest
.venv/bin/python demo.py --pretty      # recomputes artifacts/report.json
.venv/bin/python -m pytest -q          # 61 tests: gradients, optimality, errors
```

`demo.py` runs one complete, replayable round on `fixtures/clients.json`:

1. round-initial models emit conditional reports in their own label order;
2. one server call turns the reports into the global teacher `Q`;
3. each client takes exactly one clipped local gradient step at that `Q`;
4. everything is written to `artifacts/report.json`.

It also runs an **intervention**: changing client 0's round-initial model changes
its report, changes `Q`, and changes client 1's actual gradient and update (the
consumer gradient is recomputed at old vs new `Q`). Independent checks are
included: finite-difference gradients, and a separate SciPy `trust-ncg` solve
of the server objective.

## Layout

| Path | Contents |
| --- | --- |
| `starter/local_model.py` | tanh forward, local objective + analytic gradients, renormalised teacher |
| `starter/consensus.py` | report validation, convex global-logit solver, diagnostics, zero-padded baseline |
| `starter/round.py` | fixed-order round, one clipped local step, intervention contrast |
| `starter/validation.py` | strict shape/value checks → `InputError` |
| `starter/verify.py` | finite-difference and independent-solver checks |
| `tests/` | local gradients, server optimality, reference recomputation, round replay, errors |
| `NUMERICAL_METHODS.md` | objective derivation, solver, convergence criteria, cost |

## Key formulas

Local teacher renormalisation (never a truncated row):

```
t[r,l] = Q[r,S[l]] / Σ_{k∈S} Q[r,k]
```

Server objective per public row (strictly convex for `ridge > 0`; exact
normaliser of conditional reports):

```
F(z) = Σ_{i:present} a_i ( logsumexp(z[S_i]) − Σ_l p[i,r,l] z[S_i[l]] )
     + ridge/2 ‖z‖²,      Q = softmax(z)
```

`max|∇F| ≤ 1e-7` is the only convergence criterion.

## Notes / assumptions

* `step_size` in `fixtures/clients.json` is accepted as an alias for `eta`.
* The fixture's `report_weight`/`present` default to `1`/all-true when absent.
* Generated outputs go to `artifacts/` (git-ignored); fixtures are untouched.
