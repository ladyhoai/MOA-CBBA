# excavsim — Phase 1 Mesa skeleton (MOA-CBBA capstone)

Verified against **Mesa 3.5.1** (`pip install mesa pytest pandas`).

## Layout

| File | Maps to proposal |
|---|---|
| `excavsim/model.py` | §3.1 environment: 8-connected grid `G=(V,E)` via `OrthogonalMooreGrid`, property layers for terrain/`V(v)`/elevation, tasks, dump sites, seeded RNG, `DataCollector` (Table 1, item 1.3) |
| `excavsim/robot.py` | Robot `r_i` with `S_i` (immutable spec) and `M_i` (mutable state); travel→dig→dump→unload loop (Algorithm 1, line 8) |
| `excavsim/costs.py` | Eqs. (1), (3), (4), (5) — closed-form `n_ij`, `τ_ij`, `E_ij`, `J(x)` used for bidding |
| `excavsim/allocation.py` | Pluggable allocator interface; working greedy baseline; `CBBAAllocator` / `CBPAEAllocator` / `MOACBBAAllocator` scaffolds |
| `excavsim/belief.py` | Per-robot beliefs about the world's *dynamic* state — pile volumes, who is digging what, peer capacities. Updated from own sensors + gossip only; MOA-CBBA reads this instead of the shared `TaskRegistry` |
| `excavsim/pathfinding.py` | A* over the 8-connected grid (`d^{task/dump}_{ij}`) |
| `excavsim/tasks.py` | Task registry — the source of truth behind the BAM `x_ij` |
| `tests/test_costs.py` | Hand-computed checks of Eqs. (3)–(5) **and** proof that the tick-level simulation reproduces the closed forms exactly |
| `run_single.py` | One seeded run with metric printout |
| `run_batch.py` | `mesa.batch_run` sweep — the template for the CBBA/CBPAE/MOA-CBBA comparison |

## Run

```bash
python -m pytest tests/ -v   # validation suite
python run_single.py         # one seeded run
python run_batch.py          # reproducible sweep -> batch_results.csv
```

## GUI

```bash
pip install "mesa[viz]"
solara run app.py            # then open http://localhost:8765
```

Browser dashboard (`app.py`): terrain map in the Fig. 1 palette, robots
coloured by stage (grey idle, orange travelling, red digging, blue to
dump, purple unloading), yellow task squares, Play/Step/Reset, sliders
for robots/tasks/terrain mix, an allocator dropdown, and live plots of
tasks done, total energy and idle ratio. The GUI is for inspection and
debugging only — all reported results should come from the headless
`run_batch.py` path.

## Key design decisions

- **Distances are path lengths in grid steps** (each 8-connected move = 1
  step), so `d / v_max` is exact in ticks and the closed-form/simulation
  equivalence test can assert equality, not approximation. Switch to
  weighted A* when traversal cost varies (Phase 4).
- **Dig energy per tick is exactly β** — falls out of Eqs. (4)/(5): a dig
  tick removes `ρ_i/H` volume, so `β·dv·H/ρ_i = β`. The incremental
  physics therefore matches the closed form with no drift.
- **Allocators are the only swap point.** `allocate(model)` reads idle
  robots + pending tasks and calls `robot.assign(...)`. Model, metrics
  and batch code never change between algorithms.
- **`model.changed_cells`** is the hook for Algorithm 1 lines 3–7:
  Phase 4 terrain events should add coordinates there; MOA-CBBA's
  re-bid trigger reads it.

## Phase roadmap hooks

- **Phase 2 (heterogeneity):** pass per-robot `RobotSpec`s instead of
  `DEFAULT_SPEC`.
- **Phase 3 (partial observability):** give each robot a local copy of
  the property layers, updated within `sensor_range`; add a
  `last_observed` layer per robot.
- **Phase 4 (dynamics):** mutate `grid.terrain` / spawn obstacle agents
  in `model.step()`, record deltas in `changed_cells`, extend
  `blocked_cells()`.

## Validation to keep doing

- Cross-check the CBBA implementation against the MIT ACL reference
  (Choi, Brunet & How, 2009) on small instances.
- Keep a Hungarian-solver optimality check for single-shot assignments
  (scipy `linear_sum_assignment` on the τ/E cost matrix).
