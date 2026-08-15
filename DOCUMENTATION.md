# excavsim — Documentation

A guided tour of this project for new contributors, reviewers, or
anyone curious how it works, written to assume no prior familiarity
with the codebase, the underlying research papers, or (much) Python.

If you just want a quick file map and commands to run, see
[README.md](README.md) — it's a terse reference. This document is the
longer walkthrough: what the simulation actually models, why it's built
the way it is, and how the pieces connect.

**Contents**

| § | Section | For |
|---|---|---|
| 1 | [What this project is](#1-what-this-project-is) | the one-paragraph version |
| 2 | [The world, in plain terms](#2-the-world-in-plain-terms) | what is being simulated |
| 3 | [Bids and auctions](#3-how-a-decision-gets-made-bids-and-auctions) | the core idea behind every allocator |
| 4 | [The four allocators](#4-the-four-allocators-explained) | how each algorithm works |
| 5 | [Program flow](#5-program-flow) | **what actually executes, in order** |
| 6 | [Module map](#6-module-map) | which file does what |
| 7 | [Running it](#7-running-it) | commands, GUI, diagnostics |
| 8 | [Glossary](#8-glossary) | unfamiliar terms |
| 9 | [Known limitations](#9-known-limitations) | what doesn't work yet |
| A | [**Appendix A — The mathematics**](#appendix-a--the-mathematics) | **every formula, derived and located** |

---

## 1. What this project is

`excavsim` is a simulation of a **fleet of autonomous excavation
robots** digging up dirt at several sites on a map and hauling it to
dump sites, and it compares different strategies for deciding
**which robot should do which job**. That decision problem — "given a
set of robots and a set of jobs, who does what, in what order, to
finish as fast as possible using as little energy as possible" — is
called **multi-robot task allocation**, and it's the actual subject of
the project. The digging simulation is the test bed used to measure how
well each allocation strategy performs.

Three real algorithms from robotics research are implemented and
compared, plus a simple baseline:

| Allocator | Idea in one sentence |
|---|---|
| `greedy` | Each idle robot just grabs whichever unclaimed job looks cheapest right now. No coordination. |
| `cbba` | **Consensus-Based Bundle Algorithm** (Choi, Brunet & How, 2009). Robots build private wish-lists of jobs, then gossip with each other until they all agree on who owns what — with no central coordinator. |
| `cbpae` | **Consensus-Based Parallel Auction and Execution** (Das et al., 2015). Like CBBA, but a robot keeps bidding on its *next* job while still finishing its *current* one, so there's no idle gap between jobs. |
| `moa-cbba` | **Multi-Objective Adaptive CBBA** — this project's own extension of CBBA, adding workload balancing, capacity-aware matching (big robots preferring big piles), and letting several robots share one large job. |

Everything is built on [Mesa](https://mesa.readthedocs.io/), a Python
framework for agent-based simulations, and can be watched live in a
browser dashboard or run headless for batch experiments.

---

## 2. The world, in plain terms

- The world is a rectangular **grid** of cells (default 32×32). Each
  cell has a **terrain type** — `SOIL`, `GRAVEL`, `ROCK` (all diggable,
  with increasing difficulty), `BEDROCK` (a permanent wall) or
  `DUMP_SITE` (where dug material is delivered) — plus an **elevation**
  (climbing uphill costs extra energy) and a **soil volume** (how much
  diggable material remains at that cell).
- A handful of cells are marked as **tasks**: piles of dirt with a
  target volume that shrinks as robots dig them. A task is "done" once
  its volume reaches zero *and* every bit of it has actually been
  hauled to a dump site (not just dug up — see `Task`/`TaskRegistry`
  in [excavsim/tasks.py](excavsim/tasks.py)).
- A handful of robots start somewhere on the grid, each with its own
  **spec**: how much dirt it can carry (`capacity`), how fast it moves
  (`v_max`), how fast it digs (`dig_rate`), how much battery it has, and
  how far it can sense. Robots come in `small` / `medium` / `large`
  classes (see [excavsim/fleet.py](excavsim/fleet.py)) so the fleet can
  be homogeneous or a genuinely mixed team.
- Every robot repeats the same four-step loop for each task it's
  assigned: **drive to the dig site → dig until full or the task is
  empty → drive to a dump site → unload → repeat** (or go idle, if
  nothing is left to do). This loop, and the finite-state machine that
  drives it (`Stage.IDLE / TO_TASK / DIG / TO_DUMP / UNLOAD`), lives in
  [excavsim/robot.py](excavsim/robot.py).
- Time moves in discrete **ticks**. Every tick: the world may change
  (a hazard appears, an obstacle wanders, weather shifts — all optional
  and off by default), robots sense their surroundings, the chosen
  allocator decides/updates who's doing what, then every robot advances
  one tick of movement/digging/unloading. The simulation stops once
  every task is fully delivered.

Optional realism on top of that baseline (all switchable, mostly off by
default so experiments start from a clean, deterministic world):

- **Heterogeneous fleets** — robots with genuinely different specs.
- **Limited sensing** — a robot only "knows" about hazards/obstacles it
  can currently see, and has to react/re-route as it discovers more
  (rather than knowing the whole map from tick one).
- **Communication limits** — messages between robots have a range,
  can be lost, delayed, or rate-limited, instead of every robot hearing
  every other robot instantly.
- **Dynamic hazards, wandering obstacles, and weather** — the map isn't
  static; robots have to adapt mid-run.

---

## 3. How a decision gets made: bids and auctions

Every allocator, no matter how it coordinates, needs to answer one
underlying question over and over: **"If robot *i* took task *j* right
now, how many ticks would it take, and how much energy would it cost?"**
That calculation is called a **bid**, or a **leg cost**.

- [excavsim/costs.py](excavsim/costs.py) has the *pure math* version:
  given a task's volume and terrain hardness, a robot's spec, and the
  distances involved, it computes the exact number of ticks (`tau_ij`)
  and energy (`energy_ij`) the job would take, straight from the same
  formulas the paper proposal uses (Eqs. 3–5).
- [excavsim/bidding.py](excavsim/bidding.py) is the layer that
  actually gets called during a live simulation: it runs A* pathfinding
  ([excavsim/pathfinding.py](excavsim/pathfinding.py)) to find real
  distances around obstacles, then plugs them into the costs.py
  formulas. This is `leg_cost()`, the single most-called function in
  the codebase.
- A key correctness property the whole project relies on is that **a
  bid should match what actually happens** when a robot executes it
  uncontested (no other robots or obstacles in the way). This is
  verified directly in
  [tests/test_costs.py](tests/test_costs.py) — one robot, one task, no
  interference, and the tick-by-tick simulation is checked against the
  closed-form formula to the last decimal.

Once every robot can price any task, the allocators differ only in
**how they turn a pile of bids into an assignment** — see the next
section.

---

## 4. The four allocators, explained

All of them share the same interface: `allocate(model)` looks at idle
robots and open tasks, and calls `robot.assign(task_id)` for new
pairings. This is deliberate — swapping `allocator="cbba"` for
`allocator="moa-cbba"` in `ExcavationModel(...)` is the *only* thing
that changes between experiments; the world, the robots, and the
metrics code stay identical. See
[excavsim/allocation.py](excavsim/allocation.py)'s `Allocator` base
class and the `ALLOCATORS` registry.

### Greedy (`excavsim/allocation.py`, `GreedyAllocator`)
The simplest possible baseline: each idle robot, in random order, takes
the cheapest task it can reach that's still unclaimed. No negotiation,
no lookahead. Useful as a lower bar the other algorithms should beat.

### CBBA (`excavsim/allocation.py`, `CBBAAgent` / `CBBAAllocator`)
Each robot privately builds a **bundle** — an ordered wish-list of
tasks it wants, ranked by how much *reward* each addition would earn it
(a discounted, always-positive score; see `pathScoreCBBA`). Robots then
broadcast their current best claims to nearby robots and merge what
they hear using a fixed rulebook (`_CBBADecisionTable`, straight from
the original paper's "Table I") — essentially: "if my neighbour's claim
on task *j* is fresher or better than mine, adopt it; otherwise keep
mine." Repeating this gossip-and-merge cycle a bounded number of rounds
provably converges to a conflict-free assignment with no central
coordinator, even if robots can only talk to nearby robots rather than
the whole fleet.

### CBPAE (`excavsim/CBPAE.py`)
CBBA (and greedy) only start looking for a robot's *next* task once
it's finished its current one — there's a "dead" gap between jobs.
CBPAE's key idea is to close that gap: a robot keeps bidding on
candidate next-tasks *while still executing* its current one, and that
bid gets more competitive every round as the current job nears
completion (because less work is left to finish it — see
`residual_cost` in bidding.py). Consensus is reached through explicit
rule tables for "the task I'm bidding on" vs. "any other task" vs. "two
robots executing the same task" (`_bidTaskRule` / `_otherTaskRule` /
`_parallelExecutionRule`), matching the original paper's Tables 5–7.

### MOA-CBBA (`excavsim/MOACBBA.py`)
This project's own extension of CBBA, aimed at two things a plain
reward-maximising auction handles poorly for this domain:

1. **Workload balance.** A robot that's already behind schedule (its
   projected finish time is the fleet's bottleneck) should find every
   further task expensive; a robot with spare time should find tasks
   cheap. Bids are priced against the *effect on the fleet's makespan*,
   not just the individual robot's own cost.
2. **Sharing large jobs.** Several robots can be assigned to the same
   big task at once, draining its volume in parallel instead of one
   robot working through it alone (`Task.assignees` is a set, not a
   single robot id — see [excavsim/tasks.py](excavsim/tasks.py)).

It also supports (configurable, off by default where measured to be a
net loss) letting a robot switch to a cheaper task mid-approach if it
discovers one, and preferring to match big robots to big remaining
piles. Every one of these mechanisms can be individually disabled to
measure what it's actually contributing — see the constants at the top
of `MOACBBA.py` (`MAX_SHARERS`, `CAPACITY_AFFINITY`,
`ENABLE_SWITCHING`, etc.) and the extensive comments explaining what was
measured when each one was tuned.

---

## 5. Program flow

This section traces what actually executes, from launching a script to
the run ending. Read it alongside the module map in §6.

### 5.1 Startup — building a world

Everything begins with one call: `ExcavationModel(...)`. Its
`__init__` builds the whole world once, in this order (the order
matters — each step depends on the last):

```
ExcavationModel.__init__
  ├─ 1. Grid + property layers    OrthogonalMooreGrid, then three
  │                               PropertyLayers: terrain, soil_volume,
  │                               elevation
  ├─ 2. _scatter_terrain()        per-cell random SOIL/GRAVEL/ROCK
  ├─ 3. _scatter_bedrock()        impassable ridges, each rolled back if
  │      └─ _free_space_connected()   it would sever the map
  ├─ 4. _scatter_elevation()      fractal (Perlin) noise height field
  ├─ 5. _place_dump_block() x2    2×2 impassable DUMP_SITE blocks
  ├─ 6. DynamicsManager(...)      hazards/obstacles/weather (built early:
  │                               bids consult it from the first tick)
  ├─ 7. TaskRegistry + tasks      n_tasks piles at random diggable cells,
  │      └─ _random_diggable_coord()  each with a TaskMarker for the GUI
  ├─ 8. build_fleet() + robots    one RobotSpec per robot → ExcavatorRobot
  ├─ 9. CommNetwork(...)          the messaging layer
  ├─ 10. ALLOCATORS[allocator]()  instantiate the chosen algorithm
  └─ 11. DataCollector(...)       metric reporters, then collect tick 0
```

### 5.2 The tick loop — what happens every step

`model.step()` is the heartbeat. It is called repeatedly by
`model.run()`, by `mesa.batch_run`, or by the GUI's Play button, until
every task is done. Each tick runs these six stages **in this order**,
and the ordering is deliberate at every boundary:

```
model.step()                                    ── one tick ──
  │
  ├─ 1. dynamics.step(tick)      World changes first, so everyone
  │                              reacts to THIS tick's world.
  │                              Expire/spawn hazards, move obstacles,
  │                              maybe change weather.
  │
  ├─ 2. comms.flush_and_deliver()  Messages sent last tick arrive.
  │
  ├─ 3. for r in robots: r.sense()  Robots update their private maps
  │                              BEFORE bidding — a bid priced on a
  │                              stale map is a bid that can't execute.
  │
  ├─ 4. _leg_cache = {}          Clear the per-tick bid cache.
  │
  ├─ 5. allocator.allocate(model)   ◄── THE ALGORITHM (see 5.3)
  │                              Decides who does what; calls
  │                              robot.assign(task_id) for new pairings.
  │
  ├─ 6. agents.shuffle_do("step")   ◄── THE PHYSICS (see 5.4)
  │                              Every robot and TaskMarker executes one
  │                              tick, in random order so no robot has a
  │                              permanent turn-order advantage.
  │
  └─ 7. collect metrics; stop if tasks.all_done
```

### 5.3 Inside `allocate()` — the auction

This is the one stage that differs between algorithms. All four
implement the same `allocate(model)` interface, so stages 1–4 and 6–7
above never change.

**Greedy** — no negotiation at all:

```
for each idle robot (shuffled):
    price every pending task with bid_value()
    try them cheapest-first until assign() succeeds
```

**CBBA** — many rounds of gossip inside a single tick, until agreement:

```
CBBAAllocator.allocate
  ├─ _trigger()?          skip entirely if nothing has changed and the
  │                       fleet already converged
  ├─ per robot: prune(), refresh task_list, clear score memo
  ├─ max_rounds = min(CEILING, max(2, N_min × D))     ← Theorem 1
  └─ repeat up to max_rounds:
        ├─ per robot: createBundle()      phase 1 — bid
        │               └─ _pathScore() → pathScoreCBBA() → leg_cost()
        ├─ per robot: broadcast()         send (y, z, s)
        ├─ comms.flush_and_deliver()
        ├─ per robot: resolveConflicts()  phase 2 — agree
        │               ├─ _CBBADecisionTable()   Table I, per task
        │               ├─ mergeTimestamps()      Eq. 5
        │               └─ _releaseOutbid()       Eq. 6
        └─ STOP EARLY if no robot reported a change  ← converged
  └─ per idle robot: assign() the first still-owned task in its path
```

**CBPAE** — exactly one round per tick, no convergence loop; the
"waiting" happens across ticks instead, via the bidding window:

```
CBPAEAllocator.allocate
  ├─ per robot: syncExecution()      beliefs ← reality
  │             dropIfUnreachable()  give up impossible tasks
  │             dropIfTooCostly()    (off by default)
  │             computeBid()         residual + task cost  (Eq. 7)
  │             broadcast()          exactly 4 tasks, constant size
  ├─ comms.flush_and_deliver()
  ├─ per robot: consensus()          Tables 5 / 6 / 7 → _apply()
  └─ per robot: tryAssign()          commit IF window elapsed AND
                                     message quorum met
```

**MOA-CBBA** — CBBA's round structure, cost-based bids, multiple seats:

```
MOACBBAAllocator.allocate
  ├─ per robot: prune(), task_list ← tasks.seats_open(...)
  └─ repeat up to max_rounds:
        ├─ per robot: createBundle()   ONE task per round (MAX_ADDS)
        │               └─ marginalCost() → pathCost() → leg_cost()
        │                  ×capacityAffinity()
        ├─ per robot: broadcast()      bid table + completion time c_i
        ├─ comms.flush_and_deliver()
        ├─ per robot: resolveConflicts()   newest-stamp-wins merge
        ├─ per robot: releaseOutbid()
        └─ STOP EARLY if nothing changed
  └─ _execute()  en-route switching (off by default), then seat robots
                 subject to the seat cap checked against the registry
```

### 5.4 Inside a robot's `step()` — the execution loop

Independent of which allocator ran. A robot with no task counts an idle
tick and does nothing; otherwise it advances its state machine:

```
                    ┌──────────────────────────────────┐
                    ▼                                  │
  IDLE ──assign()──► TO_TASK ──arrived──► DIG ─────────┤
   ▲                   ▲                   │           │
   │                   │            hopper full or     │
   │                   │            task empty         │
   │                   │                   ▼           │
   │                   └──not done──── TO_DUMP         │
   │                    (back for more)   │            │
   │                                   arrived         │
   │                                      ▼            │
   └────task finished────────────────  UNLOAD ─────────┘
                                    (T_UNLOAD ticks)
```

Per stage, per tick:

| Stage | What runs | Key effects |
|---|---|---|
| `IDLE` | nothing | `idle_ticks += 1` |
| `TO_TASK` | `_advance_along_path()` | move ≤ v_max cells; `_spend_move()` charges ALPHA + climb energy. Blocked → `_wait_blocked()` → `_reroute()` after 2 ticks, release task after 20 |
| `DIG` | `_dig_tick()` | remove `min(rho/H, remaining, C - payload)` volume; charge BETA |
| `TO_DUMP` | `_advance_along_path()` | as TO_TASK, but climbing costs ×1.6 (loaded) |
| `UNLOAD` | `_unload_tick()` | count down T_UNLOAD, then empty hopper into `soil_delivered` |

### 5.5 Termination

The run ends when `TaskRegistry.all_done` is true — every task both dug
out **and** stamped with a `completed_tick`. The stamp is written by
`robot._stamp_if_complete` only when the *last* robot leaves a task,
which guarantees the material actually reached a dump site rather than
merely leaving the ground. That distinction is why a run cannot end
while a robot is still mid-haul, and why the reported makespan includes
the final delivery.

### 5.6 Data flow summary

Which module asks which for what, at a glance:

```
  allocator ──"what would this cost?"──► bidding.leg_cost
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                    pathfinding.astar   costs.tau_ij    robot.known_blocked
                    (real distances)    costs.energy_ij  (what THIS robot
                                        (the formulas)    believes)
                              │
  allocator ──"take it"──► robot.assign ──► Task.add_assignee
                              │
  model.step ──"execute"──► robot.step ──► _dig_tick / _advance_along_path
                              │                    │
                              │                    ▼
                              │              _spend / _spend_move
                              │              (charges the SAME terms
                              │               costs.energy_ij predicted)
                              ▼
                        DataCollector ──► metrics / CSV / GUI plots
```

---

## 6. Module map

```
excavsim/
  terrain.py      Terrain types + physical constants (hardness, energy costs).
  costs.py        Closed-form cost formulas: RobotSpec, tau_ij, energy_ij, objective.
  fleet.py        Named robot classes (small/medium/large) and fleet assembly.
  tasks.py        Task / TaskRegistry — the excavation jobs and their state.
  dynamics.py     Phase 4: hazards, wandering obstacles, weather.
  pathfinding.py  A* pathfinding + "nearest workable cell next to a target".
  comms.py        Inter-robot messaging: range, packet loss, latency, bandwidth.
  bidding.py      leg_cost() and friends — turns (robot, task) into a priced bid.
  robot.py        ExcavatorRobot: the per-tick drive/dig/dump/unload execution loop.
  allocation.py   Allocator base class, GreedyAllocator, CBBA.
  CBPAE.py        CBPAE allocator.
  MOACBBA.py      MOA-CBBA allocator (this project's extension).
  model.py        ExcavationModel: builds the world, owns the tick loop, collects metrics.
  debug.py        Read-only diagnostics: event log, invariant checks, bid-vs-actual accounting.
  __init__.py     Re-exports the common public API.

app.py            Interactive browser dashboard (Mesa SolaraViz + matplotlib).
run_single.py     One seeded run, metrics printed at the end. Best starting point.
run_batch.py      Mesa's batch_run helper over a small parameter grid.
batch30.py        Matched-seed comparison of all three real allocators; writes a CSV.
tests/            pytest validation suite (formulas vs. hand-computed values, etc.).
```

**Suggested reading order** if you're new to the code: `terrain.py` →
`costs.py` → `tasks.py` → `fleet.py` → `bidding.py` → `robot.py` →
`model.py` → `allocation.py` → `CBPAE.py` / `MOACBBA.py`. That roughly
follows "what exists" → "what it costs to act" → "how one robot acts" →
"how the world ticks" → "who decides what to do", which mirrors how the
simulation actually executes.

---

## 7. Running it

```bash
# Set up (once)
pip install mesa pytest pandas "mesa[viz]"

# Validate the cost model and a full run against hand-computed values
python -m pytest tests/ -v

# One seeded run with a metrics printout
python run_single.py

# A small parameter sweep via Mesa's batch_run (-> batch_results.csv)
python run_batch.py

# The "real" experiment: 30 matched seeds, all three algorithms head-to-head
python batch30.py                 # or: --seeds 50, --no-dynamics, etc.

# Interactive dashboard (http://localhost:8765)
solara run app.py
```

### The dashboard (`app.py`)
Three tabs: an **operations view** (live map, robot/task tables), a
**time-series** view (tasks completed / total energy / idle ratio over
time), and a **debug** tab (invariant checks, allocator internals,
bid-vs-actual cost accounting, event log). On the map, colour marks a
robot's current stage (grey = idle, orange = travelling to dig,
red = digging, blue = hauling to dump, purple = unloading); optional
overlays show each robot's planned route, its target cells, its sensor
range, and who can currently hear whom over the comm network. Sliders
control fleet size, task count, terrain mix, which allocator is active,
and the Phase 4 dynamics (hazards/obstacles/weather).

### Diagnostics (`excavsim/debug.py`)
A read-only add-on (it changes no simulation behaviour and consumes no
randomness) that can be attached to any model to catch three failure
modes that don't otherwise look different from "the simulation is just
running slowly": a task reserved by a robot that's no longer working
it, a robot whose planned route came back empty and is stuck, or two
robots disagreeing about who won the same task. Usage:

```python
from excavsim.debug import attach
mon = attach(model)     # idempotent — safe to call repeatedly
mon.summary()           # headline numbers
```

or headless: `python -m excavsim.debug --ticks 300 --allocator cbba --seed 42`.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Tick** | One discrete simulation time step. |
| **Makespan** | The tick at which the *last* task finishes — the total time to finish all the work. The primary "how fast" metric. |
| **Bid / leg cost** | The predicted (ticks, energy) cost of a robot taking a specific task, computed ahead of time. |
| **Allocator** | The algorithm deciding which robot does which task (`greedy`/`cbba`/`cbpae`/`moa-cbba`). |
| **Bundle** | A robot's private, ordered wish-list of tasks it's currently claiming (CBBA/MOA-CBBA). |
| **Consensus** | The process by which robots reconcile disagreeing beliefs about who won which task, through gossip and a fixed rulebook, without a central coordinator. |
| **Stage** | A robot's current phase of work: `IDLE`, `TO_TASK`, `DIG`, `TO_DUMP`, `UNLOAD` (see `Stage` in `bidding.py`). |
| **RobotSpec** | A robot's fixed physical properties (capacity, speed, dig rate, battery, sensor range). |
| **ST-SR / ST-MR** | Robotics task-allocation shorthand: Single-Task-robot vs. Multi-Robot — i.e., can more than one robot work the same job at once? CBBA/CBPAE are ST-SR; MOA-CBBA can be ST-MR. |
| **DMG** | "Diminishing Marginal Gain" — a mathematical property CBBA's scoring function needs for its convergence guarantee to hold. |
| **A\*** | A shortest-path search algorithm (like the routing behind GPS navigation), used here on the 8-connected grid. |
| **Property layer** | A Mesa grid overlay storing one numeric value per cell (terrain type, elevation, soil volume). |

---

## 9. Known limitations

- `tests/test_costs.py::test_simulation_matches_closed_form` currently
  fails on this branch (the simulated tick count doesn't match the
  closed-form prediction in that specific hand-built scenario). The
  other tests pass. This predates the documentation work in this pass
  and is worth investigating separately — it's exactly the kind of
  regression the "bid should match execution" invariant described in
  Section 3 is meant to catch.
- Head-on deadlocks in a one-cell-wide corridor with no alternate route
  are a documented, rare limitation of the collision-avoidance logic in
  `robot.py`.
- `MOA-CBBA`'s en-route task switching is implemented but disabled by
  default (`ENABLE_SWITCHING = False`) based on measured results — see
  the comment block above it in `MOACBBA.py` for the data.

For the project's phase roadmap (heterogeneity, partial observability,
dynamics, communication) and other design-decision notes, see
[README.md](README.md).

---

# Appendix A — The mathematics

Every formula the simulation uses, with the symbol it uses in code, the
file it lives in, and what it is for. Equation numbers refer to the
project proposal; paper citations are given where a formula is taken
from published work.

## A.1 Notation

| Symbol | Code | Meaning |
|---|---|---|
| *i* | `robot_id` | index of a robot |
| *j* | `task_id` | index of a task |
| *V<sub>j</sub>* | `task.volume` / `task.remaining` | volume of task *j* |
| *l<sub>j</sub>* | `task.cell` | grid cell of task *j* |
| *H(v)* | `HARDNESS[terrain]` | hardness multiplier of the terrain at cell *v* |
| *C<sub>i</sub>* | `spec.capacity` | payload (hopper) capacity of robot *i* |
| *v<sub>max</sub>* | `spec.v_max` | speed, in cells per tick |
| *ρ<sub>i</sub>* | `spec.dig_rate` | dig rate: volume per (hardness × tick) |
| *B<sub>i</sub>* | `spec.battery` | battery capacity |
| *σ<sub>i</sub>* | `spec.sensor_range` | sensing radius |
| *δ<sub>i</sub>* | `spec.drain_scale` | per-robot energy drain multiplier |
| *p\** | `robot.work_cell` | chosen dig position (adjacent to *l<sub>j</sub>*) |
| *q\** | `robot.dump_cell` | chosen unload position (adjacent to a dump block) |
| *d<sup>task</sup>* | `d_task` | path length, start → *p\** |
| *d<sup>dump</sup>* | `d_dump` | path length, *p\** → *q\** |
| *n<sub>ij</sub>* | `n_trips(...)` | number of dump trips |
| *x<sub>ij</sub>* | `task.assignees` | assignment matrix: 1 if robot *i* has task *j* |
| *w<sub>1</sub>, w<sub>2</sub>* | `model.w1`, `model.w2` | objective weights (time, energy) |

Constants, all in [excavsim/terrain.py](excavsim/terrain.py):

| Symbol | Code | Default | Meaning |
|---|---|---|---|
| *α* | `ALPHA` | 0.05 | energy per grid step travelled |
| *β* | `BETA` | 0.20 | energy per dig tick |
| *γ* | `GAMMA` | 0.10 | energy per metre climbed |
| *γ<sub>full</sub>* | `FULL_PAYLOAD_GAMMA` | 1.6 | climb multiplier when loaded |
| *T<sub>unload</sub>* | `T_UNLOAD` | 3 | ticks to unload, per trip |

Hardness values: SOIL 1.0, GRAVEL 1.8, ROCK 3.5, BEDROCK ∞, DUMP_SITE ∞.

## A.2 The objective function — Eq. (1)

The single number every allocator minimises, and the headline result of
every experiment. In [`costs.objective`](excavsim/costs.py), evaluated
by `model.current_objective()`.

$$J(x) \;=\; w_1 \cdot \max_i \Big( \sum_j x_{ij}\,\tau_{ij} \Big) \;+\; w_2 \cdot \sum_{i,j} x_{ij}\,E_{ij}$$

- The **first term is the makespan**: the inner sum is one robot's total
  working time, and the max picks out the robot that finishes last.
- The **second term is total fleet energy**, a plain sum.

**The asymmetry between them is the single most important fact in this
document.** Energy is *separable* — a robot's own contribution is a term
it can compute alone — while makespan is a *max*, a global coupling no
robot can evaluate without knowing everyone else's schedule. That is
precisely why greedy, CBBA and CBPAE all optimise the energy half by
construction and the makespan half only by accident, and it is the gap
MOA-CBBA's mechanism 2 (§A.7) is built to close.

## A.3 Trip count — Eq. (3)

In [`costs.n_trips`](excavsim/costs.py).

$$n_{ij} \;=\; \left\lceil \frac{V_j}{C_i} \right\rceil$$

The hopper holds *C<sub>i</sub>*, so clearing *V<sub>j</sub>* requires
filling and emptying it ⌈*V<sub>j</sub>*/*C<sub>i</sub>*⌉ times. This is
not a modelling convenience — `robot.py` physically drives to a dump
every time the hopper fills, so any other value would break the
bid-equals-execution invariant. The implementation subtracts 10⁻⁹ before
the ceiling so exact multiples don't round up, and floors the result at
1 (even a nearly-empty task needs one trip out).

Note *n<sub>ij</sub>* is a **step function of volume**, which matters
whenever a task is shared: a robot digging half a pile may still make
three trips rather than half of six. This is why
[`bidding.leg_cost`](excavsim/bidding.py) takes a `volume` override
rather than letting callers scale its result.

## A.4 Task duration — Eq. (4)

In [`costs.tau_ij`](excavsim/costs.py). Total ticks for robot *i* to
complete task *j*, start to finish:

$$\tau_{ij} \;=\; \underbrace{\frac{d^{task}}{v_{max}}}_{\text{approach}} \;+\; \underbrace{\frac{V_j \, H}{\rho_i}}_{\text{dig}} \;+\; \underbrace{\frac{(2n_{ij}-1)\, d^{dump}}{v_{max}}}_{\text{hauling}} \;+\; \underbrace{n_{ij}\, T_{unload}}_{\text{unloading}}$$

Term by term:

- **Approach** — distance in grid steps ÷ speed in cells/tick = ticks.
  Every move costs exactly 1 step, diagonals included (§A.10), which is
  what makes this exact rather than approximate.
- **Dig** — *ρ<sub>i</sub>* is volume per *hardness × tick*, so removing
  *V<sub>j</sub>* from terrain of hardness *H* takes *V<sub>j</sub>H/ρ<sub>i</sub>*
  ticks. Rock (H = 3.5) takes 3.5× as long as soil for the same volume.
- **Hauling** — **(2n − 1)**, not 2n: the robot makes *n* trips out but
  only *n−1* returns, because it ends its last trip at the dump.
- **Unloading** — a flat *T<sub>unload</sub>* ticks per trip.

## A.5 Task energy — Eq. (5)

In [`costs.energy_ij`](excavsim/costs.py). Mirrors §A.4 term for term:

$$E_{ij} \;=\; \delta_i \Big[ \underbrace{\alpha\, d^{task} + \gamma\, T\, c_{task}}_{\text{approach}} \;+\; \underbrace{\frac{\beta\, V_j\, H}{\rho_i}}_{\text{dig}} \;+\; \underbrace{\alpha (2n_{ij}-1) d^{dump} + \gamma\, T \big( n_{ij}\, c_{out}\, \gamma_{full} + (n_{ij}-1)\, c_{back} \big)}_{\text{hauling}} \Big]$$

where *T* is the weather traction multiplier and *c<sub>task</sub>*,
*c<sub>out</sub>*, *c<sub>back</sub>* are the **gross positive elevation
gain** along the three legs (§A.6).

Points worth noting:

- **Descent is free.** Only climbs are charged, which is why the climb
  terms use gross positive gain rather than net elevation change. A route
  that ends lower than it starts gets no refund.
- **Climbing loaded costs *γ<sub>full</sub>* = 1.6× more**, so the *n*
  outbound hauls carry the multiplier and the *n−1* empty returns don't.
- ***E<sub>unload</sub>* = 0** by assumption.
- **The dig term follows from Eq. (4):** a dig tick removes
  *dv = ρ<sub>i</sub>/H* volume, so its energy is
  *β · dv · H/ρ<sub>i</sub> = β* — a **constant, independent of terrain
  and robot**. This is what lets `robot._dig_tick` charge a flat `BETA`
  per tick and still match the closed form exactly.
- ***δ<sub>i</sub>* is applied once**, to the whole bracket, exactly as
  `robot._spend` applies it once per charge.

## A.6 Elevation gain along a path

In [`pathfinding.path_climb`](excavsim/pathfinding.py).

$$\text{climb}(P) \;=\; \sum_{(a,b) \in P} \max\big(0,\; h(b) - h(a)\big)$$

Summing *signed* deltas would give net change and under-charge any route
containing a descent. For the return leg of a haul, the same function is
called on the **reversed path** — descents going out are climbs coming
back.

## A.7 CBBA — the reward score, Eq. (11)

In [`allocation.pathScoreCBBA`](excavsim/allocation.py). Choi, Brunet &
How (2009). A robot's reward for executing an ordered path *P*:

$$S(P) \;=\; \sum_{k=1}^{|P|} R \cdot \lambda^{\,c_k}, \qquad c_k \;=\; \sum_{m \le k} \big( w_1 \tau_m + w_2 E_m \big)$$

with *R* = `TASK_REWARD` = 10 and *λ* = `LAMBDA` = 0.99 < 1. Each leg
starts where the previous one dumped, so *order matters* and *S* is a
property of the whole sequence.

The paper discounts by arrival *time*; this implementation discounts by
the **accrued objective** *w<sub>1</sub>τ + w<sub>2</sub>E*, so that the
reward CBBA maximises is a monotone transform of the *J(x)* the
experiment reports. Two properties survive that change, and both are
load-bearing:

1. **Non-negativity.** Every term is *Rλ<sup>c</sup>* > 0, so
   Sec. IV-A's assumption *c<sub>ij</sub>* ≥ 0 holds and marginal gains
   can never go negative.
2. **Diminishing marginal gain (DMG, Eq. 7).** Inserting a task pushes
   every later task further out in *c<sub>k</sub>*, so each existing term
   can only shrink — the Eq. (12) triangle-inequality argument with cost
   in place of time. The more a robot holds, the less another task is
   worth to it, which is what stops one robot hoarding the board.

**Marginal gain and admission** (`createBundle`, Algorithm 3): for each
candidate task *j*, try every insertion position and keep the best:

$$c_{ij} \;=\; \max_{n} \Big[ S(p_i \oplus_n j) - S(p_i) \Big]$$

admitting *j* only if *c<sub>ij</sub>* > 0 **and** *c<sub>ij</sub>* >
*y<sub>ij</sub>* (Eq. 4's indicator *h<sub>ij</sub>*), i.e. it is worth
something *and* beats the best bid this robot believes exists.

**Lemma 4 clamp** (`_clamped_gain`) forces bids to be non-increasing
across the rounds of one auction:

$$c_{ij}(t) \;=\; \min\big( c_{ij}^{raw}(t),\; c_{ij}(t-1) \big)$$

which guarantees convergence for *any* scoring scheme, DMG or not.

**Timestamp consensus, Eq. (5)** (`mergeTimestamps`), on hearing from *k*:

$$s_{ik} \leftarrow \text{now}, \qquad s_{im} \leftarrow \max(s_{im},\, s_{km}) \;\; \forall m \neq k$$

The element-wise max is how news about *distant* robots propagates
through a multi-hop network, and what lets Table I distinguish fresh
information from a stale echo.

**Eq. (6) — releasing an outbid tail** (`_releaseOutbid`): if
*n̄* is the first bundle position this robot has lost, release
*b<sub>i</sub>[n]* for all *n ≥ n̄*. The tail must go because every later
bid was priced as a marginal addition to a schedule that no longer
exists.

**Convergence bound, Theorem 1.** The auction is capped at

$$\text{rounds} \;=\; \min\Big(200,\; \max\big(2,\; N_{min} \cdot D\big)\Big), \qquad N_{min} = \min\big(|T_{open}|,\; |R| \cdot L_t\big)$$

where *L<sub>t</sub>* is the bundle limit and *D* the comm-graph
diameter ([`comms.network_diameter`](excavsim/comms.py), Eq. 19) — the
longest shortest path in the "who can hear whom" graph, computed by BFS
from every robot. Unlimited comm range ⇒ complete graph ⇒ *D* = 1.

## A.8 CBPAE — bidding while executing, Eq. (7)

In [`CBPAE.computeBid`](excavsim/CBPAE.py). Das et al. (2015). The bid
for candidate task *m* while still executing task *p*:

$$v_{n,m} \;=\; \underbrace{\big(w_1 \tau^{res} + w_2 E^{res}\big)}_{\text{residual, shrinks each tick}} \;+\; \underbrace{\big(w_1 \tau_m + w_2 E_m\big)}_{\text{cost of }m}$$

The residual ([`bidding.residual_cost`](excavsim/bidding.py)) is the
work still owed on the current task:

$$V_{out} = V_j^{rem} + \text{payload}, \qquad n_{left} = \lceil V_{out}/C_i \rceil$$
$$\tau^{res} = \frac{d_{work}}{v_{max}} + \frac{V_j^{rem} H}{\rho_i} + \frac{(2n_{left}-1) d^{dump}}{v_{max}} + n_{left} T_{unload}$$

Note *V<sub>out</sub>* includes what is already in the hopper (it still
has to be hauled) while the dig term uses only what's left in the ground.
*d<sub>work</sub>* counts only while still outbound.

**Why this is the whole point:** the residual shrinks monotonically as
execution progresses, so the same robot's bid on the same task
*improves every round*. A robot nearly finished bids close to its raw
task cost; one that just started bids much higher. Bid convention is
**cost-minimising** (lower wins, ∞ = no bid), the mirror of CBBA.

**The commit gate** (`tryAssign`, Sec. 3.7.3) — both conditions required:

$$\text{now} - tb_j \;\ge\; \text{BID\_WINDOW} \quad \wedge \quad \text{msg\_count} \;\ge\; |\text{neighbours}|$$

A fixed window for rivals to contest, *and* a message from every robot
in radio range processed without losing the bid. Together these make it
impossible for two robots to start the same task. The quorum is the
*current* neighbour count, so an isolated robot (quorum 0) can commit
immediately rather than waiting forever.

## A.9 MOA-CBBA — the four mechanisms

In [excavsim/MOACBBA.py](excavsim/MOACBBA.py).

**1. Cost-convention bidding.** Uses `_cheaper_bid` (lower wins) rather
than CBBA's reward maximisation. The motivation is not aesthetic: CBBA's
admission test `gain > 0 AND gain > y_ij` can *reject* a task outright
rather than merely rank it last — measured leaving a reachable task
unclaimed for 62 ticks with the whole fleet idle. A cost-minimiser
always ranks; it never refuses.

**2. Makespan as a marginal cost** (`marginalCost`) — the core idea:

$$\text{before} = \max(t_i,\; c_{others}), \qquad \text{after} = \max(t_i',\; c_{others})$$
$$\text{cost} \;=\; \Big[ w_1\big(\text{after} - \text{before}\big) + w_2\big(E' - E\big) \Big] \cdot A$$

where *c<sub>others</sub>* = max<sub>k≠i</sub> *c<sub>k</sub>* is the
gossiped completion time of every other robot, *t<sub>i</sub>*/*t<sub>i</sub>'*
are this robot's projected completion before/after adding the task, and
*A* is the affinity multiplier from mechanism 3.

The `max` is doing all the work. If a robot has slack — both
*t<sub>i</sub>* and *t<sub>i</sub>'* below *c<sub>others</sub>* — then
before = after and **the makespan term is exactly zero**: extra work is
free, so the robot keeps winning tasks. Once its schedule passes
*c<sub>others</sub>* it becomes the bottleneck and every further task
costs its full duration. Self-limiting, and impossible to express with a
purely local score.

**3. Capacity affinity** (`capacityAffinity`) — big machines prefer big
piles:

$$A \;=\; 1 - \kappa \cdot \mathrm{clamp}\!\left( \frac{C_i}{C_{max}} \cdot \frac{V_j}{V_{max}},\; 0,\; 1 \right) \;\in\; (1-\kappa,\; 1]$$

with *κ* = `CAPACITY_AFFINITY` = 0.25. Both factors are normalised
(against the fleet's largest hopper and the largest open task), so the
discount is dimensionless and bounded by *κ*. This *sharpens* an effect
already present in the physics — *n<sub>ij</sub>* = ⌈*V<sub>j</sub>/C<sub>i</sub>*⌉
already penalises small hoppers on big tasks — rather than inventing a
new one. Applied only to the ranking cost, **never** to `leg_cost`, so
the bid-equals-execution invariant is untouched. *κ* = 0 disables it.

**4. Sharing** (`seats_for`) — how many robots one task is worth:

$$\text{seats}_j \;=\; \max\left(1,\; \min\Big( \text{max\_sharers},\; \left\lfloor \frac{V_j}{\text{min\_share}} \right\rfloor,\; \big|\text{work\_candidates}(l_j)\big| \Big)\right)$$

Three independent caps: **policy** (configured ceiling; 1 reproduces
single-robot-per-task exactly), **economics** (a seat must be worth at
least `min_share` = 1.2 volume, or the drive isn't worth it), and
**physics** (at most 9 cells are adjacent to a target, fewer once
bedrock, dumps, hazards and parked robots are subtracted — you cannot
seat more diggers than there are places to stand).

When *k* robots share a task, each is priced for *V<sub>j</sub>/k* by
passing the share **into** `leg_cost` rather than dividing its output —
necessary because the approach leg is paid in full regardless of how
little is dug, and *n<sub>ij</sub>* is a step function (§A.3).

**Consensus merge** (`resolveConflicts`). Single-winner *y*/*z* cannot
express *k* seats, so state is per *(task, robot)*:
`bids[(j,i)] = (cost, stamp)`. The winners of *j* are the *k* cheapest
live entries. Merge rule per key:

$$\text{adopt theirs} \iff \text{no local entry} \;\vee\; s_k > s_{mine} \;\vee\; \big(s_k = s_{mine} \wedge \text{cost}_k < \text{cost}_{mine}\big)$$

The timestamp is what lets a **release** survive contact with a peer who
hasn't heard it yet — without it the peer echoes the dead claim back and
the robot re-adopts a task it dropped. (This exact bug has been hit
twice in this codebase; min-cost consensus with no clock cannot
distinguish new information from a stale echo.)

**En-route switching** (`_execute`, off by default) requires all three:

$$\text{tick} - \text{switched\_tick} \ge 15 \quad \wedge \quad \text{can\_abandon} \quad \wedge \quad \text{cost}_{new} < \text{cost}_{cur}(1 - 0.20)$$

The lockout and the margin are not redundant: the margin compares two
costs at one instant, but both move every tick as the robot walks, so a
margin alone cannot stop a cycle.

## A.10 Pathfinding

In [excavsim/pathfinding.py](excavsim/pathfinding.py). Standard A* on an
8-connected grid, scoring each cell

$$f(c) = g(c) + h(c)$$

with *g* the exact steps taken and *h* the **Chebyshev** distance to the
nearest goal:

$$h(c) = \min_{g \in \text{goals}} \max\big( |c_x - g_x|,\; |c_y - g_y| \big)$$

Chebyshev is the exact obstacle-free step count on an 8-connected grid
(one diagonal step closes *both* axis gaps at once), so it never
overestimates — it is *admissible*, which is what makes A* return a
genuinely shortest path. Every move costs exactly 1.0, diagonals
included, which is what makes *d/v<sub>max</sub>* in §A.4 exact.

Multi-goal search with `min` over the goal set is used throughout: a
robot works from any cell adjacent to a target, so a single search over
all ≤9 candidates finds the nearest reachable one — one A* call instead
of nine.

**No-diagonal-squeeze rule.** A diagonal step from *(x,y)* to
*(x+dx, y+dy)* is forbidden when **both** *(x+dx, y)* and *(x, y+dy)*
are blocked — otherwise robots thread zero-width gaps and diagonal
bedrock ridges leak. `model._free_space_connected` applies the identical
rule, or it would certify maps as connected that A* considers walled.

## A.11 Movement, sensing and communication

**Fractional speed** (`robot._advance_along_path`). *v<sub>max</sub>*
need not be an integer, so movement is banked:

$$\text{credit} \mathrel{+}= v_{max}; \quad \textbf{while } \text{credit} \ge 1: \text{ step};\ \text{credit} \mathrel{-}= 1$$

Averaged over ticks this yields exactly *v<sub>max</sub>* cells/tick,
which is what makes *d/v<sub>max</sub>* correct. Credit survives
re-planning but is forfeited when blocked.

**Dig rate per tick** (`robot._dig_tick`):

$$dv = \min\left( \frac{\rho_i}{H},\; V_j^{rem},\; C_i - \text{payload} \right)$$

**Sensing radius** (`robot.sensor_radius`): *r = σ<sub>i</sub> ·
sensor_scale(weather)*, with visibility tested as a disc,
*dx² + dy² ≤ r²*. Weather scales: clear 1.0, rain 0.8, fog 0.4,
storm 0.3.

**Obstacle prediction** (`robot.obstacle_halo`). An obstacle moves with
probability *p* = 0.5 and then picks uniformly among ~8 free
neighbours, so for any *one* ring cell:

$$P(\text{occupied next tick}) \approx \frac{p}{|\text{options}|} = \frac{0.5}{8} \approx 6.25\%$$

and its expected displacement after *t* ticks is ≈ √(*pt*) per axis
(0.7 cells after 1 tick, 1.6 after 5). Both numbers justify the tight
constants: `OBSTACLE_TTL` = 3 ticks and `HALO_HORIZON` = 2 cells,
because a one-cell-wide ring is only a meaningful prediction one or two
ticks ahead.

**Comm range** (`comms._in_range`) is **Euclidean**, not Chebyshev —
radio propagates in a circle even though robots move on a grid:

$$\sqrt{(a_x-b_x)^2 + (a_y-b_y)^2} \;\le\; \text{comm\_range}$$

**Terrain generation.** Elevation is fractal Brownian motion — Perlin
noise summed over octaves at doubling frequency and halving amplitude:

$$\text{field} = \sum_{k=0}^{\text{octaves}-1} \text{persistence}^k \cdot \text{noise}\big(\text{lacunarity}^k \cdot x\big)$$

then normalised to [0,1] and scaled by `elevation_scale`. Task volumes
are drawn **log-uniformly** over [*lo*, *hi*]:

$$V_j = lo \cdot \left(\frac{hi}{lo}\right)^{u}, \qquad u \sim \text{Uniform}(0,1)$$

which spreads durations over the 5–10× range needed for allocation to
matter. A uniform draw produced tasks too similar in size for any
makespan-aware allocator to distinguish itself.

## A.12 Conservation invariant

Not from any paper — this project's own correctness check, verified
every tick by [excavsim/debug.py](excavsim/debug.py) and reported per
run by `batch30.py`:

$$\underbrace{\sum_j V_j^{rem}}_{\text{in the ground}} \;+\; \underbrace{\sum_i \text{payload}_i}_{\text{in hoppers}} \;+\; \underbrace{\sum_i \text{delivered}_i}_{\text{at dumps}} \;=\; \text{constant}$$

Digging moves volume from term 1 to term 2; unloading from term 2 to
term 3. Any drift means volume was created or destroyed, and **no other
metric in that run can be trusted**. It is why `robot._release_task`
puts a non-empty hopper's contents back in the ground rather than
letting a dropped task silently delete soil.

## A.13 Bid accuracy

The measured form of the bid-equals-execution invariant, in
`debug.accuracy_summary`:

$$\text{ratio}_\tau = \frac{\text{actual ticks}}{\tau_{ij}}, \qquad \text{ratio}_E = \frac{\text{actual energy}}{E_{ij}}$$

Both should be **exactly 1.0 for a single robot working uncontested** —
which is what `tests/test_costs.py::test_simulation_matches_closed_form`
asserts. Above 1.0 means execution cost more than promised (contention,
re-routes, hazards — expected in a busy fleet). **Below 1.0 is the
dangerous direction**: a bid that over-charged, which distorts every
allocator comparison, and which mean-and-max reporting alone will hide.
Hence min is reported too.
