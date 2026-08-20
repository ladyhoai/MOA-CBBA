"""Matched-seed comparison of the three allocators.

    python batch30.py                 # 30 seeds, default config
    python batch30.py --seeds 50      # more seeds
    python batch30.py --no-dynamics   # hazards/obstacles/weather off

Prints one line per seed in the format you asked for, then a summary,
and writes batch30.csv for plotting.

WHY BOTH MEAN AND MEDIAN. Makespan here is heavily right-skewed: a run
that hits a bad geometry can take three times the typical value, and one
such run drags a mean far more than it should. Earlier in this project a
four-seed CBBA mean of 490 was produced by three runs near 300 and one at
1078. The median is the number to lead with; the mean is worth printing
only so the gap between them tells you how skewed the sample is.

WHY CBBA'S IDLE IS IN THE CSV. MOA-CBBA's mechanisms all recover idle
robot time, and they cost extra travel to do it. So the expected pattern
is that MOA wins where the baseline leaves robots standing around and
loses where it does not. Plotting `cbba_idle` against `moa - cbba` tests
that directly, and a conditional result ("wins when baseline idle is
high") is worth more in a write-up than an average that hides the
reversal.

New-reader primer: this is the "real" experiment script (run_batch.py is
a smaller Mesa-native example). For each of `--seeds` random seeds it
runs the SAME randomly generated world (terrain, robots, tasks) once per
allocator in ALLOCATORS, so the three algorithms are compared on
identical instances rather than on luck. `run_one()` runs a single
(seed, allocator) pair to completion and returns its metrics as a dict;
`main()` (at the bottom of the file) loops over seeds/allocators, prints
a running summary, and writes every row to batch30.csv.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time

from excavsim.model import ExcavationModel


def resolved_config(cfg: dict) -> dict:
    """Every constructor parameter and the value this batch will use.

    Read from the signature rather than hard-coded, so the printed record
    is the truth even after someone edits a default in model.py.
    """
    import inspect
    out = {}
    for name, p in inspect.signature(ExcavationModel.__init__).parameters.items():
        if name in ("self", "seed", "allocator") or p.default is p.empty:
            continue
        out[name] = cfg.get(name, p.default)
    return out

ALLOCATORS = ("cbba", "cbpae", "moa-cbba")

# ONLY the overrides, matching the model_instance in app.py. Everything
# else falls through to ExcavationModel's own defaults.
#
# Deliberately not a full parameter list. Restating a default here means
# that when the model's default later changes, the batch keeps silently
# using the old value and stops measuring what the dashboard measures.
# That has already happened once in this project: app.py pinned
# fleet_mode="capacity" long after the model had moved to "full", and
# SolaraViz passes every model_params entry to the constructor, so the
# stale copy won. Listing only the overrides makes that impossible.
#
# The full resolved set is read back from the constructor signature and
# printed at the top of every run, so the report still gets an explicit,
# reproducible record.
CONFIG = dict(
    weather_enabled=True,
    weather_change_rate=0.05,
    hazard_rate=0.05,
    hazard_size=2,
    hazard_duration=8,
    obstacle_rate=0.3,
    max_obstacles=10,
)

MAX_TICKS = 8000        # safety stop; a run that hits this is reported


def run_one(seed: int, allocator: str, cfg: dict) -> dict:
    model = ExcavationModel(seed=seed, allocator=allocator, **cfg)
    for _ in range(MAX_TICKS):
        if model.tasks.all_done:
            break
        model.step()

    stamps = [t.completed_tick for t in model.tasks.all
              if t.completed_tick is not None]
    robots = model.robots
    ticks = max(1, model.tick)
    return {
        "makespan": max(stamps) if stamps else float("nan"),
        "energy": sum(r.energy_used for r in robots),
        "objective": model.current_objective(),
        "idle": sum(r.idle_ticks for r in robots) / (ticks * len(robots)),
        "waits": sum(r.wait_ticks for r in robots),
        "steps": sum(r.distance_travelled for r in robots),
        "max_sharers": getattr(model.allocator, "max_sharers_seen", 1),
        # Conservation: soil only moves ground -> hopper -> dump, so this
        # must be 0.00 in every row. A non-zero value means volume was
        # created or destroyed and NOTHING else in the row can be trusted.
        "carried": sum(r.payload for r in robots),
        "delivered": sum(getattr(r, "soil_delivered", 0.0) for r in robots),
        "finished": model.tasks.all_done,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--start", type=int, default=1,
                    help="first seed; seeds are start..start+seeds-1")
    ap.add_argument("--no-dynamics", action="store_true",
                    help="hazards, obstacles and weather all off")
    ap.add_argument("--hazard-rate", type=float, default=None)
    ap.add_argument("--obstacle-rate", type=float, default=None)
    ap.add_argument("--tasks", type=int, default=None)
    ap.add_argument("--robots", type=int, default=None)
    ap.add_argument("--csv", default="batch30.csv")
    args = ap.parse_args(argv)

    cfg = dict(CONFIG)
    if args.no_dynamics:
        cfg.update(hazard_rate=0.0, obstacle_rate=0.0, weather_enabled=False)
    if args.hazard_rate is not None:
        cfg["hazard_rate"] = args.hazard_rate
    if args.obstacle_rate is not None:
        cfg["obstacle_rate"] = args.obstacle_rate
    if args.tasks is not None:
        cfg["n_tasks"] = args.tasks
    if args.robots is not None:
        cfg["n_robots"] = args.robots

    seeds = list(range(args.start, args.start + args.seeds))
    full = resolved_config(cfg)
    print(f"# {len(seeds)} seeds x {len(ALLOCATORS)} allocators, "
          f"seeds {seeds[0]}..{seeds[-1]}")
    print("# overrides : " + ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())))
    print("# resolved  : " + ", ".join(f"{k}={v}" for k, v in sorted(full.items())))
    print()

    results: dict[str, dict[int, dict]] = {a: {} for a in ALLOCATORS}
    rows = []
    t0 = time.time()

    for seed in seeds:
        parts = []
        for alloc in ALLOCATORS:
            r = run_one(seed, alloc, cfg)
            results[alloc][seed] = r
            rows.append({"seed": seed, "allocator": alloc, **r})
            parts.append(f"{alloc}: {r['makespan']:.0f} "
                         f"(e = {r['energy']:.2f})")
            if not r["finished"]:
                parts[-1] += " [NOT FINISHED]"
            if r["carried"] > 1e-6:
                parts[-1] += f" [!! {r['carried']:.2f} SOIL STRANDED]"
        # the line you asked for
        print(f"seed {seed}: " + ", ".join(parts), flush=True)

    print(f"\n# {time.time() - t0:.0f}s total\n")

    # ----------------------------------------------------------------- #
    def col(alloc, key):
        return [results[alloc][s][key] for s in seeds]

    print(f"{'allocator':10}{'makespan median':>17}{'mean':>8}{'sd':>8}"
          f"{'energy med':>12}{'J(x) med':>10}{'idle':>7}")
    for alloc in ALLOCATORS:
        mk = col(alloc, "makespan")
        print(f"{alloc:10}{statistics.median(mk):17.1f}"
              f"{statistics.mean(mk):8.1f}"
              f"{(statistics.stdev(mk) if len(mk) > 1 else 0):8.1f}"
              f"{statistics.median(col(alloc, 'energy')):12.2f}"
              f"{statistics.median(col(alloc, 'objective')):10.1f}"
              f"{statistics.mean(col(alloc, 'idle')):7.3f}")

    # head-to-head: per-seed wins say more than a difference of means
    print()
    for alloc in ALLOCATORS:
        if alloc == "moa-cbba":
            continue
        diffs = [results["moa-cbba"][s]["makespan"] - results[alloc][s]["makespan"]
                 for s in seeds]
        wins = sum(1 for d in diffs if d < 0)
        ties = sum(1 for d in diffs if d == 0)
        print(f"moa-cbba vs {alloc:6}: better on {wins}/{len(seeds)} seeds"
              f"{f' ({ties} tied)' if ties else ''}, "
              f"median diff {statistics.median(diffs):+.1f} ticks")

    # the conditional result: does MOA win where the baseline idles?
    x = [results["cbba"][s]["idle"] for s in seeds]
    y = [results["moa-cbba"][s]["makespan"] - results["cbba"][s]["makespan"]
         for s in seeds]
    if len(seeds) > 2 and statistics.pstdev(x) > 0:
        r = statistics.correlation(x, y)
        print(f"\ncorrelation(cbba idle, moa - cbba makespan) = {r:+.2f}")
        print("  negative => MOA-CBBA gains most where the baseline leaves"
              " robots idle,\n  which is the mechanism it is built on.")

    bad = [r for r in rows if r["carried"] > 1e-6 or not r["finished"]]
    print(f"\nconservation/termination problems: {len(bad)} of {len(rows)} runs")

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())