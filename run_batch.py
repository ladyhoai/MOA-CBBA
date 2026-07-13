"""Headless batch experiments — the pattern for the three-way
MOA-CBBA vs CBPAE vs CBBA comparison in every phase.

Once the other allocators exist, extend the "allocator" list below;
nothing else changes. Results land in a tidy DataFrame keyed by
(allocator, seed/iteration), ready for pandas/matplotlib analysis.

    python run_batch.py
"""

import pandas as pd
from mesa.batchrunner import batch_run

from excavsim.model import ExcavationModel

params = {
    "allocator": ["greedy"],          # + "cbba", "cbpae", "moa-cbba"
    "n_robots": [2, 4],
    "n_tasks": [8],
    "width": 32,
    "height": 32,
}

results = batch_run(
    ExcavationModel,
    parameters=params,
    rng=list(range(5)),    # explicit seeds -> fully reproducible sweep
    max_steps=5000,
    data_collection_period=-1,   # collect only the final state
    number_processes=1,          # set None to use all cores
    display_progress=True,
)

df = pd.DataFrame(results)
cols = ["allocator", "n_robots", "iteration",
        "tasks_done", "total_energy", "mean_idle_ratio", "tick"]
summary = (df[cols].drop_duplicates()
           .groupby(["allocator", "n_robots"])
           .agg(mean_energy=("total_energy", "mean"),
                std_energy=("total_energy", "std"),
                mean_makespan=("tick", "mean")))
print(summary)
df.to_csv("batch_results.csv", index=False)
print("full results -> batch_results.csv")
