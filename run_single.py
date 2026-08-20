"""Single seeded run with per-tick metrics printed at the end.

    python run_single.py

The simplest possible way to use this package: build one
ExcavationModel, run it to completion, print the results. Good starting
point for understanding the model's public API before reading the
GUI (app.py) or the batch runners (run_batch.py / batch30.py).
"""

from excavsim import ExcavationModel

# width/height: grid size in cells. n_robots/n_tasks: fleet size and
# number of dig sites. allocator picks which task-assignment algorithm
# runs ("greedy" | "cbba" | "cbpae" | "moa-cbba"). seed makes the random
# terrain/task/robot placement reproducible.
m = ExcavationModel(width=32, height=32, n_robots=4, n_tasks=10,
                    allocator="greedy", seed=1)
m.run(max_ticks=10_000)   # advances model.step() until every task is done

print(f"finished at tick {m.tick}, makespan={m.makespan}")
print(f"J(x) = {m.current_objective():.2f}  "
      f"(w1={m.w1}, w2={m.w2})")
print(f"total energy = {sum(r.energy_used for r in m.robots):.2f}")
for i, r in enumerate(m.robots):
    print(f"  robot {i}: tasks={r.tasks_completed} "
          f"energy={r.energy_used:.2f} idle={r.idle_ticks}")

df = m.datacollector.get_model_vars_dataframe()
print("\nlast 5 ticks of model metrics:")
print(df.tail())
