"""Validation tests — the trust layer discussed for the methodology.

1. Eqs. (3)-(5) against hand-computed values.
2. The incremental simulation reproduces the closed-form tau_ij and E_ij
   exactly for a single robot on a single task (no contention), which is
   the property that makes bids meaningful predictions of execution.

Run with:  python -m pytest tests/ -v

New-reader primer: read this file top to bottom as a worked example of
the cost model. Each `test_*_hand_computed` function plugs simple,
easy-to-verify-by-hand numbers into costs.py's formulas and checks the
result against arithmetic done in the comment above it -- a good way to
see exactly what tau_ij/energy_ij compute. The later tests build an
actual ExcavationModel and step it tick-by-tick to prove the live
simulation matches those same formulas, and that a full run finishes
and is reproducible given the same seed.
"""

import math

import pytest

from excavsim import ExcavationModel, RobotSpec, energy_ij, n_trips, tau_ij
from excavsim.robot import Stage
from excavsim.terrain import ALPHA, BETA, T_UNLOAD, Terrain

SPEC = RobotSpec(capacity=1.5, v_max=1.0, dig_rate=0.5,
                 battery=100.0, sensor_range=8.0)


# --------------------------------------------------------------------- #
# Eq. (3)
# --------------------------------------------------------------------- #
def test_n_trips_hand_computed():
    assert n_trips(volume=3.0, capacity=1.5) == 2
    assert n_trips(volume=3.1, capacity=1.5) == 3
    assert n_trips(volume=0.2, capacity=1.5) == 1


# --------------------------------------------------------------------- #
# Eq. (4): V=3, H=1 (soil), d_task=10, d_dump=4, C=1.5, v=1, rho=0.5
#   n = 2
#   T = 10/1 + 3*1/0.5 + (2*2-1)*4/1 + 2*T_UNLOAD = 10 + 6 + 12 + 6 = 34
# --------------------------------------------------------------------- #
def test_tau_hand_computed():
    t = tau_ij(SPEC, volume=3.0, hardness=1.0, d_task=10, d_dump=4)
    assert t == pytest.approx(10 + 6 + 12 + 2 * T_UNLOAD)


# --------------------------------------------------------------------- #
# Eq. (5): same numbers
#   E = ALPHA*10 + BETA*3*1/0.5 + ALPHA*3*4 + 0
# --------------------------------------------------------------------- #
def test_energy_hand_computed():
    e = energy_ij(SPEC, volume=3.0, hardness=1.0, d_task=10, d_dump=4)
    assert e == pytest.approx(ALPHA * 10 + BETA * 6 + ALPHA * 12)


# --------------------------------------------------------------------- #
# Simulation vs closed form on a controlled single-robot scenario.
# --------------------------------------------------------------------- #
def _controlled_model():
    m = ExcavationModel(width=16, height=16, n_robots=1, n_tasks=1,
                        rock_fraction=0.0, gravel_fraction=0.0, seed=42)
    robot = m.robots[0]
    task = m.tasks.all[0]
    # Pin everything so distances are known and terrain is uniform soil.
    robot.move_to(m.grid[(0, 0)])
    task.cell = (5, 0)
    task.volume = task.remaining = 3.0
    m.grid.terrain.data[:, :] = int(Terrain.SOIL)
    m.dump_sites = [(9, 0)]
    m.grid.terrain.data[(9, 0)] = int(Terrain.DUMP_SITE)
    return m, robot, task


def test_simulation_matches_closed_form():
    m, robot, task = _controlled_model()
    d_task, d_dump = 5.0, 4.0  # straight lines, no obstacles
    expected_t = tau_ij(robot.spec, 3.0, 1.0, d_task, d_dump)
    expected_e = energy_ij(robot.spec, 3.0, 1.0, d_task, d_dump)

    robot.assign(task.task_id)
    t0 = m.tick
    while robot.stage is not Stage.IDLE and m.tick < 500:
        m.tick += 1
        robot.step()

    assert task.done
    assert (m.tick - t0) == pytest.approx(expected_t)
    assert robot.energy_used == pytest.approx(expected_e)


# --------------------------------------------------------------------- #
# End-to-end smoke: greedy baseline finishes and metrics are sane.
# --------------------------------------------------------------------- #
def test_greedy_run_completes_and_is_reproducible():
    def run(seed):
        m = ExcavationModel(n_robots=3, n_tasks=6, allocator="greedy",
                            seed=seed)
        m.run(max_ticks=5000)
        return m

    m1, m2 = run(7), run(7)
    assert not m1.tasks.unfinished
    assert m1.current_objective() == m2.current_objective()  # seeded
    df = m1.datacollector.get_model_vars_dataframe()
    assert df["soil_moved"].iloc[-1] == pytest.approx(
        sum(t.volume for t in m1.tasks.all))
    assert math.isfinite(m1.current_objective())
