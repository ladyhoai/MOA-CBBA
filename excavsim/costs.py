"""Closed-form Phase 1 cost model — Eqs. (3), (4), (5) of the proposal.

These functions are what bid computation (CBBA/CBPAE/MOA-CBBA) calls.
The simulation in robot.py applies the same physics incrementally, so a
robot that executes a task in isolation accrues exactly tau_ij ticks and
E_ij energy; tests/test_costs.py asserts this equivalence.

Note on BETA: from Eq. (4), a dig tick removes dv = rho_i / H volume, so
the dig energy per tick is beta * dv * H / rho_i = beta, a constant.
The incremental simulation exploits this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .terrain import ALPHA, BETA, T_UNLOAD


@dataclass(frozen=True)
class RobotSpec:
    """Immutable properties S_i (Phase 1 subset; extend in Phase 2)."""

    capacity: float      # C_i, payload capacity
    v_max: float         # cells per tick
    dig_rate: float      # rho_i, volume/(hardness*tick)
    battery: float       # B_i, total battery capacity
    sensor_range: float  # sigma_i (unused until Phase 3)


def n_trips(volume: float, capacity: float) -> int:
    """Eq. (3): n_ij = ceil(V_j / C_i)."""
    return math.ceil(volume / capacity)


def tau_ij(spec: RobotSpec, volume: float, hardness: float,
           d_task: float, d_dump: float) -> float:
    """Eq. (4): travel + dig + dump trips + unloading, in ticks."""
    n = n_trips(volume, spec.capacity)
    t_task = d_task / spec.v_max
    t_dig = volume * hardness / spec.dig_rate
    t_dump = (2 * n - 1) * d_dump / spec.v_max
    t_unload = n * T_UNLOAD
    return t_task + t_dig + t_dump + t_unload


def energy_ij(spec: RobotSpec, volume: float, hardness: float,
              d_task: float, d_dump: float) -> float:
    """Eq. (5): E_unload = 0 by assumption."""
    n = n_trips(volume, spec.capacity)
    e_task = ALPHA * d_task
    e_dig = BETA * volume * hardness / spec.dig_rate
    e_dump = ALPHA * (2 * n - 1) * d_dump
    return e_task + e_dig + e_dump


def objective(makespan: float, total_energy: float,
              w1: float = 1.0, w2: float = 1.0) -> float:
    """Eq. (1): J(x) = w1 * max_i sum_j x_ij tau_ij + w2 * sum E_ij."""
    return w1 * makespan + w2 * total_energy
