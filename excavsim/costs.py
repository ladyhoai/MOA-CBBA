"""Closed-form Phase 1 cost model — Eqs. (3), (4), (5) of the proposal.

These functions are what bid computation (CBBA/CBPAE/MOA-CBBA) calls.
The simulation in robot.py applies the same physics incrementally, so a
robot that executes a task in isolation accrues exactly tau_ij ticks and
E_ij energy.

INVARIANT (bid == uncontended execution). Three things used to break it
and are now handled here, because a bid that under-reports cost makes
every allocator comparison meaningless:

  1. n_ij was pinned to 1, so a chunk with V_j > C_i was billed for one
     dump trip while the robot actually made ceil(V_j/C_i) of them. It
     also made capacity heterogeneity invisible to the allocator — with
     n == 1 the payload capacity never appears in any bid, so
     fleet_mode="capacity" could not possibly change an allocation.
  2. drain_scale was applied in robot._spend but not here, so heavy
     machines' bids were low by their drain multiplier (1.4x for large).
  3. climb energy was passed in as NET elevation change while the sim
     charges only POSITIVE gain, and the sim scaled it by weather
     traction. Both are now explicit arguments.

Note on BETA: from Eq. (4), a dig tick removes dv = rho_i / H volume, so
the dig energy per tick is beta * dv * H / rho_i = beta, a constant.
The incremental simulation exploits this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .terrain import ALPHA, BETA, T_UNLOAD, GAMMA, FULL_PAYLOAD_GAMMA


@dataclass(frozen=True)
class RobotSpec:
    """Immutable properties S_i (Phase 1 subset; extended in Phase 2)."""

    capacity: float      # C_i, payload capacity
    v_max: float         # cells per tick
    dig_rate: float      # rho_i, volume/(hardness*tick)
    battery: float       # B_i, total battery capacity
    sensor_range: float  # sigma_i (unused until Phase 3)
    # --- Phase 2 (Table 1, item 2.4) -------------------------------- #
    drain_scale: float = 1.0   # battery drain multiplier; heavy machines
                               # spend more per step travelled and per
                               # dig tick. 1.0 reproduces Phase 1 exactly.
    name: str = "default"      # robot class label, for reporting


def n_trips(volume: float, capacity: float) -> int:
    """Eq. (3): n_ij = ceil(V_j / C_i).

    Restored. robot.py physically drives to the dump every time the
    hopper fills, so this is not a modelling choice — it is what the
    simulation does. Pinning it to 1 was the single largest source of
    bid/execution divergence.
    """
    return max(1, math.ceil(volume / capacity - 1e-9))


def tau_ij(spec: RobotSpec, volume: float, hardness: float,
           d_task: float, d_dump: float) -> float:
    """Eq. (4): travel + dig + dump trips + unloading, in ticks."""
    n = n_trips(volume, spec.capacity)
    t_task = d_task / spec.v_max
    t_dig = volume * hardness / spec.dig_rate
    t_dump = (2 * n - 1) * d_dump / spec.v_max   # n out, n-1 back
    t_unload = n * T_UNLOAD
    return t_task + t_dig + t_dump + t_unload


def energy_ij(spec: RobotSpec, volume: float, hardness: float,
              d_task: float, d_dump: float,
              climb_to_task: float = 0.0,
              climb_to_dump: float = 0.0,
              climb_from_dump: float = 0.0,
              traction: float = 1.0) -> float:
    """Eq. (5), matched term-by-term to robot._spend_move / _dig_tick.

    Climb arguments are GROSS POSITIVE elevation gain along each leg
    (descent is free), which is what the simulation charges:
      - climb_to_task   : start -> p*, robot empty
      - climb_to_dump   : p* -> q*, robot loaded  (x FULL_PAYLOAD_GAMMA)
      - climb_from_dump : q* -> p*, robot empty   (the n-1 return legs)
    E_unload = 0 by assumption.
    """
    n = n_trips(volume, spec.capacity)
    e_task = ALPHA * d_task + GAMMA * traction * climb_to_task
    e_dig = BETA * volume * hardness / spec.dig_rate
    e_dump = (ALPHA * (2 * n - 1) * d_dump
              + GAMMA * traction * (n * climb_to_dump * FULL_PAYLOAD_GAMMA
                                    + (n - 1) * climb_from_dump))
    # Applied once, here, exactly as robot._spend applies it once.
    return (e_task + e_dig + e_dump) * spec.drain_scale


def objective(makespan: float, total_energy: float,
              w1: float = 1.0, w2: float = 1.0) -> float:
    """Eq. (1): J(x) = w1 * max_i sum_j x_ij tau_ij + w2 * sum E_ij."""
    return w1 * makespan + w2 * total_energy