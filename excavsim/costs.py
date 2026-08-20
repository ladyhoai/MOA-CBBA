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

New-reader primer: think of this file as a pure calculator. Given "how
big is the job" (volume, hardness) and "how capable is the robot"
(RobotSpec) and "how far does it have to travel" (distances passed in by
the caller), these functions compute two numbers ahead of time: how many
simulated ticks the job will take (tau_ij) and how much battery it will
cost (energy_ij). Nothing here moves a robot or touches the grid -- that
happens tick-by-tick in robot.py. The two are meant to agree exactly
when only one robot is working alone (see the module-level INVARIANT
note above), which is what lets an allocator "bid" on a task using these
formulas and trust the bid is realistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .terrain import ALPHA, BETA, T_UNLOAD, GAMMA, FULL_PAYLOAD_GAMMA


@dataclass(frozen=True)
class RobotSpec:
    """Immutable properties S_i (Phase 1 subset; extended in Phase 2).

    A @dataclass auto-generates __init__/__repr__/__eq__ from the
    fields below; `frozen=True` makes instances read-only after
    creation, matching the "S_i is fixed for the robot's lifetime,
    only M_i (mutable state, tracked on ExcavatorRobot) changes"
    split described in robot.py."""

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
    """How many round trips to the dump a robot needs to clear `volume`.

    MATH -- Eq. (3):

        n_ij = ceil(V_j / C_i)

    where V_j is the task volume and C_i the robot's hopper capacity.
    The hopper holds C_i, so a pile of V_j needs the hopper filled and
    emptied ceil(V_j / C_i) times. The `- 1e-9` absorbs floating-point
    error so that an exact multiple (e.g. V=3.0, C=1.5) gives 2 rather
    than 3; `max(1, ...)` guarantees at least one trip even for a
    near-empty task, since the robot must still drive out and unload.

    CALLED BY: tau_ij and energy_ij (below, both use n to price the
    haul legs), bidding.residual_cost, and tests/test_costs.py.

    Restored. robot.py physically drives to the dump every time the
    hopper fills, so this is not a modelling choice — it is what the
    simulation does. Pinning it to 1 was the single largest source of
    bid/execution divergence.
    """
    return max(1, math.ceil(volume / capacity - 1e-9))


def tau_ij(spec: RobotSpec, volume: float, hardness: float,
           d_task: float, d_dump: float) -> float:
    """Total TIME in ticks for robot i to complete task j, start to finish.

    MATH -- Eq. (4), the sum of four terms:

        tau_ij = d_task/v_max              (drive to the dig site)
               + V_j * H / rho_i           (dig the whole volume)
               + (2n - 1) * d_dump/v_max   (n hauls out, n-1 returns)
               + n * T_UNLOAD              (fixed unload time per trip)

    Term by term:
      - travel: distance in grid steps / speed in cells-per-tick = ticks.
      - dig: dig_rate rho_i is volume per (hardness x tick), so removing
        V_j from terrain of hardness H takes V_j * H / rho_i ticks.
        Harder ground => more ticks for the same volume.
      - haul: the robot ends AT the dump on its final trip and does not
        drive back, hence (2n - 1) one-way legs rather than 2n.
      - unload: T_UNLOAD ticks per trip, a fixed constant (terrain.py).

    CALLED BY: bidding.leg_cost (which supplies real A*-measured
    distances), and tests/test_costs.py for hand-computed validation.

    See also energy_ij, its energy-side twin with matching terms.
    """
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
    """Total ENERGY for robot i to complete task j, start to finish.

    MATH -- Eq. (5). Three groups of terms, all scaled at the end by the
    robot's drain multiplier:

        E_approach = ALPHA*d_task  + GAMMA*traction*climb_to_task
        E_dig      = BETA * V_j * H / rho_i
        E_haul     = ALPHA*(2n-1)*d_dump
                   + GAMMA*traction*( n*climb_to_dump*FULL_PAYLOAD_GAMMA
                                    + (n-1)*climb_from_dump )
        E_ij       = (E_approach + E_dig + E_haul) * drain_scale

    Where the pieces come from:
      - ALPHA is energy per grid step, so ALPHA * distance is flat-ground
        travel cost. Same (2n - 1) leg count as in tau_ij.
      - GAMMA is energy per metre of elevation GAINED. Descent is free,
        which is why the climb arguments are gross positive gain per leg
        (computed by pathfinding.path_climb) rather than net change.
      - Climbing while loaded costs FULL_PAYLOAD_GAMMA (1.6x) more, so
        the n outbound (loaded) hauls carry that multiplier and the
        n-1 empty returns do not.
      - `traction` is the weather multiplier (dynamics.traction_scale):
        rain/storm make climbing more expensive.
      - E_dig follows from the dig time in tau_ij: a dig tick costs BETA,
        and digging takes V*H/rho ticks, so E_dig = BETA*V*H/rho. See the
        BETA note in the module docstring above.
      - E_unload = 0 by assumption.

    Climb arguments are GROSS POSITIVE elevation gain along each leg
    (descent is free), which is what the simulation charges:
      - climb_to_task   : start -> p*, robot empty
      - climb_to_dump   : p* -> q*, robot loaded  (x FULL_PAYLOAD_GAMMA)
      - climb_from_dump : q* -> p*, robot empty   (the n-1 return legs)

    CALLED BY: bidding.leg_cost, and tests/test_costs.py.

    MIRRORED BY: robot._spend_move (the ALPHA/GAMMA terms, charged per
    step taken) and robot._dig_tick (the BETA term, charged per dig
    tick). Those two must stay term-for-term consistent with this
    function or bids stop predicting execution -- see the INVARIANT note
    in the module docstring.
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
    """The single score a whole simulation run is judged by (lower is
    better).

    MATH -- Eq. (1):

        J(x) = w1 * max_i( sum_j x_ij * tau_ij )  +  w2 * sum_ij E_ij
             = w1 * makespan                      +  w2 * total_energy

    x_ij is the binary assignment matrix (1 if robot i does task j). The
    inner sum is one robot's total working time; the max over robots is
    therefore when the LAST robot finishes -- the makespan. The second
    term is simply every robot's energy added up. w1/w2 trade the two
    objectives off against each other, and are set per model run
    (ExcavationModel(w1=..., w2=...)).

    Note the asymmetry that motivates MOA-CBBA: energy is a plain SUM
    (each robot's own cost is separable, so a robot can price its own
    contribution locally) while makespan is a MAX (a global coupling no
    robot can evaluate alone). See MOACBBA.py mechanism 2.

    CALLED BY: model.current_objective, which supplies the live makespan
    and summed energy. Reported by run_single.py, batch30.py and the GUI.
    """
    return w1 * makespan + w2 * total_energy