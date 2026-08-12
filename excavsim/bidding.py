"""Shared bid machinery: leg costs, comparators, execution stages.

SINGLE SOURCE OF TRUTH. allocation.py and CBPAE.py used to carry their
own byte-identical copies of leg_cost / bid_value / _better_bid; CBPAE
imported one and CBBA used the other, so any fix landed in one allocator
and not the other. Both now import from here.

Two bid conventions live side by side, deliberately:
  - CBBA (Choi et al. 2009) maximises a non-negative reward. Higher
    wins, "no bid" is 0.0 (their y_i is in R_+^Nt, initialised to zero).
  - CBPAE (Das et al. 2015, Sec. 3.1) minimises effort. Lower wins,
    "no bid" is +inf. Their Tables 5 and 6 are written as b_k < b_n, so
    transcribing them faithfully requires the cost convention.
Mixing them up silently inverts every auction, hence two comparators
with loud names rather than one shared one.
"""

from __future__ import annotations

from enum import Enum, auto

from .costs import energy_ij, n_trips, tau_ij
from .pathfinding import astar, nearest_work_path, path_climb
from .terrain import ALPHA, BETA, GAMMA, HARDNESS, T_UNLOAD, Terrain

INF = float("inf")
EPS = 1e-9

NOBID = 0.0     # CBBA reward convention: y_ij starts at zero
NOCOST = INF    # CBPAE cost convention: no bid == infinitely expensive


class Stage(Enum):
    IDLE = auto()
    TO_TASK = auto()
    DIG = auto()
    TO_DUMP = auto()
    UNLOAD = auto()


# ------------------------------------------------------------------ #
# leg costs
# ------------------------------------------------------------------ #
def leg_cost(model, robot, task, startPos=None,
             volume=None) -> tuple[float, float, object]:
    """(tau_ij, E_ij, q*) for robot i taking task j, starting from
    `startPos` (its current cell if None).

    `volume` overrides task.remaining, which is what lets an allocator
    price a SHARE of a task. Scaling the returned totals afterwards is
    not equivalent and gets the answer wrong in two ways: the approach
    leg (d_task / v_max, ALPHA * d_task) is paid in full no matter how
    little the robot digs, and n_ij = ceil(V/C) is a step function, so a
    robot digging half the pile may make three trips rather than half of
    six. Passing the share in gets both right for free.

    Returns (inf, inf, None) when the task or a dump site is unreachable.
    """
    coord = task.cell
    terrain = Terrain(int(model.grid.terrain.data[coord]))
    h = HARDNESS[terrain]
    if h == INF:                      # bedrock / dump: not diggable
        return INF, INF, None
    start = robot.cell.coordinate if startPos is None else startPos

    # Bundle construction evaluates the same (robot, task, start) triple
    # once per candidate insertion position, so an auction round over n
    # tasks costs O(n^3) A* searches without this. Cleared every tick by
    # model.step(); robots never move inside a round.
    vol = task.remaining if volume is None else max(0.0, float(volume))

    cache = getattr(model, "_leg_cache", None)
    key = (robot.robot_id, task.task_id, start, round(vol, 9))
    if cache is not None and key in cache:
        return cache[key]

    dig = nearest_work_path(start, [coord], model.grid.width,
                            model.grid.height, model.blocked_cells())
    if dig is None:
        if cache is not None:
            cache[key] = (INF, INF, None)
        return INF, INF, None
    p_star, d_task, path_task = dig

    dump = model.dump_work_path(p_star)
    if dump is None:
        if cache is not None:
            cache[key] = (INF, INF, None)
        return INF, INF, None
    q_star, d_dump, path_dump = dump

    elev = model.grid.elevation.data
    traction = model.dynamics.traction_scale()

    t = tau_ij(robot.spec, vol, h, d_task, d_dump)
    e = energy_ij(robot.spec, vol, h, d_task, d_dump,
                  climb_to_task=path_climb(path_task, elev),
                  climb_to_dump=path_climb(path_dump, elev),
                  climb_from_dump=path_climb(path_dump[::-1], elev),
                  traction=traction)
    if cache is not None:
        cache[key] = (t, e, q_star)
    return t, e, q_star


def residual_cost(model, robot) -> tuple[float, float]:
    """(tau, E) still owed on the robot's CURRENT task.

    This is the w_j,p term of Das et al. Eq. (7): when a robot bids on
    its next task while executing another, the unfinished part of the
    current task is part of the effort. Without it, CBPAE's bids are
    static and the "multi-round dynamic bidding with improved bids in
    each round" that the paper's whole argument rests on does nothing.

    Deliberately a function of WORK REMAINING rather than position, so
    it is monotonically non-increasing as execution progresses -- which
    is the property the bidding window in Sec. 3.7.3 relies on.
    """
    if robot.task_id is None:
        return 0.0, 0.0
    task = model.tasks.get(robot.task_id)
    spec = robot.spec
    terrain = Terrain(int(model.grid.terrain.data[task.cell]))
    h = HARDNESS[terrain]
    if h == INF:
        return 0.0, 0.0

    vol_out = task.remaining + robot.payload      # still to be hauled away
    if vol_out <= EPS:
        return 0.0, 0.0
    n_left = n_trips(vol_out, spec.capacity)

    # approach leg, only while still outbound
    d_to_work = 0.0
    if robot.stage is Stage.TO_TASK and robot.work_cell is not None:
        p = astar(robot.cell.coordinate, robot.work_cell,
                  model.grid.width, model.grid.height,
                  model.blocked_cells())
        d_to_work = float(len(p) - 1) if p else 0.0

    d_dump = 0.0
    if robot.work_cell is not None and robot.dump_cell is not None:
        p = astar(robot.work_cell, robot.dump_cell, model.grid.width,
                  model.grid.height, model.blocked_cells())
        d_dump = float(len(p) - 1) if p else 0.0

    t_dig = task.remaining * h / spec.dig_rate
    tau = (d_to_work / spec.v_max + t_dig
           + (2 * n_left - 1) * d_dump / spec.v_max
           + n_left * T_UNLOAD)
    e = ((ALPHA * (d_to_work + (2 * n_left - 1) * d_dump)
          + BETA * task.remaining * h / spec.dig_rate) * spec.drain_scale)
    return tau, e


def bid_value(model, robot, task, w1: float = 1.0, w2: float = 1.0,
              startPos=None) -> tuple[float, object]:
    """Marginal cost w1*tau_ij + w2*E_ij (cost convention: lower is
    better). Used by the greedy baseline and by CBPAE."""
    t, e, q = leg_cost(model, robot, task, startPos)
    if q is None:
        return INF, None
    return w1 * t + w2 * e, q


# ------------------------------------------------------------------ #
# comparators
# ------------------------------------------------------------------ #
def _better_bid(bid_a, agent_a, bid_b, agent_b) -> bool:
    """CBBA: true if (bid_a by agent_a) beats (bid_b by agent_b).

    Reward maximisation (Choi et al.'s convention): the HIGHER bid wins.
    Exact ties break on the lower robot id -- Sec. III-B requires a
    systematic tie-break, otherwise two robots computing identical bids
    both keep believing they won and consensus never converges.
    """
    if agent_a is None:
        return False
    if agent_b is None:
        return bid_a > NOBID
    if bid_a > bid_b + EPS:
        return True
    if bid_b > bid_a + EPS:
        return False
    return agent_a < agent_b


def _cheaper_bid(bid_a, agent_a, bid_b, agent_b) -> bool:
    """CBPAE: true if (bid_a by agent_a) beats (bid_b by agent_b).

    Das et al. Sec. 3.1: "a lower bid value corresponds to a higher
    bid". Ties break on the lower robot index, matching the
    (b_k = b_n) and (a_k < n) rows of their Table 5.
    """
    if agent_a is None or bid_a >= NOCOST:
        return False
    if agent_b is None or bid_b >= NOCOST:
        return True
    if bid_a < bid_b - EPS:
        return True
    if bid_b < bid_a - EPS:
        return False
    return agent_a < agent_b