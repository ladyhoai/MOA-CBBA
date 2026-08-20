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

New-reader primer: "bidding" here means answering the question "if
robot i took task j right now, starting from some position, how many
ticks and how much energy would that cost?" -- that's leg_cost(), the
core function of the file. Everything else builds on it: bid_value()
turns a (time, energy) pair into one comparable number using the
model's w1/w2 weights, residual_cost() accounts for work a robot has
already committed to on its current task, and the two "_better/_cheaper
_bid" functions decide which of two competing bids wins an auction
under each algorithm's own convention (reward-maximising for CBBA,
cost-minimising for CBPAE/MOA-CBBA).
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
    """A robot's execution state machine (see robot.py's step()):
    IDLE (no task) -> TO_TASK (driving to dig site) -> DIG (excavating)
    -> TO_DUMP (hauling a full hopper) -> UNLOAD (emptying at the dump)
    -> back to TO_TASK (if the task isn't finished) or IDLE (if it is).
    """
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

    THE CENTRAL FUNCTION OF THE BIDDING LAYER. Every allocator prices
    every candidate (robot, task) pair through here.

    WHAT IT DOES, in order:
      1. Look up the terrain hardness H at the task cell. Bedrock/dump
         are infinitely hard, so those return an infeasible bid at once.
      2. Plan the APPROACH leg: find p*, the nearest cell the robot can
         dig this task from, and the route to it (nearest_work_path).
         d_task is that route's length in steps.
      3. Plan the HAUL leg: from p*, find q*, the nearest cell it can
         unload at, and the route (model.dump_work_path). d_dump is that
         length.
      4. Feed H, the volume, d_task and d_dump into the closed-form
         formulas costs.tau_ij and costs.energy_ij, adding the elevation
         gain of each route (path_climb) and the current weather traction
         multiplier.

    MATH: see costs.tau_ij / costs.energy_ij for the formulas. The three
    climb arguments correspond to the three legs of the round trip:
    approach (empty), haul out (loaded, so charged FULL_PAYLOAD_GAMMA),
    and return (empty, walked n-1 times) -- the return climb is the
    outbound route reversed, since its descents are climbs coming back.

    RETURNS: (tau, E, q*) -- predicted ticks, predicted energy, and the
    chosen unload cell. q* is None (with tau = E = +inf) when the task or
    a dump site is unreachable, which is every caller's "infeasible" test.

    KEY DETAIL -- routes are planned on robot.known_blocked(), i.e. what
    THIS robot has actually sensed, not the model's ground truth. A bid
    must be priced on the same information the robot will execute
    against, or bids stop predicting execution.

    `volume` overrides task.remaining, which is what lets an allocator
    price a SHARE of a task. Scaling the returned totals afterwards is
    not equivalent and gets the answer wrong in two ways: the approach
    leg (d_task / v_max, ALPHA * d_task) is paid in full no matter how
    little the robot digs, and n_ij = ceil(V/C) is a step function, so a
    robot digging half the pile may make three trips rather than half of
    six. Passing the share in gets both right for free.

    CACHING: results are memoised in model._leg_cache, keyed by
    (robot, task, start, volume). Bundle construction re-prices the same
    triple once per candidate insertion position, which without the cache
    costs O(n^3) A* searches per auction round. The cache is cleared at
    the top of every tick by model.step, and robots never move during an
    auction, so a cached entry cannot go stale within a round.

    CALLED BY: bid_value (below), allocation.pathScoreCBBA,
    CBPAE.biddableTasks / dropIfUnreachable / dropIfTooCostly,
    MOACBBAAgent.pathCost, MOACBBAAllocator._execute, and debug.py's
    bid-vs-actual accounting.
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

    # The ROBOT's map, not the model's: a bid must be priced on the same
    # information the robot will actually execute against.
    known = robot.known_blocked()
    dig = nearest_work_path(start, [coord], model.grid.width,
                            model.grid.height, known)
    if dig is None:
        if cache is not None:
            cache[key] = (INF, INF, None)
        return INF, INF, None
    p_star, d_task, path_task = dig

    dump = model.dump_work_path(p_star, blocked=known)
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

    "If this robot carried on and finished what it is already doing, how
    much more time and energy would that take?" Zero for an idle robot.

    MATH -- the same shape as costs.tau_ij / energy_ij, but every term
    counts only what is LEFT:

        V_out   = task.remaining + robot.payload   (still to haul away)
        n_left  = ceil(V_out / C_i)
        tau     = d_to_work/v_max                  (only while outbound)
                + task.remaining * H / rho_i       (dig what's left)
                + (2*n_left - 1) * d_dump/v_max
                + n_left * T_UNLOAD
        E       = (ALPHA*(d_to_work + (2*n_left-1)*d_dump)
                 + BETA * task.remaining * H / rho_i) * drain_scale

    Note V_out includes what is already in the hopper (it still has to be
    driven to a dump) while the DIG terms use task.remaining alone (soil
    in the hopper has already been dug). d_to_work is counted only in
    stage TO_TASK -- once digging has started the approach is spent.
    Elevation is omitted here, unlike leg_cost: this is a residual
    estimate used for ranking, not a bid that must match execution.

    This is the w_j,p term of Das et al. Eq. (7): when a robot bids on
    its next task while executing another, the unfinished part of the
    current task is part of the effort. Without it, CBPAE's bids are
    static and the "multi-round dynamic bidding with improved bids in
    each round" that the paper's whole argument rests on does nothing.

    Deliberately a function of WORK REMAINING rather than position, so
    it is monotonically non-increasing as execution progresses -- which
    is the property the bidding window in Sec. 3.7.3 relies on.

    CALLED BY: CBPAEAgent.computeBid (added to every candidate bid, so
    bids improve as the current task nears completion) and
    MOACBBAAgent.pathCost (as the starting offset of a path's cost).
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
                  robot.known_blocked())
        d_to_work = float(len(p) - 1) if p else 0.0

    d_dump = 0.0
    if robot.work_cell is not None and robot.dump_cell is not None:
        p = astar(robot.work_cell, robot.dump_cell, model.grid.width,
                  model.grid.height, robot.known_blocked())
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
    """Collapse a (time, energy) leg cost into ONE comparable number.

    MATH:  bid = w1 * tau_ij  +  w2 * E_ij

    the same weighted combination as the global objective J(x)
    (costs.objective), applied to a single task rather than a whole run.
    That correspondence is deliberate: a robot minimising this locally is
    minimising a component of the number the experiment actually reports.

    Cost convention -- LOWER IS BETTER, and +inf means "cannot do it".

    RETURNS (value, q*), passing the unload cell through from leg_cost.

    CALLED BY: GreedyAllocator.allocate. (CBPAE and MOA-CBBA call
    leg_cost directly, because they add further terms -- residual cost,
    makespan impact, capacity affinity -- before combining.)
    """
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

    DECISION ORDER:
      1. A bid with no owner (agent None) never wins.
      2. Against an UNOWNED task, any strictly positive bid wins.
      3. Otherwise compare values with an EPS tolerance, so
         floating-point noise is not treated as a real difference.
      4. On a genuine tie, the LOWER robot id wins.

    Step 4 is not cosmetic. Sec. III-B requires a systematic tie-break:
    without one, two robots computing identical bids would each keep
    believing they won, neither would yield, and consensus would never
    converge. Every robot applies the same rule to the same numbers, so
    they all reach the same verdict independently.

    CALLED BY: CBBAAgent._CBBADecisionTable (its `better()` helper),
    which is Table I's bid comparison.
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

    The MIRROR IMAGE of _better_bid: same structure, opposite direction.
    Das et al. Sec. 3.1: "a lower bid value corresponds to a higher
    bid", so cheaper wins and NOCOST (+inf) is the "no bid" sentinel
    where CBBA uses 0.0.

    DECISION ORDER: an unowned or infinite-cost bid loses; a real bid
    beats an unowned/infinite one; otherwise cheaper wins by more than
    EPS; exact ties break on the lower robot index -- matching the
    (b_k = b_n) and (a_k < n) rows of their Table 5, and serving the
    same anti-livelock purpose described in _better_bid.

    CALLED BY: CBPAEAgent.computeBid, to decide whether this robot can
    beat the incumbent bid on a task it does not currently hold.
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