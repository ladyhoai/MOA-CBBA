"""A* over the 8-connected grid, with multi-goal support and
deterministic work-cell selection.

Distances are path lengths in grid steps (each of the 8 moves = 1 step).
Since robots now work *near* target cells rather than on them, planning
targets are small regions: the goal test accepts any cell in `goals`,
and the heuristic is the min octile distance over the goal set.

Determinism: the heap key includes coordinates, so among equally short
paths the same goal cell wins every run. costs.py bids and robot.py
planning both go through nearest_work_cell, which is what keeps the
bid = execution invariant intact in uncontended scenarios.

New-reader primer: `astar()` is a textbook A* pathfinder (like the
routing algorithm behind GPS, but on a grid) -- it finds the shortest
route between two cells while stepping around anything in `blocked`.
Everything else in this file exists to answer a slightly different
question specific to this simulation: a robot doesn't need to stand
exactly ON a task's cell to dig it, it can work from any adjacent cell,
so `work_candidates`/`nearest_work_cell`/`nearest_work_path` find the
CLOSEST cell next to a target that the robot can actually reach and
work from.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

Coord = tuple[int, int]

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Forbid a diagonal step when BOTH of the orthogonal cells it passes
# between are blocked.
#
# On an 8-connected grid a move from (x, y) to (x+1, y+1) touches the
# corner shared by (x+1, y) and (x, y+1). If both of those are solid,
# the robot is squeezing through a zero-width gap between two rocks --
# geometrically impossible for anything with a body, and it makes walls
# leak. _scatter_bedrock draws ridges along (1, 1) and (1, -1) as well
# as the axes, so a diagonal ridge is a STAIRCASE of diagonally-touching
# cells: every step of it had a gap, and a ridge that looks like a solid
# wall on screen stopped nothing.
#
# "Both blocked" is the minimum rule and the one implemented. The
# stricter "either blocked" (no corner-cutting, appropriate for a
# vehicle with real width) would also forbid clipping the outside corner
# of a single rock; it is a larger behavioural change and can make tight
# work cells unreachable, so it is left as a note rather than a default.
NO_DIAGONAL_SQUEEZE = True


def _squeezes(cur: Coord, dx: int, dy: int, blocked: set[Coord]) -> bool:
    """True if stepping (dx, dy) from `cur` cuts between two blocked
    cells. Orthogonal moves never do.

    MATH: a diagonal step from (x, y) to (x+dx, y+dy) passes through the
    corner shared by the two orthogonal neighbours (x+dx, y) and
    (x, y+dy). If BOTH are solid, the move threads a zero-width gap.
    `if not (dx and dy)` short-circuits orthogonal moves, where one of
    dx/dy is 0 and no corner is cut. See the NO_DIAGONAL_SQUEEZE note
    above for why this rule exists.

    CALLED BY: astar (per candidate move) and
    model._free_space_connected -- the flood fill MUST apply the same
    rule as the planner, or it will certify a map as connected that
    robots cannot actually cross.
    """
    if not (dx and dy):
        return False
    return ((cur[0] + dx, cur[1]) in blocked
            and (cur[0], cur[1] + dy) in blocked)


def chebyshev(a: Coord, b: Coord) -> int:
    """Chebyshev ("chessboard king") distance between two cells.

    MATH:  d(a, b) = max(|ax - bx|, |ay - by|)

    This is the exact minimum number of steps on an 8-connected grid
    when nothing is in the way, because one diagonal step reduces BOTH
    axis gaps at once -- so the journey costs whichever axis gap is
    larger, not their sum. That makes it the natural metric here and an
    admissible A* heuristic (it never overestimates).

    CALLED BY: _h (the A* heuristic), robot._dig_tick (is the robot
    still adjacent to its target?), and debug.py's proximity checks.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _h(c: Coord, goals: frozenset[Coord]) -> float:
    """A* heuristic: optimistic steps remaining from `c` to the goal SET.

    MATH:  h(c) = min over g in goals of chebyshev(c, g)

    Taking the MINIMUM over the goal set keeps the heuristic admissible
    for multi-goal search -- reaching the nearest goal cell is the
    cheapest way to satisfy "reach any goal", so this can never
    overestimate the true remaining cost, which is what guarantees A*
    returns a genuinely shortest path.

    CALLED BY: astar, once per cell pushed onto the open heap.
    """
    return min(chebyshev(c, g) for g in goals)


def astar(
    start: Coord,
    goal: Coord | Iterable[Coord],
    width: int,
    height: int,
    blocked: set[Coord] | None = None,
) -> list[Coord] | None:
    """Shortest path from start to ANY cell in `goal` (inclusive).

    Blocked cells are impassable unless they are goals themselves.
    Returns the path, or None. Length in steps is len(path) - 1.

    MATH -- standard A*. Each cell is scored

        f(c) = g(c) + h(c)

    where g(c) is the exact number of steps taken to reach c so far and
    h(c) is the optimistic estimate of steps still to go (`_h`, the min
    Chebyshev distance to the goal set). Cells are expanded from a
    min-heap in increasing f, so the first time a goal is popped its
    path is provably shortest -- this holds precisely because h never
    overestimates (it is "admissible"). Every move costs exactly 1.0,
    including diagonals, which is what makes a path length in STEPS
    directly convertible to ticks via d / v_max in costs.tau_ij.

    The heap key is (f, x, y, tie) rather than just f so ties break
    deterministically on coordinates: two runs with the same inputs
    always return the same path, which is what keeps bids reproducible.

    `came` records, for each visited cell, which neighbour we stepped
    from, so the path is reconstructed backwards from the goal at the
    end. Returns None when the goal set is unreachable, which callers
    treat as an infinite-cost bid.

    CALLED BY: nearest_work_cell / nearest_work_path / astar_distance
    (below), robot._plan_leg (routing a real leg) and
    bidding.residual_cost (measuring remaining distance).
    """
    goals = frozenset([goal] if isinstance(goal, tuple) else goal)
    if not goals:
        return None
    blocked = blocked or set()
    if start in goals:
        return [start]
    # heap key: (f, x, y, tie) -> fully deterministic pop order
    open_heap: list[tuple[float, int, int, int, Coord]] = [
        (_h(start, goals), start[0], start[1], 0, start)] # pyright: ignore[reportArgumentType]
    g: dict[Coord, float] = {start: 0.0}     # cheapest known steps-from-start per cell
    came: dict[Coord, Coord] = {}            # cell -> the cell we reached it from
    tie = 0
    while open_heap:
        _, _, _, _, cur = heapq.heappop(open_heap)
        if cur in goals:
            # Walk `came` back from the goal to `start`, then reverse it
            # into start-to-goal order.
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy in MOORE:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in blocked and nxt not in goals:
                continue
            # Applied even when the destination is a goal: a work cell
            # that can only be reached by passing through a wall is not
            # reachable, and pretending otherwise produces a bid the
            # robot can never execute.
            if NO_DIAGONAL_SQUEEZE and _squeezes(cur, dx, dy, blocked):
                continue
            ng = g[cur] + 1.0
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                tie += 1
                heapq.heappush(open_heap,
                               (ng + _h(nxt, goals), nxt[0], nxt[1], tie, nxt)) # pyright: ignore[reportArgumentType]
    return None

def path_climb(path, elevation) -> float:
    """Total elevation GAINED along the path.

    MATH:  climb = sum over consecutive (a, b) in path of
                   max(0, elevation[b] - elevation[a])

    i.e. add up only the POSITIVE elevation deltas, clamping descents to
    zero. This feeds the GAMMA * climb terms of costs.energy_ij.

    CALLED BY: bidding.leg_cost, three times per bid -- once for the
    approach leg, once for the loaded haul out (path as given), and once
    for the empty return (the same path reversed, since what was a
    descent going out is a climb coming back).

    Only ascents are charged, because that is exactly what
    robot._spend_move does (`if gain > 0.0`). Summing the signed deltas
    -- the old behaviour -- returned the NET change, so any route with a
    descent anywhere in it was bid below its true execution cost, and a
    route that ended lower than it started was billed a climb credit
    that the simulation never gives back.

    For the return leg of a dump trip, call this with the reversed path:
    the descents on the way out are ascents on the way home.

    NOTE (open): a diagonal step covers sqrt(2) ground for the same
    elevation delta, so strictly it should cost less per metre climbed
    than an orthogonal one. The simulation makes the same simplification,
    so the invariant holds; both would have to change together.
    """
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        d = float(elevation[b]) - float(elevation[a])
        if d > 0.0:
            total += d
    return total

def astar_distance(
    start: Coord,
    goal: Coord | Iterable[Coord],
    width: int,
    height: int,
    blocked: set[Coord] | None = None,
) -> float:
    """Just the LENGTH of the shortest path, in grid steps, or +inf if
    there is no route.

    MATH: a path of N cells contains N-1 moves, hence len(path) - 1.
    That step count is what costs.tau_ij divides by v_max to get ticks
    and costs.energy_ij multiplies by ALPHA to get travel energy.

    Convenience wrapper over astar for callers that do not need the
    route itself. (Currently unused by the simulation proper -- the hot
    paths use nearest_work_path, which returns distance and route
    together from one search.)
    """
    path = astar(start, goal, width, height, blocked)
    return float("inf") if path is None else float(len(path) - 1)


def work_candidates(
    targets: Iterable[Coord],
    width: int,
    height: int,
    blocked: set[Coord],
) -> set[Coord]:
    """Cells from which a robot can work on `targets`: every in-bounds,
    unblocked cell within Chebyshev distance 1 of any target (the target
    itself included when traversable — a robot may dig the cell it
    stands on, but can never stand on a dump cell, which is blocked).

    MATH: the 3x3 Moore neighbourhood of each target, so at most 9 cells
    per target, minus anything out of bounds or impassable. That cap of
    9 is why MOACBBAAllocator.seats_for uses this as a physical limit on
    how many robots can share a task -- you cannot seat more diggers
    than there are cells to stand in.

    CALLED BY: nearest_work_cell and nearest_work_path (below), and
    MOACBBAAllocator.seats_for.
    """
    cands: set[Coord] = set()
    for tx, ty in targets:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = (tx + dx, ty + dy)
                if 0 <= c[0] < width and 0 <= c[1] < height \
                        and c not in blocked:
                    cands.add(c)
    return cands

# DEPRECATED because nearest_work_path exists
def nearest_work_cell(
    start: Coord,
    targets: Iterable[Coord],
    width: int,
    height: int,
    blocked: set[Coord],
    occupied: set[Coord] | None = None,
) -> tuple[Coord, float] | None:
    """The reachable work cell for `targets` nearest to `start`, and its
    path distance. `occupied` (other robots) removes candidates AND is
    avoided en route; pass nothing for uncontended cost estimates.
    Deterministic for fixed inputs. None if unreachable.

    Returns (cell, distance). This is the "p*" of the cost model when
    targets is a task cell, and "q*" when targets is a dump block.

    HOW: rather than running one A* per candidate cell and taking the
    best, it runs a SINGLE multi-goal A* whose goal set is every
    candidate. A* expands cells in order of increasing cost, so the
    first candidate it reaches is by construction the nearest one --
    one search instead of up to nine.

    CALLED BY: robot.assign (pick p* at assignment time), robot._reroute
    (pick a fresh p* when stuck) and model.dump_work_cell (pick q*).
    """
    occupied = occupied or set()
    cands = work_candidates(targets, width, height, blocked) - occupied
    if not cands:
        return None
    path = astar(start, cands, width, height, blocked | occupied)
    if path is None:
        return None
    return path[-1], float(len(path) - 1)

def nearest_work_path(
    start: Coord,
    targets: Iterable[Coord],
    width: int,
    height: int,
    blocked: set[Coord],
    occupied: set[Coord] | None = None,
):
    """Like nearest_work_cell, but also returns the route taken, so
    callers can measure elevation gain along it.
    Returns (cell, distance, path) or None.

    The route is needed because costs.energy_ij charges for elevation
    GAINED, which depends on the actual cells traversed and cannot be
    recovered from the endpoints alone -- see path_climb.

    CALLED BY: bidding.leg_cost (the approach leg of every bid) and
    model.dump_work_path (the haul leg).
    """
    occupied = occupied or set()
    cands = work_candidates(targets, width, height, blocked) - occupied
    if not cands:
        return None
    path = astar(start, cands, width, height, blocked | occupied)
    if path is None:
        return None
    return path[-1], float(len(path) - 1), path