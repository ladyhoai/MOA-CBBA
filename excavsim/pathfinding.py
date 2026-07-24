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
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

Coord = tuple[int, int]

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _h(c: Coord, goals: frozenset[Coord]) -> float:
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
    g: dict[Coord, float] = {start: 0.0}
    came: dict[Coord, Coord] = {}
    tie = 0
    while open_heap:
        _, _, _, _, cur = heapq.heappop(open_heap)
        if cur in goals:
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
            ng = g[cur] + 1.0
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                came[nxt] = cur
                tie += 1
                heapq.heappush(open_heap,
                               (ng + _h(nxt, goals), nxt[0], nxt[1], tie, nxt)) # pyright: ignore[reportArgumentType]
    return None

def path_climb(path, elevation) -> float:
    """Total elevation travelled along the path which is passed as a parameter"""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    # TODO: I think diagonal travel will have a different way to calculate the elevation gain. TODO

    for a, b in zip(path, path[1:]):
        d = float(elevation[b]) - float(elevation[a])
        total += d
    
    return total

def astar_distance(
    start: Coord,
    goal: Coord | Iterable[Coord],
    width: int,
    height: int,
    blocked: set[Coord] | None = None,
) -> float:
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
    stands on, but can never stand on a dump cell, which is blocked)."""
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
    Deterministic for fixed inputs. None if unreachable."""
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
    Returns (cell, distance, path) or None."""
    occupied = occupied or set()
    cands = work_candidates(targets, width, height, blocked) - occupied
    if not cands:
        return None
    path = astar(start, cands, width, height, blocked | occupied)
    if path is None:
        return None
    return path[-1], float(len(path) - 1), path