"""Excavation tasks t_j: a target cell l_j and a target volume V_j.

ONE TASK PER SITE, BUT POSSIBLY SEVERAL ROBOTS PER TASK. Static
decomposition (pre-splitting a site into chunks) is gone -- a site is
one indivisible task. What replaced it is genuine concurrent sharing:
`assignees` is a SET, so k robots can work the same task at the same
time and the volume drains k times faster.

That is a deliberate move from ST-SR to ST-MR in the Gerkey & Mataric
taxonomy, and it should be stated as such in the write-up. It is also
opt-in per allocator: CBBA and CBPAE filter on "is anybody on this
task", so they keep putting exactly one robot on each task and remain
ST-SR. Only MOA-CBBA fills more than one seat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Coord = tuple[int, int]


@dataclass
class Task:
    """One excavation site. Exactly one robot per Task."""

    task_id: int
    cell: Coord            # l_j
    volume: float          # V_j
    remaining: float = field(init=False)
    assignees: set[int] = field(default_factory=set)   # robot_ids at work
    completed_tick: int | None = None

    def __post_init__(self) -> None:
        self.remaining = self.volume

    @property
    def done(self) -> bool:
        return self.remaining <= 1e-9

    @property
    def assigned_to(self) -> int | None:
        """Read-only compatibility view: the lowest-id robot on this
        task, or None if nobody is. Every ST-SR call site asks one of
        two questions -- "is anyone on it" or "is it me" -- and both
        still answer correctly, because those allocators never seat more
        than one robot. Writes go through add_assignee / drop_assignee,
        so nothing can silently overwrite a co-worker."""
        return min(self.assignees) if self.assignees else None

    @property
    def sharers(self) -> int:
        return len(self.assignees)

    def add_assignee(self, robot_id: int) -> None:
        self.assignees.add(robot_id)

    def drop_assignee(self, robot_id: int) -> None:
        self.assignees.discard(robot_id)


class TaskRegistry:
    """Holds all tasks; the source of truth behind the BAM x_ij."""

    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 0

    # ---------------------------------------------------------------- #
    def add(self, cell: Coord, volume: float) -> Task:
        t = Task(self._next_id, cell, volume)
        self._tasks[t.task_id] = t
        self._next_id += 1
        return t

    def get(self, task_id: int) -> Task:
        return self._tasks[task_id]

    def at_cell(self, cell: Coord) -> list[Task]:
        return [t for t in self._tasks.values() if t.cell == cell]

    # ---------------------------------------------------------------- #
    @property
    def all(self) -> list[Task]:
        return list(self._tasks.values())

    @property
    def pending(self) -> list[Task]:
        """Unfinished with NOBODY on it -- the open column set of the BAM
        for the single-assignment allocators."""
        return [t for t in self._tasks.values()
                if not t.done and not t.assignees]

    def seats_open(self, robot_id: int, max_sharers: int = 1) -> list[Task]:
        """Tasks this robot may bid on when several robots per task are
        allowed: unfinished, and either it is already seated or a seat is
        free. `pending` is the max_sharers=1 case of this."""
        return [t for t in self._tasks.values()
                if not t.done and (robot_id in t.assignees
                                   or len(t.assignees) < max_sharers)]

    @property
    def unfinished(self) -> list[Task]:
        return [t for t in self._tasks.values() if not t.done]

    @property
    def all_done(self) -> bool:
        """Every task finished and stamped."""
        return all(t.done and t.completed_tick is not None
                   for t in self._tasks.values())