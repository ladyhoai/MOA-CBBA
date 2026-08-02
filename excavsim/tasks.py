"""Excavation tasks t_j: a target cell l_j and a target volume V_j."""

from __future__ import annotations

from dataclasses import dataclass, field

Coord = tuple[int, int]


@dataclass
class Task:
    task_id: int
    cell: Coord            # l_j
    volume: float          # V_j (target excavation volume)
    remaining: float = field(init=False)
    future_remaining: float = field(init=False)
    assigned_to: int | None = None   # robot unique_id, or None
    completed_tick: int | None = None

    def __post_init__(self) -> None:
        self.remaining = self.volume

    @property
    def done(self) -> bool:
        return self.remaining <= 1e-9


class TaskRegistry:
    """Holds all tasks; the source of truth behind the BAM x_ij."""

    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 0

    def add(self, cell: Coord, volume: float) -> Task:
        t = Task(self._next_id, cell, volume)
        t.future_remaining = volume
        self._tasks[t.task_id] = t
        self._next_id += 1
        return t

    def get(self, task_id: int) -> Task:
        return self._tasks[task_id]

    @property
    def all(self) -> list[Task]:
        return list(self._tasks.values())

    @property
    def pending(self) -> list[Task]:
        """Unfinished and unassigned — the open column set of the BAM."""
        return [t for t in self._tasks.values() if not t.done and t.assigned_to is None]

    @property
    def unfinished(self) -> list[Task]:
        return [t for t in self._tasks.values() if not t.done]

    @property
    def isAllTaskDone(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.done and t.completed_tick is not None]