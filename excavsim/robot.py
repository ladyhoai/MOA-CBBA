"""Robot agent r_i: immutable spec S_i, mutable state M_i, and the
travel -> dig -> dump -> unload loop of Algorithm 1, line 8.

Physics rules (v2):
- Robots never share a cell (collision-free travel, dig, dump).
- Work happens from range: a robot digs a task cell from that cell or
  any Moore-adjacent cell, and unloads from any cell adjacent to its
  dump site's 2x2 block (dump cells themselves are impassable).
- Work cells are chosen deterministically (nearest_work_cell), the same
  procedure bids use, so uncontended execution matches Eqs. (4)-(5)
  exactly. Contention adds waits/detours on top: realized cost >= bid.
- Blocked movement: wait with movement credit reset; after STUCK_LIMIT
  consecutive blocked ticks, re-plan the current leg around the robots
  occupying the way. Head-on deadlock in a 1-wide corridor with no
  alternative route is a documented limitation (rare on open maps).
"""

from __future__ import annotations

from enum import Enum, auto

from mesa.discrete_space import CellAgent
from .allocation import ALLOCATORS

from .costs import RobotSpec
from .pathfinding import astar, chebyshev, nearest_work_cell
from .terrain import ALPHA, BETA, DIGGABLE, HARDNESS, Terrain

STUCK_LIMIT = 3


class Stage(Enum):
    IDLE = auto()
    TO_TASK = auto()
    DIG = auto()
    TO_DUMP = auto()
    UNLOAD = auto()


class ExcavatorRobot(CellAgent):
    """One excavation robot. `model` is an ExcavationModel."""

    def __init__(self, model, cell, spec: RobotSpec):
        super().__init__(model)
        self.cell = cell
        self.spec = spec                     # S_i (immutable)
        # M_i (mutable state)
        self.battery = spec.battery
        self.payload = 0.0
        # bookkeeping / metrics
        self.energy_used = 0.0
        self.idle_ticks = 0
        self.busy_ticks = 0
        self.tasks_completed = 0
        self.wait_ticks = 0        # ticks spent blocked by other robots
        # execution state
        self.stage = Stage.IDLE
        self.task_id: int | None = None
        self.work_cell: tuple[int, int] | None = None   # p*: dig position
        self.dump_cell: tuple[int, int] | None = None   # q*: unload position
        self._path: list[tuple[int, int]] = []
        self._dest: tuple[int, int] | None = None
        self._move_credit = 0.0
        self._unload_left = 0
        self._stuck = 0

        self.robot_id = len(model.robots)

    # The parameters below are used for CBBA (non-greedy algorithms)
        # Each robot instance will run its own copy of the algorithm because it is decentralised
        self.CBBA = ALLOCATORS["cbba"]()

        # Each robot can hold 4 task at one (but they still have to visit the dump site to complete one task).
        # In the future, if we can combine the excavator and dump truck into a single machine, this variable will
        # indicate the maximum payload of the machine. (Still to be decided)
        self.capacity = 6

        # communication (Phase 6): thin wrappers over the model's network
    # ------------------------------------------------------------------ #
    def send(self, payload: dict, to=None) -> None:
        """Queue a message; to=None broadcasts to robots in range."""
        self.model.comms.send(self, payload, to)

    def receive_all(self):
        """Drain this robot's inbox (list of Message, FIFO)."""
        return self.model.comms.receive_all(self)

    # ------------------------------------------------------------------ #
    # assignment interface used by allocators (writes the BAM row x_i)
    # ------------------------------------------------------------------ #
    def assign(self, task_id: int) -> bool:
        """Take task j. Returns False (no state change) if no dig
        position is reachable; allocators should then skip the task."""
        task = self.model.tasks.get(task_id)
        found = nearest_work_cell(self.cell.coordinate, [task.cell],
                                  self.model.grid.width,
                                  self.model.grid.height,
                                  self.model.blocked_cells())
        if found is None:
            return False
        self.work_cell, _ = found
        self.task_id = task_id
        task.assigned_to = self.unique_id
        self.dump_cell = None
        self.stage = Stage.TO_TASK
        self._plan_leg(self.work_cell)
        return True

    # ------------------------------------------------------------------ #
    # per-tick execution
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        if self.stage is Stage.IDLE:
            self.idle_ticks += 1
            return
        self.busy_ticks += 1
        if self.stage is Stage.TO_TASK:
            if self._advance_along_path():
                self.stage = Stage.DIG
        elif self.stage is Stage.DIG:
            self._dig_tick()
        elif self.stage is Stage.TO_DUMP:
            if self._advance_along_path():
                self.stage = Stage.UNLOAD
                self._unload_left = self.model.t_unload
        elif self.stage is Stage.UNLOAD:
            self._unload_tick()

    def updateTaskList(self, taskList) -> None:
        """Update the task list for the class instances of all allocators. Call this function in model.py 
        when new tasks is created at the beginning of the simulation. taskList is a list in TaskRegistry"""
        self.CBBA.task_list = taskList

    # ------------------------------------------------------------------ #
    # movement with collision avoidance
    # ------------------------------------------------------------------ #
    def _other_robot_cells(self) -> set[tuple[int, int]]:
        return {r.cell.coordinate for r in self.model.robots if r is not self}

    def _plan_leg(self, dest: tuple[int, int],
                  avoid_robots: bool = False) -> None:
        self._dest = dest
        blocked = self.model.blocked_cells()
        if avoid_robots:
            blocked = blocked | self._other_robot_cells()
        path = astar(self.cell.coordinate, dest,
                     self.model.grid.width, self.model.grid.height,
                     blocked=blocked)
        self._path = path[1:] if path else []
        self._move_credit = 0.0
        self._stuck = 0

    def _advance_along_path(self) -> bool:
        """Move up to v_max cells; never enter an occupied cell.
        Returns True once the destination is reached."""
        if not self._path:
            return True
        occupied = self._other_robot_cells()
        self._move_credit += self.spec.v_max
        while self._path and self._move_credit >= 1.0:
            nxt = self._path[0]
            if nxt in occupied:
                # blocked: wait, forfeit banked motion; re-route if stuck
                self._move_credit = 0.0
                self._stuck += 1
                self.wait_ticks += 1
                if self._stuck >= STUCK_LIMIT:
                    self._reroute(occupied)
                return False
            self._path.pop(0)
            self.move_to(self.model.grid[nxt])
            self._move_credit -= 1.0
            self._stuck = 0
            self._spend(ALPHA)  # travel energy per grid step (Eq. 5)
        return not self._path

    # ------------------------------------------------------------------ #
    # work stages
    # ------------------------------------------------------------------ #
    def _reroute(self, occupied: set[tuple[int, int]]) -> None:
        """Stuck: pick a fresh, currently-free work/dump cell for the
        same target and route around parked robots. Falls back to
        keeping the old destination if no free alternative exists."""
        if self.stage is Stage.TO_TASK and self.task_id is not None:
            task = self.model.tasks.get(self.task_id)
            found = nearest_work_cell(self.cell.coordinate, [task.cell],
                                      self.model.grid.width,
                                      self.model.grid.height,
                                      self.model.blocked_cells(), occupied)
            if found is not None:
                self.work_cell = found[0]
        elif self.stage is Stage.TO_DUMP:
            found = self.model.dump_work_cell(self.cell.coordinate, occupied)
            if found is not None:
                self.dump_cell = found[0]
        dest = (self.work_cell if self.stage is Stage.TO_TASK
                else self.dump_cell)
        self._plan_leg(dest if dest is not None else self._dest,
                       avoid_robots=True)

    def _dig_tick(self) -> None:
        task = self.model.tasks.get(self.task_id)
        coord = task.cell
        if chebyshev(self.cell.coordinate, coord) > 1:  # drifted? re-plan
            self._plan_leg(self.work_cell)
            self.stage = Stage.TO_TASK
            return
        terrain = Terrain(int(self.model.grid.terrain.data[coord]))
        if terrain not in DIGGABLE or task.done:
            self._go_dump() if self.payload > 1e-9 else self._finish_task()
            return
        hardness = HARDNESS[terrain]
        dv = self.spec.dig_rate / hardness            # volume this tick
        dv = min(dv, task.remaining, self.spec.capacity - self.payload)
        if dv > 1e-12:
            self.payload += dv
            task.remaining -= dv
            self.model.grid.soil_volume.data[coord] -= dv
            self._spend(BETA)  # dig energy per tick (see costs.py)
        if task.done or self.payload >= self.spec.capacity - 1e-9:
            self._go_dump()

    def _go_dump(self) -> None:
        if self.dump_cell is None:  # q* chosen once per task, from p*
            found = self.model.dump_work_cell(self.work_cell)
            if found is None:       # no dump reachable: abandon safely
                self._finish_task()
                return
            self.dump_cell, _ = found
        self.stage = Stage.TO_DUMP
        self._plan_leg(self.dump_cell)

    def _unload_tick(self) -> None:
        self._unload_left -= 1
        if self._unload_left > 0:
            return
        self.payload = 0.0  # E_unload = 0 by assumption
        task = self.model.tasks.get(self.task_id)
        if task.done:
            self._finish_task()
        else:
            self.stage = Stage.TO_TASK
            self._plan_leg(self.work_cell)   # return to the same p*

    def _finish_task(self) -> None:
        task = self.model.tasks.get(self.task_id)
        task.completed_tick = self.model.tick
        task.assigned_to = None
        self.tasks_completed += 1
        self.task_id = None
        self.work_cell = None
        self.dump_cell = None
        self.stage = Stage.IDLE  # Algorithm 1, line 11: x_ij <- 0

    def _spend(self, amount: float) -> None:
        self.energy_used += amount
        self.battery = max(0.0, self.battery - amount)
