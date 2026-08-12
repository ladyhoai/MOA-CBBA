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

Energy is accounted in three buckets (travel / climb / dig) as well as
in the total, because "large robots burn more" and "large robots are
sent uphill more" are different findings and the total cannot tell them
apart. The GUI inspector reads these.
"""

# TODO: LIDAR range -- _other_robot_cells currently gives every robot
# omniscient knowledge of every other robot's position.

from __future__ import annotations

from mesa.discrete_space import CellAgent

from .allocation import CBBAAgent
from .bidding import Stage
from .CBPAE import CBPAEAgent
from .MOACBBA import MOACBBAAgent
from .costs import RobotSpec
from .pathfinding import astar, chebyshev, nearest_work_cell
from .terrain import ALPHA, BETA, DIGGABLE, HARDNESS, Terrain, FULL_PAYLOAD_GAMMA, GAMMA

STUCK_LIMIT = 2

# Consecutive blocked/no-route ticks before a robot in execution phase 1
# gives the chunk back. Without this a robot walled off from its target
# holds the reservation for the rest of the run: CBPAE has
# dropIfUnreachable, CBBA has no drop path at all, so the release has to
# exist at the physics layer where it applies to every allocator equally.
UNREACHABLE_PATIENCE = 20


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
        self.energy_travel = 0.0     # ALPHA per grid step
        self.energy_climb = 0.0      # GAMMA per metre gained
        self.energy_dig = 0.0        # BETA per dig tick
        self.metres_climbed = 0.0
        self.distance_travelled = 0.0   # grid steps; Das et al. Fig. 4
        self.idle_ticks = 0
        self.busy_ticks = 0
        self.tasks_completed = 0
        self.tasks_dropped = 0
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
        self._waiting = 0      # consecutive ticks making no progress

        self.robot_id = len(model.robots)

        # Decentralised allocators: each robot runs its own copy.
        self.CBBA = CBBAAgent()
        self.CBPAE = CBPAEAgent()
        self.MOACBBA = MOACBBAAgent()

        # Max number of tasks in a CBBA bundle (L_t in Choi et al.).
        # RENAMED from `capacity`, which collided with spec.capacity
        # (payload, C_i). The two are unrelated: this is an algorithm
        # parameter, that is a physical property, and with a
        # heterogeneous fleet the collision silently produced wrong
        # bundle limits. Setting this to 1 reduces CBBA to CBAA
        # (Choi et al. Sec. IV-B) -- that is the decomposition ablation.
        self.bundle_limit = 6

    # ------------------------------------------------------------------ #
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
        position OR no dump position is reachable; allocators should
        then skip the task.

        Work cells claimed by other robots are excluded: with task
        decomposition, sibling chunks share a cell l_j, so without this
        two sharers pick the same p*, collide, and burn STUCK_LIMIT
        ticks each before _reroute untangles them."""
        task = self.model.tasks.get(task_id)
        if task.done or self.robot_id in task.assignees:
            return False
        # Seat limits are allocator policy (MOA-CBBA decides how many
        # robots a task is worth); the robot only enforces physics --
        # it cannot stand where a co-worker is already standing.
        shared = bool(task.assignees)
        claimed = self.model.claimed_work_cells(exclude=self)
        found = nearest_work_cell(self.cell.coordinate, [task.cell],
                                  self.model.grid.width,
                                  self.model.grid.height,
                                  self.model.blocked_cells(),
                                  claimed)
        if found is None:
            if shared:
                # The uncontended fallback ignores claimed cells, which is
                # harmless when nobody else is on this task and a
                # guaranteed collision when somebody is: both robots would
                # pick the same p* and burn STUCK_LIMIT ticks each.
                return False
            found = nearest_work_cell(self.cell.coordinate, [task.cell],
                                      self.model.grid.width,
                                      self.model.grid.height,
                                      self.model.blocked_cells())
            if found is None:
                return False
        work_cell, _ = found
        # q* is chosen once, from p*, at assignment time -- but it can
        # legitimately fail (no dump reachable), and indexing [0] on the
        # None return crashed the allocator instead of skipping the task.
        dump = self.model.dump_work_cell(work_cell)
        if dump is None:
            return False

        self.work_cell = work_cell
        self.dump_cell = dump[0]
        self.task_id = task_id
        task.add_assignee(self.robot_id)
        self._waiting = 0
        self.stage = Stage.TO_TASK
        self._plan_leg(self.work_cell)
        return True

    def abandon_task(self) -> None:
        """Drop the current task and return it to the pool.

        Das et al. Sec. 3.7.4: a robot may stop only during the FIRST
        phase of execution (travelling to the task), so that the task
        state is unchanged and the task stays reallocatable. Callers
        must check `can_abandon` first."""
        self._release_task()

    def _release_task(self) -> None:
        """Return the current chunk to the pool UNFINISHED.

        The counterpart of _finish_task, and the distinction matters:
        _finish_task stamps completed_tick, which is what makespan reads
        and what TaskRegistry.all_done tests. Stamping it on a chunk that
        still has volume in the ground puts a completion that never
        happened into the headline metric, inflates tasks_completed, and
        leaves CBPAE holding e = EXEC for a chunk nobody is working.

        Anything that is not "the volume reached zero" ends here."""
        if self.task_id is None:
            return
        task = self.model.tasks.get(self.task_id)
        task.drop_assignee(self.robot_id)
        # A robot can leave an ALREADY-EMPTY task through this path (no
        # reachable dump). If it was the last seat, nothing else will
        # ever stamp the task and all_done stays false forever.
        self._stamp_if_complete(task)
        self.tasks_dropped += 1
        self.task_id = None
        self.work_cell = None
        self.dump_cell = None
        self._path = []
        self._dest = None
        self._waiting = 0
        self.stage = Stage.IDLE

    @property
    def can_abandon(self) -> bool:
        """Phase 1 of execution only, and nothing in the hopper."""
        return (self.task_id is not None
                and self.stage is Stage.TO_TASK
                and self.payload <= 1e-9)

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
        """Seed both allocators' task lists. Called from model.py once
        the registry is populated."""
        self.CBBA.task_list = list(taskList)

    # ------------------------------------------------------------------ #
    # movement with collision avoidance
    # ------------------------------------------------------------------ #
    # TODO: CHANGE THIS TO LIDAR ROBOT DETECTION
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
        Returns True once the destination is REACHED.

        Arrival is `we are standing on _dest`, not `the path list is
        empty`. Those are different whenever _plan_leg failed, and
        conflating them was expensive in both directions:

          TO_TASK: a failed plan reported arrival, step() flipped to DIG,
          _dig_tick found the robot out of range, re-planned, failed
          again, and the robot ping-ponged between two stages in place
          for as long as the blockage lasted -- while counting every one
          of those ticks as busy_ticks, so mean_idle_ratio showed a fully
          occupied fleet.

          TO_DUMP: worse. A failed plan reported arrival, step() went to
          UNLOAD, and _unload_tick emptied the hopper wherever the robot
          happened to be standing -- soil deleted without ever reaching a
          dump site, and the task possibly closed out on the strength of
          it."""
        occupied = self._other_robot_cells() | self.model.dynamics.blocked()
        if not self._path:
            if self._dest is None or self.cell.coordinate == self._dest:
                return True
            self._wait_blocked(occupied)     # no route: wait, do not arrive
            return False
        self._move_credit += self.spec.v_max
        while self._path and self._move_credit >= 1.0:
            nxt = self._path[0]
            if nxt in occupied:
                self._wait_blocked(occupied)
                return False
            currentPosition = self.cell.coordinate
            self._path.pop(0)
            self.move_to(self.model.grid[nxt])
            self._move_credit -= 1.0
            self._stuck = 0
            self._waiting = 0
            self._spend_move(currentPosition, nxt)  # Eq. 5 travel term
        return (not self._path
                and (self._dest is None or self.cell.coordinate == self._dest))

    def _wait_blocked(self, occupied: set[tuple[int, int]]) -> None:
        """One tick of no progress: forfeit banked motion, count the wait,
        re-route on STUCK_LIMIT, and give the chunk back if this has been
        going on long enough that nothing is coming of it."""
        self._move_credit = 0.0
        self._stuck += 1
        self._waiting += 1
        self.wait_ticks += 1
        if self._waiting >= UNREACHABLE_PATIENCE and self.can_abandon:
            self._release_task()
            return
        if self._stuck >= STUCK_LIMIT:
            self._reroute(occupied)

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
        dest = dest if dest is not None else self._dest
        self._plan_leg(dest, avoid_robots=True)
        # avoid_robots=True searches a STRICTLY LARGER blocked set than
        # the plan that just failed, so escalating to it when already
        # stuck can only ever return the same route or none at all --
        # "re-planned (10, 7) -> (10, 7); path now 0 steps". Treating
        # other robots as walls is the optimistic case (they move); fall
        # back to routing through them rather than surrendering.
        if not self._path and dest is not None \
                and self.cell.coordinate != dest:
            self._plan_leg(dest, avoid_robots=False)

    def _dig_tick(self) -> None:
        task = self.model.tasks.get(self.task_id)
        coord = task.cell
        if chebyshev(self.cell.coordinate, coord) > 1:  # drifted? re-plan
            self._plan_leg(self.work_cell)
            self.stage = Stage.TO_TASK
            return
        terrain = Terrain(int(self.model.grid.terrain.data[coord]))
        if task.done:
            self._go_dump() if self.payload > 1e-9 else self._finish_task()
            return
        if terrain not in DIGGABLE:
            # Not diggable and not empty: nobody can ever finish this
            # chunk. Haul what is in the hopper first, then give the
            # chunk back UNFINISHED -- it is not a completion.
            self._go_dump() if self.payload > 1e-9 else self._release_task()
            return
        hardness = HARDNESS[terrain]
        dv = self.spec.dig_rate / hardness            # volume this tick
        dv = min(dv, task.remaining, self.spec.capacity - self.payload)
        if dv > 1e-12:
            self.payload += dv
            task.remaining -= dv
            self.model.grid.soil_volume.data[coord] -= dv
            self._spend(BETA, "dig")  # dig energy per tick (see costs.py)
        if task.done or self.payload >= self.spec.capacity - 1e-9:
            self._go_dump()

    def _go_dump(self) -> None:
        if self.dump_cell is None:  # q* chosen once per task, from p*
            found = self.model.dump_work_cell(self.work_cell)
            if found is None:       # no dump reachable: give it back
                self._release_task()
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

    def _stamp_if_complete(self, task) -> None:
        """A task is finished when the soil is IN THE DUMP, not when the
        hole is empty.

        With sharing those are different moments. Robot A digs the last
        of the volume and drives off with a full hopper; robot B, seated
        on the same task with an empty hopper, sees task.done on its very
        next DIG tick and used to stamp completed_tick immediately. That
        made TaskRegistry.all_done true -- and model.step() stops on
        all_done -- while A was still in TO_DUMP carrying soil that never
        reached a dump site. The run reported every task complete with
        material still in a hopper, and the makespan was short by the
        whole final haul.

        Stamping when the LAST seat empties fixes both: every sharer has
        by then either unloaded or given the task up, so the volume is
        genuinely delivered. Call this AFTER drop_assignee, from every
        exit path, or a task whose last robot leaves through the
        unreachable-dump route never gets stamped and the run never
        terminates."""
        if task.done and not task.assignees and task.completed_tick is None:
            task.completed_tick = self.model.tick

    def _finish_task(self) -> None:
        """This robot is done with the task. The TASK is done only when
        the last sharer says so -- see _stamp_if_complete."""
        task = self.model.tasks.get(self.task_id)
        if not task.done:
            # Safety net for any future call site that gets this wrong:
            # a completion stamp on a task with volume left is a lie the
            # makespan cannot detect.
            self._release_task()
            return
        task.drop_assignee(self.robot_id)
        self._stamp_if_complete(task)
        self.tasks_completed += 1
        self.task_id = None
        self.work_cell = None
        self.dump_cell = None
        self.stage = Stage.IDLE  # Algorithm 1, line 11: x_ij <- 0

    # ------------------------------------------------------------------ #
    # energy
    # ------------------------------------------------------------------ #
    def _spend_move(self, frm, to) -> None:
        """Only elevation GAINED costs energy (descent is free), and
        climbing loaded costs FULL_PAYLOAD_GAMMA times more."""
        self.distance_travelled += 1.0
        self._spend(ALPHA, "travel")
        elev = self.model.grid.elevation.data
        traction = self.model.dynamics.traction_scale()
        gain = float(elev[to]) - float(elev[frm])
        if gain > 0.0:
            self.metres_climbed += gain
            loaded = self.payload > 1e-9
            factor = FULL_PAYLOAD_GAMMA if loaded else 1.0
            self._spend(GAMMA * gain * traction * factor, "climb")

    def _spend(self, amount: float, kind: str = "travel") -> None:
        """Single point where energy leaves the battery, so the Phase 2
        drain multiplier is applied once and cannot drift out of sync
        with energy_ij (which also scales its whole return value)."""
        amount *= self.spec.drain_scale
        self.energy_used += amount
        if kind == "climb":
            self.energy_climb += amount
        elif kind == "dig":
            self.energy_dig += amount
        else:
            self.energy_travel += amount
        self.battery = max(0.0, self.battery - amount)

    # ------------------------------------------------------------------ #
    @property
    def payload_capacity(self) -> float:
        """C_i. Exposed for the DataCollector and the GUI inspector --
        `capacity` on this object is now the bundle limit."""
        return self.spec.capacity

    @property
    def robot_class(self) -> str:
        return self.spec.name