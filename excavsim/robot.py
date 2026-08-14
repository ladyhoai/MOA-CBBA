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

# How long a sighting stays in the occupancy map without being seen
# again.
#
# Sightings originally never expired: a cell was only cleared if the
# robot went back and saw it empty. That is wrong, because hazards last
# 5-40 ticks and obstacles WANDER, so a cell seen blocked and then left
# behind is usually clear long before anyone returns. The map filled
# with ghosts and the fleet routed around hazards that no longer
# existed. An expiry is the cheapest correct answer: believe what you
# saw, but not forever.
MAP_TTL = 30

# Obstacles wander every tick with probability obstacle_move_probability
# (0.5 by default), so where one was seen three ticks ago says almost
# nothing about where it is now. Remembering them on the hazard timescale
# was pure ghost generation: MAP_TTL = 30 kept a wandering machine pinned
# to a cell it had left twenty-something ticks earlier.
OBSTACLE_TTL = 3

# Cells adjacent to an obstacle seen THIS TICK.
#
# The arithmetic, corrected. dynamics moves an obstacle with probability
# obstacle_move_probability (0.5) and then picks uniformly among its
# free neighbours, so for any ONE ring cell
#
#     P(occupied next tick) = 0.5 * 1/|options| ~= 0.5/8 = 6.25%
#
# not the 50% an earlier version of this comment claimed. The 50% figure
# is the chance it moves into SOME ring cell, and a route crosses one or
# two of the eight, not all of them. (The genuinely 50% cell is the one
# the obstacle already occupies -- it stays put half the time -- and that
# cell is in known_blocked already.)
#
# The ring is a ONE-TICK prediction, so it may only be applied to cells
# the robot reaches in about one tick. An obstacle performs a random
# walk with p = 0.5, so its expected displacement after t ticks is about
# sqrt(0.5 t) per axis: 0.7 cells after 1 tick, 1.0 after 2, 1.6 after
# 5. The ring is one cell wide, so beyond ~2 ticks the obstacle is
# typically outside it and the ring is superstition. HALO_HORIZON was 6
# CELLS, which at v_max 0.8-1.2 is 5 to 7.5 ticks -- three times past
# the point where the prediction means anything.
#
# Measured in CELLS but compared in CHEBYSHEV distance, because that is
# the metric the robot actually moves in: a diagonal step costs one step
# like any other. The old Euclidean test made the horizon anisotropic,
# reaching a full 6 steps along the axes but only 4 diagonally.
HALO_HORIZON = 2

# Plan around robots the sensor can see, instead of only after walking
# into one.
#
# Measured cause of blocking over four seeds: 46.4% no route at all,
# 32.0% hazards, 18.7% ROBOTS, 2.9% obstacles. And every one of the
# robot collisions was with a machine the blocked robot could SEE --
# none were out of sensor range. The information was there; _plan_leg
# simply never asked, because avoid_robots defaulted to False at all
# four normal call sites and only _reroute passed True. So the sequence
# was: plan straight through a visible machine, hit it, burn
# STUCK_LIMIT ticks, then re-plan around it.
AVOID_ROBOTS_WHEN_PLANNING = True


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
        # Wandering obstacles, kept apart from hazards because they age
        # on a completely different timescale. cell -> tick seen.
        self._obstacles: dict[tuple[int, int], int] = {}
        # Occupancy memory: hazard cells this robot has SEEN blocked and
        # not yet seen cleared. Static terrain is not in here -- see
        # known_blocked.
        # Occupancy map: dynamic cell -> tick this robot SAW it blocked.
        # First-hand only; robots do not share maps.
        self._map: dict[tuple[int, int], int] = {}
        # Total volume actually unloaded at a dump site. The third term
        # of the conservation identity checked in debug.py.
        self.soil_delivered = 0.0

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
        claimed = self.model.claimed_work_cells(exclude=self, observer=self)
        found = nearest_work_cell(self.cell.coordinate, [task.cell],
                                  self.model.grid.width,
                                  self.model.grid.height,
                                  self.known_blocked(),
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
                                      self.known_blocked())
            if found is None:
                return False
        work_cell, _ = found
        # q* is chosen once, from p*, at assignment time -- but it can
        # legitimately fail (no dump reachable), and indexing [0] on the
        # None return crashed the allocator instead of skipping the task.
        dump = self.model.dump_work_cell(work_cell,
                                         blocked=self.known_blocked())
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
        # Conservation guard. Every caller should already have checked
        # can_abandon (stage TO_TASK, empty hopper -- Das et al. Sec.
        # 3.7.4), so this should never fire. If one slips through, put
        # the soil back in the ground rather than let it vanish into a
        # hopper: the volume was subtracted from task.remaining when it
        # was dug, so returning it is what keeps
        #     soil in ground + soil in hoppers + soil at dump
        # constant. Better a task that looks briefly bigger than a
        # makespan computed over soil nobody ever moved.
        if self.payload > 1e-9:
            task.remaining += self.payload
            self.model.grid.soil_volume.data[task.cell] += self.payload
            task.completed_tick = None      # it is demonstrably not done
            self.payload = 0.0
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
    # ------------------------------------------------------------------ #
    # sensing (sigma_i, Phase 3)
    # ------------------------------------------------------------------ #
    @property
    def sensor_radius(self) -> float:
        """Effective sensing range this tick: the spec's sigma_i degraded
        by weather. Fog scales it to 0.4 and storm to 0.3, so a small
        machine (sigma = 6) sees barely two cells in a storm while a
        large one (sigma = 10) still sees three."""
        return self.spec.sensor_range * self.model.dynamics.sensor_scale()

    def sense(self) -> None:
        """Update the occupancy memory from what is visible right now.

        Two directions, and the second one matters as much as the first:
        cells inside the radius that are blocked get REMEMBERED, and
        cells inside the radius that are clear get FORGOTTEN. Without the
        forgetting, an expired hazard would be avoided for the rest of
        the run; without the remembering, a robot would forget an
        obstacle the moment it turned away and oscillate in front of it.

        Only DYNAMIC blockage is sensed. Bedrock and dump sites are
        surveyed before work starts, so they belong in the map, not in
        the sensor -- see known_blocked.
        """
        if not getattr(self.model, "sensing_enabled", False):
            return
        r = self.sensor_radius
        if r <= 0.0:
            return
        cx, cy = self.cell.coordinate
        dyn = self.model.dynamics
        hazards = dyn.hazard_cells()
        obstacles = dyn.obstacle_cells()
        now = self.model.tick
        rr = r * r
        lo_x, hi_x = int(cx - r), int(cx + r)
        lo_y, hi_y = int(cy - r), int(cy + r)
        for x in range(lo_x, hi_x + 1):
            for y in range(lo_y, hi_y + 1):
                dx, dy = x - cx, y - cy
                if dx * dx + dy * dy > rr:
                    continue                     # outside the disc
                c = (x, y)
                # Two registers, because the two things age differently.
                if c in hazards:
                    self._map[c] = now
                else:
                    self._map.pop(c, None)
                if c in obstacles:
                    self._obstacles[c] = now
                else:
                    self._obstacles.pop(c, None)

        # Age out. Hazards sit still, so they keep for MAP_TTL; obstacles
        # wander, so a sighting is worthless within a few ticks.
        cutoff = now - MAP_TTL
        for c in [k for k, t in self._map.items() if t < cutoff]:
            del self._map[c]
        ob_cutoff = now - OBSTACLE_TTL
        for c in [k for k, t in self._obstacles.items() if t < ob_cutoff]:
            del self._obstacles[c]

    def known_blocked(self) -> set[tuple[int, int]]:
        """What THIS robot believes is impassable: the surveyed map plus
        whatever its sensor has found. Everything planned or bid on goes
        through here rather than model.blocked_cells(), so a robot can
        route straight into a hazard it has not seen yet -- and then
        re-route when it comes into range, which is the whole point."""
        if not getattr(self.model, "sensing_enabled", False):
            return self.model.blocked_cells()
        return (self.model._static_blocked() | self._map.keys()
                | self._obstacles.keys())

    def obstacle_halo(self, origin: tuple[int, int]) -> set[tuple[int, int]]:
        """Cells an obstacle seen THIS TICK may step into on the next.

        Three corrections against the first version, all of them making
        the ring smaller and the claim honest -- see HALO_HORIZON above
        for the arithmetic:

          - CHEBYSHEV distance, not Euclidean, because that is the metric
            the robot moves in.
          - HALO_HORIZON = 2, not 6, because a one-cell ring is only a
            valid prediction about one or two ticks ahead.
          - Only sightings from the CURRENT tick. _obstacles holds them
            for OBSTACLE_TTL ticks, and a three-tick-old position is
            already ~1.2 cells wrong, so drawing a one-cell ring around
            it compounds a stale position with a short-lived prediction.

        The honest expected value is small: about 6% per ring cell
        crossed. This is a cheap hedge, not a large win, and it should
        not be described as one.
        """
        out: set[tuple[int, int]] = set()
        if not self._obstacles:
            return out
        ox, oy = origin
        now = self.model.tick
        w, h = self.model.grid.width, self.model.grid.height
        for (cx, cy), seen in self._obstacles.items():
            if seen < now:
                continue                     # stale position, stale ring
            if max(abs(cx - ox), abs(cy - oy)) > HALO_HORIZON:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (cx + dx, cy + dy)
                    if 0 <= n[0] < w and 0 <= n[1] < h:
                        out.add(n)
        return out

    def visible_robots(self) -> list:
        """Other robots this one can currently SEE.

        Positions are sensed, never remembered: a robot moves every tick,
        so a sighting from ten ticks ago is worse than no information at
        all -- it would have the planner routing around a machine that
        has long since driven off. Contrast the obstacle map, which is
        remembered precisely because hazards sit still.
        """
        if not getattr(self.model, "sensing_enabled", False):
            return [r for r in self.model.robots if r is not self]
        rr = self.sensor_radius ** 2
        cx, cy = self.cell.coordinate
        out = []
        for r in self.model.robots:
            if r is self:
                continue
            ox, oy = r.cell.coordinate
            if (ox - cx) ** 2 + (oy - cy) ** 2 <= rr:
                out.append(r)
        return out

    def _other_robot_cells(self) -> set[tuple[int, int]]:
        """For PLANNING: only robots in sensor range. A robot cannot steer
        around a machine it has no way of knowing is there."""
        return {r.cell.coordinate for r in self.visible_robots()}

    def _all_robot_cells(self) -> set[tuple[int, int]]:
        """For PHYSICS: every robot, seen or not. Two machines cannot
        occupy one cell regardless of who noticed whom -- not seeing
        something is not permission to drive through it. The gap between
        this and _other_robot_cells is exactly where unseen robots cause
        real waits instead of tidy detours."""
        return {r.cell.coordinate for r in self.model.robots if r is not self}

    def _plan_leg(self, dest: tuple[int, int],
                  avoid_robots: bool = AVOID_ROBOTS_WHEN_PLANNING) -> None:
        """Route to `dest`, avoiding what can actually be avoided.

        Two kinds of blockage, and they must not be treated alike:

        HARD -- bedrock, dump blocks, sensed hazards. These do not move,
        so a route through them is not a route.

        SOFT -- other robots, and the obstacle halo. A machine standing
        in the way will drive off; a halo cell is only ~6% likely to be
        occupied at all. Treating either as a wall makes reachable work
        look impossible, so both are PREFERENCES: plan with them, drop
        them if that leaves no route.

        Preferences are dropped weakest-evidence-first: the halo goes
        before the robots do, because a machine is standing there NOW
        whereas the halo is a guess about the next tick.
        """
        self._dest = dest
        hard = self.known_blocked()
        w, h = self.model.grid.width, self.model.grid.height
        start = self.cell.coordinate

        soft = self._other_robot_cells() if avoid_robots else set()
        halo = set()
        if getattr(self.model, "obstacle_halo_enabled", False):
            halo = self.obstacle_halo(start)

        path = None
        tried = []
        for extra in ((soft | halo) - {dest}, soft - {dest}, set()):
            if extra in tried:
                continue          # same search as a tier already run
            tried.append(extra)
            path = astar(start, dest, w, h, hard | extra)
            if path:
                break

        self._path = path[1:] if path else []
        # _move_credit is deliberately NOT reset here. It is banked
        # fractional movement -- a robot at v_max = 1.2 carries 0.2 of a
        # step into the next tick -- and re-planning a route is not a
        # reason to lose it. Zeroing it here charged up to a full tick
        # every time _plan_leg ran, which is once per stage change and
        # again on every re-route: 312 times in a four-seed measurement,
        # for +8.7% on travel against the closed-form tau_ij.
        # (_wait_blocked still forfeits it, and that one IS deliberate:
        # a robot that could not move did not accumulate anything.)
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
        occupied = self._all_robot_cells() | self.model.dynamics.blocked()
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
        going on long enough that nothing is coming of it.

        `occupied` is the PHYSICAL set -- every robot and every hazard,
        seen or not -- because that is what actually stops the wheels.
        The re-plan gets a different set: only what this robot can sense.
        Handing the physical set to _reroute would let a stuck machine
        route around hazards and robots it has no way of knowing about,
        which is omniscience smuggled in through the recovery path. The
        machine that just blocked us is adjacent and therefore inside the
        sensor disc anyway, so nothing needed is lost.
        """
        self._move_credit = 0.0
        self._stuck += 1
        self._waiting += 1
        self.wait_ticks += 1
        if self._waiting >= UNREACHABLE_PATIENCE and self.can_abandon:
            self._release_task()
            return
        if self._stuck >= STUCK_LIMIT:
            sensed = self._other_robot_cells() | (
                self.known_blocked() - self.model._static_blocked())
            self._reroute(sensed)

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
                                      self.known_blocked(), occupied)
            if found is not None:
                self.work_cell = found[0]
        elif self.stage is Stage.TO_DUMP:
            found = self.model.dump_work_cell(
                self.cell.coordinate, occupied,
                blocked=self.known_blocked())
            if found is not None:
                self.dump_cell = found[0]
        dest = (self.work_cell if self.stage is Stage.TO_TASK
                else self.dump_cell)
        dest = dest if dest is not None else self._dest
        self._plan_leg(dest, avoid_robots=True)
        # The explicit second attempt this used to need is gone:
        # _plan_leg now falls back through its own preference tiers, so
        # a failed robot-avoiding search already retries without them.

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
            found = self.model.dump_work_cell(
                self.work_cell, blocked=self.known_blocked())
            if found is None:
                # No reachable dump AND soil in the hopper. This used to
                # call _release_task(), which drops the task and the seat
                # but never touches self.payload -- so the volume was
                # already subtracted from task.remaining at dig time,
                # never delivered, and carried around for the rest of the
                # run. Measured: 0.34-0.99 units stranded per CBPAE run,
                # tasks stamped complete on soil sitting in a hopper, the
                # robot permanently short of that much capacity, and the
                # haul energy for it never charged.
                #
                # A loaded machine cannot abandon its load, so it WAITS
                # in place and retries. _dig_tick calls _go_dump again on
                # the next tick, and hazards expire, so this clears
                # itself. It cannot hang forever: the wait counter feeds
                # the monitor's stall check, and the task is still held
                # so nothing else is blocked by it being in limbo.
                self.wait_ticks += 1
                self._waiting += 1
                self._path = []
                return
            self.dump_cell, _ = found
        self.stage = Stage.TO_DUMP
        self._plan_leg(self.dump_cell)

    def _unload_tick(self) -> None:
        self._unload_left -= 1
        if self._unload_left > 0:
            return
        self.soil_delivered += self.payload    # conservation bookkeeping
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