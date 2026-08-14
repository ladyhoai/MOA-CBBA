"""ExcavationModel: the Phase 1 simulation core.

- OrthogonalMooreGrid  -> 8-connected G = (V, E)
- PropertyLayers       -> H(v) via terrain type, V(v), elevation
- TaskRegistry         -> T = {t_1..t_n}
- ExcavatorRobot       -> R = {r_1..r_m}
- DataCollector        -> metrics per tick (soil moved, energy, idle,
                          tasks per robot), Table 1 item 1.3

Every run is seeded (Mesa passes `seed` to a per-model RNG), so batch
experiments are reproducible.
"""

from __future__ import annotations
import numpy as np
from mesa import DataCollector, Model
from mesa.discrete_space import (FixedAgent, OrthogonalMooreGrid,
                                 PropertyLayer)
from perlin_numpy import generate_fractal_noise_2d
from .allocation import ALLOCATORS
from .MOACBBA import register as _register_moacbba

_register_moacbba()
from .comms import CommNetwork
from .costs import RobotSpec, objective
from .fleet import ROBOT_CLASSES, build_fleet, fleet_summary
from .pathfinding import nearest_work_cell, nearest_work_path, work_candidates
from .robot import ExcavatorRobot
from .tasks import TaskRegistry
from .terrain import T_UNLOAD, Terrain
from .dynamics import Coord, DynamicsManager

# Kept for anything still importing it; the homogeneous baseline is now
# fleet_mode="none", which uses the medium class.
DEFAULT_SPEC = ROBOT_CLASSES["medium"]


class TaskMarker(FixedAgent):
    """Passive agent standing on an excavation site so the GUI draws it
    through the standard agent pipeline. No physics: it never moves,
    digs, or spends energy."""

    def __init__(self, model, cell, task):
        super().__init__(model)
        self.cell = cell
        self.task = task

    @property
    def remaining(self) -> float:
        return self.task.remaining

    @property
    def done(self) -> bool:
        return self.task.done

    def step(self) -> None:
        if self.done:
            self.remove()


class ExcavationModel(Model):
    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        n_robots: int = 4,
        n_tasks: int = 8,
        allocator: str = "cbba",
        w1: float = 1,
        w2: float = 1,

        # --- Phase 2 heterogeneity (Table 1, items 2.1-2.4) --------- #
        # "full" by default: under "capacity" only the payload differs,
        # so v_max, dig_rate and drain_scale are identical across the
        # fleet and robots are near-interchangeable -- most allocations
        # are then nearly as good as each other and no allocator can
        # separate from another. "capacity" remains available as the
        # Phase 2 ablation.
        fleet_mode: str = "full",       # "none" | "capacity" | "full"
        fleet_mix: tuple[str, ...] | None = None,   # None -> small/medium/large
        fleet_shuffle: bool = False,    # False -> deterministic round-robin
        # Finite by default. At None every robot hears every other every
        # round, so CBBA's Table I has nothing to resolve: every log
        # reports last_round=1, converged=True, and the s vector,
        # mergeTimestamps and the whole third-agent half of the decision
        # table are dead code the run still pays for. At ~10 on a 32x32
        # grid the network is a genuine multi-hop mesh and consensus
        # becomes the thing being compared. None restores the old
        # complete-graph behaviour.
        # Phase 3 sensing. False = the old omniscient behaviour, where
        # every robot sees every hazard the tick it appears. True makes a
        # robot plan against what its OWN sensor has found, so sigma_i
        # (a dead RobotSpec field until now) and the weather sensor_scale
        # (computed and only ever displayed) both start doing work.
        sensing_enabled: bool = True,
        comm_range: float | None = 10.0,
        packet_loss: float = 0.0,
        comm_latency: int = 0,
        comm_bandwidth: int | None = None,

        # Phase 4 dynamics default to OFF. They used to default ON
        # (0.05 / 0.3), which meant every run that did not explicitly
        # zero them -- including every Phase 1/2 baseline -- was measured
        # against a moving target.
        hazard_rate: float = 0.0,          # 4.2 zones/tick (0 = off)
        obstacle_rate: float = 0.0,        # 4.3 obstacles/tick (0 = off)
        weather_enabled: bool = False,     # 4.4 weather on/off
        weather_change_rate: float = 0.0,  # 4.4 transitions/tick
        hazard_duration: int = 5, hazard_size: int = 2,
        obstacle_duration: int = 60, obstacle_move_prob: float = 0.5,
        max_obstacles: int = 10,

        rock_fraction: float = 0.15,
        gravel_fraction: float = 0.2,

        # --- bedrock ridges (impassable) ---------------------------- #
        # Terrain.BEDROCK, HARDNESS[BEDROCK] = inf and _static_blocked's
        # np.where were all wired up and never exercised: _scatter_terrain
        # only ever wrote ROCK and GRAVEL, so the map was an open plain
        # and every robot could reach every task at roughly equal cost.
        # Ridges (not noise) create regions, which is what makes WHICH
        # robot gets WHICH task change the cost by a factor rather than a
        # few percent. 0 ridges reproduces the open-plain behaviour.
        bedrock_ridges: int = 6,
        bedrock_length: int = 10,

        elevation_scale: float = 15.0,
        elevation_octaves: int = 4,  # How detailed
        elevation_persistence: float = 0.5,

        # Log-uniform over a wider range. Uniform(1, 4) plus chunking
        # produced NO critical path: chunk_count splits by
        # volume // MIN_CHUNK, so the biggest sites yielded the SMALLEST
        # chunks and every chunk ended up between 1.0 and ~2.0. Makespan
        # was then total-work / n_robots regardless of who did what, and
        # no makespan-aware allocator could beat any other. Durations
        # need to differ by 5-10x for the allocation to matter.
        task_volume: tuple[float, float] = (1.0, 12.0),
        volume_dist: str = "loguniform",   # or "uniform" for the old runs

        seed: int | None = None,
    ):
        super().__init__(rng=int(seed) if seed is not None else None)
        self.w1, self.w2 = w1, w2
        self.sensing_enabled = bool(sensing_enabled)
        # Set by whichever allocator wants predictive obstacle avoidance;
        # see MOACBBAAllocator. Off unless an allocator asks, so CBBA and
        # CBPAE keep planning exactly as before and the mechanism stays a
        # clean ablation axis rather than a change to the whole world.
        self.obstacle_halo_enabled = False
        self.t_unload = T_UNLOAD
        self.tick = 0
        # Largest CBBA bundle any robot has held this run. Cheap, and
        # it is the one-number check that L_t > 1 is actually being used
        # -- i.e. that this is CBBA and not CBAA in disguise.
        self._max_bundle_seen = 0
        self._static_blocked_cache: set[Coord] | None = None
        # (robot, task, start) -> (tau, E, q*) for ONE auction round.
        # Robots do not move during allocate(), so nothing that
        # feeds a leg cost can change inside a round.
        self._leg_cache: dict = {}
        self.changed_cells: set[Coord] = set()  # Algorithm 1, line 3 hook
        # print("Allocator used: ", allocator)

        # --- grid and property layers ---------------------------------- #
        self.grid = OrthogonalMooreGrid((width, height), torus=False,
                                        random=self.random)
        self.grid.add_property_layer(
            PropertyLayer("terrain", (width, height),
                          default_value=int(Terrain.SOIL), dtype=int))
        self.grid.add_property_layer(
            PropertyLayer("soil_volume", (width, height),
                          default_value=0.0, dtype=float))
        self.grid.add_property_layer(
            PropertyLayer("elevation", (width, height),
                          default_value=0.0, dtype=float))
        self._scatter_terrain(rock_fraction, gravel_fraction)
        self._scatter_bedrock(bedrock_ridges, bedrock_length)
        self._scatter_elevation(elevation_octaves, elevation_persistence, elevation_scale)

        # --- dump sites: 2x2 impassable blocks --------------------------- #
        self.dump_blocks: list[tuple[Coord, ...]] = []
        while len(self.dump_blocks) < 2:
            self._place_dump_block()

        # --- Phase 4 dynamics (built early: blocked_cells() and
        # traction_scale() are consulted from the first bid onwards) --- #
        self.dynamics = DynamicsManager(
            self, hazard_rate=hazard_rate, obstacle_rate=obstacle_rate,
            weather_enabled=weather_enabled,
            weather_change_rate=weather_change_rate,
            hazard_duration=hazard_duration, hazard_size=hazard_size,
            obstacle_duration=obstacle_duration,
            obstacle_move_probability=obstacle_move_prob,
            max_obstacles=max_obstacles)

        # --- tasks ------------------------------------------------------ #
        self.tasks = TaskRegistry()
        lo, hi = task_volume
        for _ in range(n_tasks):
            cell = self._random_diggable_coord()
            vol = (self.random.uniform(lo, hi) if volume_dist == "uniform"
                   else lo * (hi / lo) ** self.random.random())
            self.grid.soil_volume.data[cell] += vol
            task = self.tasks.add(cell, vol)
            TaskMarker(self, self.grid[cell], task)
        # print(len(self.tasks.all), "tasks placed")
        # --- robots (Phase 2: one spec per robot, not one for all) ------ #
        self.fleet_mode = fleet_mode
        self.fleet_specs = build_fleet(
            n_robots, mode=fleet_mode, mix=fleet_mix,
            rng=self.random if fleet_shuffle else None)
        self.fleet_summary = fleet_summary(self.fleet_specs)

        self.robots: list[ExcavatorRobot] = []
        for spec in self.fleet_specs:
            cell = self.grid[self._random_empty_coord()]
            self.robots.append(ExcavatorRobot(self, cell, spec))

        # Populating the list of task for each robot to prepare for CBBA
        for robot in self.robots:
            robot.updateTaskList(self.tasks.all)
        
        # --- Phase 6 communication layer (neutral by default) ----------- #
        # Change comm_range to a number so that the s vector is utilised

        # The GUI slider cannot express None, so it sends 0 for
        # "unlimited". Anything <= 0 would otherwise mean a robot can
        # hear nobody at all, which silently disables consensus.
        if comm_range is not None and comm_range <= 0:
            comm_range = None
        self.comms = CommNetwork(self, comm_range=comm_range,
                                 packet_loss=packet_loss,
                                 latency=comm_latency,
                                 bandwidth=comm_bandwidth)
                
        self.allocator = ALLOCATORS[allocator]()        

        # --- metrics (Table 1, item 1.3) -------------------------------- #
        self.datacollector = DataCollector(
            model_reporters={
                "tick": lambda m: m.tick,
                "tasks_done": lambda m: sum(t.done for t in m.tasks.all),
                "soil_moved": lambda m: sum(t.volume - t.remaining
                                            for t in m.tasks.all),
                "total_energy": lambda m: sum(r.energy_used for r in m.robots),
                "mean_idle_ratio": lambda m: (
                    sum(r.idle_ticks for r in m.robots)
                    / max(1, m.tick * len(m.robots))),
                # Das et al. report average distance per ACTIVE robot
                # (a robot with at least one task allocated) alongside
                # execution time; without it their Fig. 4 has no analogue.
                "mean_distance_active": lambda m: (
                    sum(r.distance_travelled for r in m.robots
                        if r.tasks_completed or r.task_id is not None)
                    / max(1, sum(1 for r in m.robots
                                 if r.tasks_completed or r.task_id is not None))),
                "total_distance": lambda m: sum(r.distance_travelled
                                                for r in m.robots),
                "energy_travel": lambda m: sum(r.energy_travel for r in m.robots),
                "energy_climb": lambda m: sum(r.energy_climb for r in m.robots),
                "energy_dig": lambda m: sum(r.energy_dig for r in m.robots),
                "tasks_dropped": lambda m: sum(r.tasks_dropped for r in m.robots),
            },
            agenttype_reporters={
                ExcavatorRobot: {
                    "battery": "battery",
                    "payload": "payload",
                    "energy_used": "energy_used",
                    "idle_ticks": "idle_ticks",
                    "tasks_completed": "tasks_completed",
                    "distance_travelled": "distance_travelled",
                    "energy_travel": "energy_travel",
                    "energy_climb": "energy_climb",
                    "energy_dig": "energy_dig",
                    "metres_climbed": "metres_climbed",
                    "wait_ticks": "wait_ticks",
                    # Phase 2: without these you cannot tell whether the
                    # large machines are hoarding tasks or sitting idle.
                    # Callables, not attribute names, so this works
                    # regardless of what properties robot.py exposes.
                    "robot_class": lambda a: a.spec.name,
                    "payload_capacity": lambda a: a.spec.capacity,
                },
            },
        )

        self.datacollector.collect(self)

    # -------------------------------------------------------------------- #
    def step(self) -> None:
        self.tick += 1
        self.dynamics.step(self.tick)
        self.comms.flush_and_deliver(self.tick)    # in-flight messages land

        # Sense BEFORE bidding: a bid priced on a stale occupancy map is
        # a bid the robot cannot execute, which breaks the
        # bid == execution invariant the whole cost model rests on.
        for r in self.robots:
            r.sense()

        self._leg_cache = {}                       # fresh per tick
        self.allocator.allocate(self)              # bidding + consensus
        self.agents.shuffle_do("step")             # execute (random order)
        # Every bundle-based agent a robot happens to carry, so the figure
        # is not silently 0 on a moa-cbba run and misread as "this is
        # CBAA". getattr, not attribute access: this metric runs on every
        # tick of every run regardless of the allocator in use, so a
        # robot built without one of these agents must not take the whole
        # simulation down for the sake of a logging number.
        self._max_bundle_seen = max(
            [self._max_bundle_seen]
            + [len(getattr(r, attr).bundle)
               for r in self.robots
               for attr in ("CBBA", "MOACBBA")
               if getattr(r, attr, None) is not None])
        self.changed_cells.clear()
        self.datacollector.collect(self)

        # This is the stopping condition of the whole simulation
        if self.tasks.all_done:
            self.running = False  # lets mesa.batch_run stop early

    def run(self, max_ticks: int = 5000) -> None:
        """Run to completion.

        The loop condition used to be `self.tasks.unfinished`, i.e.
        remaining volume, while step() stops on all_done, i.e. volume
        removed AND hauled AND stamped. So the run ended the moment the
        last cell was empty, leaving robots mid-haul and the final tasks
        with completed_tick = None -- which the makespan property then
        skipped. Every reported makespan was short by roughly one dump
        trip, and by more for the allocators that finished with several
        robots still loaded."""
        while not self.tasks.all_done and self.tick < max_ticks:
            self.step()

    # -------------------------------------------------------------------- #
    @property
    def makespan(self) -> int:
        """max_i busy time; equals last task completion for a full run."""
        done = [t.completed_tick for t in self.tasks.all
                if t.completed_tick is not None]
        return max(done) if done else self.tick

    def current_objective(self) -> float:
        """J(x) as in Eq. (1) — the primary comparison metric."""
        return objective(self.makespan,
                         sum(r.energy_used for r in self.robots),
                         self.w1, self.w2)

    #### COULD BE DEPRECATED BECAUSE DUMP_WORK_PATH EXISTS
    def dump_work_cell(self, coord: Coord,
                       occupied: set[Coord] | None = None,
                       blocked: set[Coord] | None = None):
        """Nearest unload position: a traversable cell adjacent to any
        dump block, chosen deterministically. Returns (cell, dist) or
        None. Pass `occupied` to exclude/avoid other robots; omit it for
        uncontended cost estimates (bids).

        `blocked` lets a caller plan on ITS OWN map. Without it the haul
        leg was planned omnisciently while the approach leg used the
        robot's sensed map, so a robot was blind to hazards on the way to
        the dig site but knew every hazard on the way to the dump -- and
        the haul is the larger leg, walked n = ceil(V/C) times."""
        best = None
        known = self.blocked_cells() if blocked is None else blocked
        for block in self.dump_blocks:
            found = nearest_work_cell(coord, block, self.grid.width,
                                      self.grid.height, known, occupied)
            if found and (best is None or found[1] < best[1]):
                best = found
        return best
    
    def dump_work_path(self, coord: Coord,
                       occupied: set[Coord] | None = None,
                       blocked: set[Coord] | None = None):
        """Like dump_work_cell, but also returns the route, so callers
        can measure elevation gain. Returns (cell, dist, path) or None."""
        best = None
        known = self.blocked_cells() if blocked is None else blocked
        for block in self.dump_blocks:
            found = nearest_work_path(coord, block, self.grid.width,
                                      self.grid.height, known, occupied)
            if found and (best is None or found[1] < best[1]):
                best = found
        return best
    
    def _static_blocked(self) -> set[Coord]:
        """Permanently impassable cells: bedrock and dump blocks.

        Cached. Digging removes VOLUME, never terrain TYPE, so this set
        is fixed after setup -- but it was being rebuilt with a numpy
        `where` scan on every blocked_cells() call, which is several
        times per A* and thousands of times per auction round. Call
        invalidate_static_blocked() if terrain types ever become
        mutable (e.g. if bedrock is added at runtime)."""
        if self._static_blocked_cache is None:
            t = self.grid.terrain.data
            xs, ys = np.where((t == int(Terrain.BEDROCK))
                              | (t == int(Terrain.DUMP_SITE)))
            self._static_blocked_cache = set(zip(xs.tolist(), ys.tolist()))
        return self._static_blocked_cache

    def invalidate_static_blocked(self) -> None:
        self._static_blocked_cache = None

    def blocked_cells(self) -> set[Coord]:
        """Impassable cells: bedrock and dump blocks. Phase 4 adds
        hazards and dynamic obstacles here.
        
        A cell a robot currently OCCUPIES is never reported as blocked,
        even if a hazard grew over it — otherwise a robot caught inside a
        new zone could never path out. It can leave; it just can't be
        newly routed in"""

        occupied = {r.cell.coordinate for r in self.robots}
        return (self._static_blocked() | self.dynamics.blocked()) - occupied

    def _place_dump_block(self) -> None:
        """Stamp a 2x2 DUMP_SITE block at a random free corner."""
        w, h = self.grid.width, self.grid.height
        for _ in range(500):
            cx = self.random.randrange(w - 1)
            cy = self.random.randrange(h - 1)
            block = ((cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1))
            if any(self.grid.terrain.data[c] == int(Terrain.DUMP_SITE)
                   for c in block):
                continue
            for c in block:
                self.grid.terrain.data[c] = int(Terrain.DUMP_SITE)
            self.dump_blocks.append(block)
            self._static_blocked_cache = None
            return
        raise RuntimeError("could not place dump block")

    # -------------------------------------------------------------------- #
    def _scatter_terrain(self, rock_frac: float, gravel_frac: float) -> None:
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                u = self.random.random()
                if u < rock_frac:
                    self.grid.terrain.data[x, y] = int(Terrain.ROCK)
                elif u < rock_frac + gravel_frac:
                    self.grid.terrain.data[x, y] = int(Terrain.GRAVEL)

    def _scatter_bedrock(self, n_ridges: int, length: int) -> None:
        """Impassable ridges, added one at a time and rolled back if they
        would cut the map in two.

        A disconnected map is not a harder instance, it is a broken one:
        tasks behind a wall are unreachable, robots stall forever and the
        run never terminates. Each ridge is therefore committed only if
        every free cell is still reachable from every other.
        """
        if n_ridges <= 0 or length <= 0:
            return
        w, h = self.grid.width, self.grid.height
        for _ in range(n_ridges):
            x, y = self.random.randrange(w), self.random.randrange(h)
            dx, dy = self.random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
            placed = []
            for _ in range(length):
                if not (0 <= x < w and 0 <= y < h):
                    break
                if self.grid.terrain.data[x, y] != int(Terrain.BEDROCK):
                    placed.append((x, y))
                    self.grid.terrain.data[x, y] = int(Terrain.BEDROCK)
                x, y = x + dx, y + dy
            if placed and not self._free_space_connected():
                for cx, cy in placed:       # roll the whole ridge back
                    self.grid.terrain.data[cx, cy] = int(Terrain.SOIL)

    def _free_space_connected(self) -> bool:
        """Flood fill over non-bedrock cells; True if they are one
        component. Dump blocks are placed later and are only 2x2, so
        bedrock is the only thing that can realistically sever the map."""
        w, h = self.grid.width, self.grid.height
        rock = int(Terrain.BEDROCK)
        free = [(x, y) for x in range(w) for y in range(h)
                if self.grid.terrain.data[x, y] != rock]
        if not free:
            return False
        seen = {free[0]}
        stack = [free[0]]
        while stack:
            cx, cy = stack.pop()
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    n = (cx + ddx, cy + ddy)
                    if (0 <= n[0] < w and 0 <= n[1] < h and n not in seen
                            and self.grid.terrain.data[n[0], n[1]] != rock):
                        seen.add(n)
                        stack.append(n)
        return len(seen) == len(free)

    def _scatter_elevation(self, octaves: int, persistence: float, height_scale: float) -> None:
        if height_scale <= 0.0:
            return
        w, h = self.grid.width, self.grid.height
        res, lacunarity = 2, 2 # fractal parameters
        block = res * lacunarity ** (octaves - 1) # Required noise size
        
        # Round up to the nearest multiple of "block"
        pw = -(-w // block) * block
        ph = -(-h // block) * block

        field = generate_fractal_noise_2d((pw, ph), (res, res), octaves=octaves, persistence=persistence, lacunarity=lacunarity, rng=self.rng)[:w, :h]
        field = field - field.min() # shift so minimum = 0
        peak = field.max()

        # normalise
        if peak > 0:
            field = field / peak
        
        self.grid.elevation.data[:, :] = height_scale * field


    def claimed_work_cells(self, exclude=None, observer=None) -> set[Coord]:
        """Work cells other robots have already committed to. Passed to
        nearest_work_cell at assignment so two robots sharing a site do
        not both target the same dig position.

        `observer` restricts the answer to what that robot could actually
        learn. A committed work cell is an INTENTION, not an object, so
        it cannot be seen -- it has to be told. The filter is therefore
        comm range, not sensor range: you see where a machine IS, you
        hear where it is GOING. With comm_range=None this is unrestricted
        and the old behaviour is reproduced exactly.
        """
        others = (self.robots if observer is None
                  else [r for r in self.robots if r is not observer]
                  if self.comms.comm_range is None
                  else self.comms.neighbors(observer))
        return {r.work_cell for r in others
                if r is not exclude and r.work_cell is not None}

    def _random_empty_coord(self) -> Coord:
        forbidden = (int(Terrain.DUMP_SITE), int(Terrain.BEDROCK))
        while True:
            c = (self.random.randrange(self.grid.width),
                 self.random.randrange(self.grid.height))
            if not self.grid[c].agents \
                    and int(self.grid.terrain.data[c]) not in forbidden:
                return c

    def _random_diggable_coord(self) -> Coord:
        while True:
            c = self._random_empty_coord()
            if self.grid.terrain.data[c] != int(Terrain.DUMP_SITE):
                # force diggable terrain at task cells
                if self.grid.terrain.data[c] == int(Terrain.BEDROCK):
                    continue
                return c