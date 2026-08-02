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
from .comms import CommNetwork
from .costs import RobotSpec, objective
from .pathfinding import nearest_work_cell, nearest_work_path
from .robot import ExcavatorRobot
from .tasks import TaskRegistry
from .terrain import T_UNLOAD, Terrain
from .dynamics import Coord, DynamicsManager

DEFAULT_SPEC = RobotSpec(capacity=1.5, v_max=1.0, dig_rate=0.5,
                         battery=100.0, sensor_range=8.0)


class TaskMarker(FixedAgent):
    """Passive agent standing on a task cell so the GUI draws tasks
    through the standard agent pipeline. Removes itself on completion.
    No physics: it never moves, digs, or spends energy."""

    def __init__(self, model, cell, task):
        super().__init__(model)
        self.cell = cell
        self.task = task

    def step(self) -> None:
        if self.task.done:
            self.remove()


class ExcavationModel(Model):
    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        n_robots: int = 4,
        n_tasks: int = 8,
        allocator: str = "cbpae",
        w1: float = 1.0,
        w2: float = 1.0,
        comm_range: float | None = None,
        packet_loss: float = 0.0,
        comm_latency: int = 0,
        comm_bandwidth: int | None = None,

        hazard_rate: float = 0.05,          # 4.2 zones/tick (0 = off)
        obstacle_rate: float = 0.3,        # 4.3 obstacles/tick (0 = off)
        weather_enabled: bool = False,     # 4.4 weather on/off
        weather_change_rate: float = 0.0,  # 4.4 transitions/tick
        hazard_duration: int = 5, hazard_size: int = 2,
        obstacle_duration: int = 60, obstacle_move_prob: float = 0.5,
        max_obstacles: int = 10,

        rock_fraction: float = 0.15,
        gravel_fraction: float = 0.2,

        elevation_scale: float = 15.0,
        elevation_octaves: int = 4,  # How detailed
        elevation_persistence: float = 0.5,

        task_volume: tuple[float, float] = (1.0, 4.0),
        seed: int | None = None,
    ):
        super().__init__(rng=int(seed) if seed is not None else None)
        self.w1, self.w2 = w1, w2
        self.t_unload = T_UNLOAD
        self.tick = 0
        self.changed_cells: set[Coord] = set()  # Algorithm 1, line 3 hook

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
        self._scatter_elevation(elevation_octaves, elevation_persistence, elevation_scale)

        # --- dump sites: 2x2 impassable blocks --------------------------- #
        self.dump_blocks: list[tuple[Coord, ...]] = []
        while len(self.dump_blocks) < 2:
            self._place_dump_block()

        # --- tasks ------------------------------------------------------ #
        self.tasks = TaskRegistry()
        lo, hi = task_volume
        for _ in range(n_tasks):
            cell = self._random_diggable_coord()
            vol = self.random.uniform(lo, hi)
            self.grid.soil_volume.data[cell] += vol
            task = self.tasks.add(cell, vol)
            TaskMarker(self, self.grid[cell], task)
        # print(len(self.tasks.all), "tasks placed")
        # --- robots ----------------------------------------------------- #
        self.robots: list[ExcavatorRobot] = []
        for _ in range(n_robots):
            cell = self.grid[self._random_empty_coord()]
            self.robots.append(ExcavatorRobot(self, cell, DEFAULT_SPEC))

        # Populating the list of task for each robot to prepare for CBBA
        for robot in self.robots:
            robot.updateTaskList(self.tasks.all)
        
        # --- Phase 6 communication layer (neutral by default) ----------- #
        # Change comm_range to a number so that the s vector is utilised

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
            },
            agenttype_reporters={
                ExcavatorRobot: {
                    "battery": "battery",
                    "payload": "payload",
                    "energy_used": "energy_used",
                    "idle_ticks": "idle_ticks",
                    "tasks_completed": "tasks_completed",
                },
            },
        )

        self.dynamics = DynamicsManager(
            self, hazard_rate=hazard_rate, obstacle_rate=obstacle_rate,
            weather_enabled=weather_enabled,
            weather_change_rate=weather_change_rate,
            hazard_duration=hazard_duration, hazard_size=hazard_size,
            obstacle_duration=obstacle_duration,
            obstacle_move_probability=obstacle_move_prob,
            max_obstacles=max_obstacles)
        self.datacollector.collect(self)

    # -------------------------------------------------------------------- #
    def step(self) -> None:
        self.tick += 1
        self.dynamics.step(self.tick)
        self.comms.flush_and_deliver(self.tick)    # in-flight messages land

        self.allocator.allocate(self)              # bidding + consensus (Greedy allocation)
        self.agents.shuffle_do("step")             # execute (random order)
        self.changed_cells.clear()
        self.datacollector.collect(self)

        for r in self.robots:
            if r.robot_id == 0 and r.task_id is None and len(self.tasks.pending) > 0:
                j = r.CBPAE.bidTask
                print(f"R0 idle w/ work: bid={j} "
                    f"winner={r.CBPAE._winner(j) if j is not None else '-'} "
                    f"reachable={len(r.CBPAE.biddable_tasks(self, r))} "
                    f"pending={len(self.tasks.pending)}")

        # This is the stopping condition of the whole simulation
        if len(self.tasks.isAllTaskDone) == len(self.tasks._tasks):
            self.running = False  # lets mesa.batch_run stop early

    def run(self, max_ticks: int = 5000) -> None:
        while self.tasks.unfinished and self.tick < max_ticks:
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
                       occupied: set[Coord] | None = None):
        """Nearest unload position: a traversable cell adjacent to any
        dump block, chosen deterministically. Returns (cell, dist) or
        None. Pass `occupied` to exclude/avoid other robots; omit it for
        uncontended cost estimates (bids)."""
        best = None
        for block in self.dump_blocks:
            found = nearest_work_cell(coord, block, self.grid.width,
                                      self.grid.height,
                                      self.blocked_cells(), occupied)
            if found and (best is None or found[1] < best[1]):
                best = found
        return best
    
    def dump_work_path(self, coord: Coord,
                       occupied: set[Coord] | None = None):
        """Like dump_work_cell, but also returns the route, so callers
        can measure elevation gain. Returns (cell, dist, path) or None."""
        best = None
        for block in self.dump_blocks:
            found = nearest_work_path(coord, block, self.grid.width,
                                      self.grid.height,
                                      self.blocked_cells(), occupied)
            if found and (best is None or found[1] < best[1]):
                best = found
        return best
    
    def _static_blocked(self) -> set[Coord]:
        # Permanently impassable cells: bedrock and dumpblocks
        t = self.grid.terrain.data
        xs, ys = np.where((t == int(Terrain.BEDROCK)) | (t == int(Terrain.DUMP_SITE)))
        return set(zip(xs.tolist(), ys.tolist()))

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
