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

from mesa import DataCollector, Model
from mesa.discrete_space import (FixedAgent, OrthogonalMooreGrid,
                                 PropertyLayer)

from .allocation import ALLOCATORS
from .costs import RobotSpec, objective
from .pathfinding import nearest_work_cell
from .robot import ExcavatorRobot
from .tasks import TaskRegistry
from .terrain import T_UNLOAD, Terrain

Coord = tuple[int, int]

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
        allocator: str = "greedy",
        w1: float = 1.0,
        w2: float = 1.0,
        rock_fraction: float = 0.15,
        gravel_fraction: float = 0.2,
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

        # --- robots ----------------------------------------------------- #
        self.robots: list[ExcavatorRobot] = []
        for _ in range(n_robots):
            cell = self.grid[self._random_empty_coord()]
            self.robots.append(ExcavatorRobot(self, cell, DEFAULT_SPEC))

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
        self.datacollector.collect(self)

    # -------------------------------------------------------------------- #
    def step(self) -> None:
        self.tick += 1
        self.allocator.allocate(self)              # bidding + consensus
        self.agents.shuffle_do("step")             # execute (random order)
        self.changed_cells.clear()
        self.datacollector.collect(self)

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

    def blocked_cells(self) -> set[Coord]:
        """Impassable cells: bedrock and dump blocks. Phase 4 adds
        hazards and dynamic obstacles here."""
        import numpy as np
        t = self.grid.terrain.data
        xs, ys = np.where((t == int(Terrain.BEDROCK))
                          | (t == int(Terrain.DUMP_SITE)))
        return set(zip(xs.tolist(), ys.tolist()))

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
