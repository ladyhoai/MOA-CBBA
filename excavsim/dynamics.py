"""Phase 4: everything that makes the world change WHILE the simulation
runs, instead of being fixed at setup.

New-reader primer: three independent kinds of change live here, all
driven by simple per-tick dice rolls (`rng.random() < rate`):
  - HazardZone  -- a temporary blob of impassable cells (like a cave-in
    or a flooded patch) that appears, sits still for a fixed duration,
    then disappears.
  - Obstacle    -- a single impassable cell that appears, randomly
    wanders one step at a time (like another vehicle or a person), and
    eventually disappears.
  - Weather     -- a global mode ("clear"/"rain"/"fog"/"storm") that
    scales how far robots can sense (sensor_scale) and how much extra
    energy climbing costs (traction_scale).
DynamicsManager.step() is called once per tick from model.py and is the
only place any of this actually changes; everything else in the file is
either state (the dataclasses below) or read-only queries robots and the
allocator use to find out what's currently blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import ExcavationModel


Coord = tuple[int, int]

# One named weather mode and the two multipliers it applies. Looked up
# by name (self.weather) rather than stored as an object on the manager,
# so the "current weather" is always just a short string.
@dataclass(frozen=True)
class WeatherState:
    name:str
    sensor_scale: float     # multiplies robot.sensor_radius (see robot.py)
    traction_scale: float   # multiplies climb energy cost (see costs.py)

WEATHER = {
    "clear": WeatherState("clear", 1.0, 1.0),
    "rain": WeatherState("rain", 0.8, 1.4),
    "fog": WeatherState("fog", 0.4, 1.0),
    "storm": WeatherState("storm", 0.3, 1.6)
}

@dataclass
class HazardZone:
    """A blob of grid cells that is impassable until `expires_tick`."""
    cells: frozenset[Coord]
    expires_tick: int

@dataclass
class Obstacle:
    """A single impassable, wandering cell, alive until `expires_tick`."""
    cell: Coord
    expires_tick: int

@dataclass
class DynamicsManager: # This class will be instantiated by the model
    """Owns every hazard/obstacle/weather-state and advances them each
    tick. One instance lives on ExcavationModel as `model.dynamics`."""
    model: ExcavationModel
    # Hazard
    hazard_rate: float = 0.0 # probability of a new zone appearing every tick
    hazard_size: int = 3
    hazard_duration: int = 40 # the zone will stay active for 40 ticks
    # Obstacles
    obstacle_rate: float = 0.0
    obstacle_duration: int = 60
    obstacle_move_probability: float = 0.5 # chance for the obstacle to step each tick
    max_obstacles: int = 6
    # Weather
    weather_enabled: bool = False
    weather_change_rate: float = 0.0 # probability for the weather to change every tick

    hazards: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    weather: str = "clear"

    def hazard_cells(self) -> set[Coord]:
        """Zone cells only. These STAY PUT until they expire, so a
        sighting keeps its value for as long as the zone lives."""
        out: set[Coord] = set()
        for hazard in self.hazards:
            out |= hazard.cells
        return out

    def obstacle_cells(self) -> set[Coord]:
        """Wandering obstacles only. Each one steps with probability
        obstacle_move_probability EVERY tick, so a sighting is stale
        almost immediately -- which is why the robot ages these out far
        faster than hazards, and why their NEIGHBOURHOOD matters more
        than the cell they were last seen in."""
        return {o.cell for o in self.obstacles}

    # Returning all blocked cells
    def blocked(self) -> set[Coord]:
        """Every DYNAMICALLY blocked cell: hazard zones plus obstacles.

        Static blockage (bedrock, dump blocks) is not included -- that
        comes from model._static_blocked. model.blocked_cells() unions
        the two for the full picture.

        CALLED BY: model.blocked_cells, robot._advance_along_path (the
        physical "can I actually move there" test) and the GUI overlays.
        """
        out: set[Coord] = set()
        for hazard in self.hazards:
            out |= hazard.cells
        for obstacles in self.obstacles:
            out.add(obstacles.cell)
        return out

    def sensor_scale(self) -> float:
        """Weather multiplier on sensing range (1.0 when weather is off).

        MATH: robot.sensor_radius = spec.sensor_range * sensor_scale().
        Fog scales to 0.4 and storm to 0.3, so a small machine
        (sigma = 6) sees barely two cells in a storm while a large one
        (sigma = 10) still sees three.

        CALLED BY: robot.sensor_radius, consulted every time a robot
        senses or plans.
        """
        return WEATHER[self.weather].sensor_scale if self.weather_enabled else 1.0

    def traction_scale(self) -> float:
        """Weather multiplier on CLIMB energy (1.0 when weather is off).

        MATH: appears as the `traction` factor in costs.energy_ij's
        GAMMA * traction * climb terms, and in robot._spend_move. Rain
        (1.4) and storm (1.6) make uphill travel more expensive; fog does
        not (0.4 sensor, 1.0 traction -- it hides the ground, it does not
        make it slippery).

        CALLED BY: bidding.leg_cost (so bids price current weather) and
        robot._spend_move (so execution charges it).
        """
        return WEATHER[self.weather].traction_scale if self.weather_enabled else 1.0
    
    def step(self, now: int) -> None:
        """Advance the world by one tick: expire old hazards/obstacles,
        maybe spawn new ones, move existing obstacles, maybe change the
        weather. Called once per tick from ExcavationModel.step(),
        before robots sense or bid, so everyone reacts to this tick's
        world, not last tick's."""
        rng = self.model.random
        changed = self.model.changed_cells

        # removing expired hazards
        for hz in [h for h in self.hazards if now > h.expires_tick]:
            changed |= hz.cells
            self.hazards.remove(hz)
        
        # spawning new hazards based on probability
        if self.hazard_rate > 0 and rng.random() < self.hazard_rate:
            cells = self._random_zone(rng)
            if cells:
                self.hazards.append(HazardZone(frozenset(cells), now + self.hazard_duration))
                changed |= cells
        # remove expired obstacle
        for ob in [o for o in self.obstacles if now > o.expires_tick]:
            changed.add(ob.cell)
            self.obstacles.remove(ob)
        
        # moving the obstacle based on probability
        for ob in self.obstacles:
            if rng.random() < self.obstacle_move_probability:
                nxt = self._wander(ob.cell, rng)
                if nxt != ob.cell:
                    changed.add(ob.cell)
                    changed.add(nxt)
                    ob.cell = nxt
        
        if (self.obstacle_rate > 0 and len(self.obstacles) < self.max_obstacles and rng.random() < self.obstacle_rate):
            c = self._free_cell(rng)
            if c is not None:
                self.obstacles.append(Obstacle(c, now + self.obstacle_duration))
                changed.add(c)

        # Weather transitions
        if self.weather_enabled and self.weather_change_rate > 0 and rng.random() < self.weather_change_rate:
            self.weather = rng.choice(list(WEATHER.keys()))

    #---------------------------HELPER FUNCTIONS--------------------------------#
    def _random_zone(self, rng) -> set[Coord]:
        """Pick the cells for a new hazard zone: a square blob centred on
        a random cell.

        MATH: every cell within +/- hazard_size on each axis, i.e. a
        (2r+1) x (2r+1) square -- 25 cells at the default r=2. Cells that
        are permanently blocked or currently occupied by a robot are
        excluded, so a hazard can never spawn on top of a machine or
        double-block bedrock.

        CALLED BY: step, when the per-tick hazard_rate roll succeeds.
        """
        w, h = self.model.grid.width, self.model.grid.height
        cx, cy = rng.randrange(w), rng.randrange(h)
        r = self.hazard_size
        base = self.model._static_blocked() # This will be implemented in model.py
        occupied = {rb.cell.coordinate for rb in self.model.robots}
        cells = set()
        for dx in range (-r, r + 1):
            for dy in range (-r, r + 1):
                c = (cx + dx, cy + dy)
                # check if the cell is withing constraints
                if 0 <= c[0] < w and 0 <= c[1] < h and c not in base and c not in occupied:
                    cells.add(c)
        return cells
    
    # Moving the dynamic obstacle
    def _wander(self, cell:Coord, rng) -> Coord:
        """One random-walk step for an obstacle: pick uniformly among the
        free cells of its 8-neighbourhood, or stay put if boxed in.

        MATH: this is what makes an obstacle's position a random walk.
        Combined with the caller's obstacle_move_probability p = 0.5,
        the chance any ONE specific neighbouring cell is entered next
        tick is p / |options| ~= 0.5/8 = 6.25%, and expected displacement
        after t ticks is about sqrt(p*t) per axis. Those two numbers are
        exactly why robot.py keeps obstacle sightings for only
        OBSTACLE_TTL = 3 ticks and draws its predictive "halo" just one
        cell wide -- see the arithmetic in robot.py's HALO_HORIZON note.

        CALLED BY: step, once per obstacle whose move roll succeeded.
        """
        w, h = self.model.grid.width, self.model.grid.height
        blocked = self.model.blocked_cells()
        occ = {rb.cell.coordinate for rb in self.model.robots}
        others = {o.cell for o in self.obstacles if o.cell != cell}
        # checking for possible move directions
        options = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                c = (cell[0] + dx, cell[1] + dy)
                if 0 <= c[0] < w and 0 <= c[1] < h and c not in blocked and c not in occ and c not in others:
                    options.append(c)
        # randomly pick from a list of possible move options
        return rng.choice(options) if options else cell

    def _free_cell(self, rng) -> Coord | None:
        """A random cell suitable for spawning a new obstacle, or None.

        Rejection sampling: try up to 60 random cells and take the first
        that is not blocked, not under a robot, and not an unfinished
        task cell (an obstacle sitting on a task would make it
        permanently undiggable). Giving up after 60 tries rather than
        looping forever means a nearly-full map simply spawns nothing
        that tick.

        CALLED BY: step, when the obstacle_rate roll succeeds and the
        max_obstacles cap has not been reached.
        """
        w, h = self.model.grid.width, self.model.grid.height
        blocked = self.model.blocked_cells()
        occ = {rb.cell.coordinate for rb in self.model.robots}
        tasks = {t.cell for t in self.model.tasks.unfinished}
        # return a random free cell
        for _ in range(60):
            c = (rng.randrange(w), rng.randrange(h))
            if c not in blocked and c not in occ and c not in tasks:
                return c
        return None