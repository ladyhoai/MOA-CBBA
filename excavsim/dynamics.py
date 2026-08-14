from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import ExcavationModel


Coord = tuple[int, int]

# Weather
@dataclass(frozen=True)
class WeatherState:
    name:str
    sensor_scale: float
    traction_scale: float

WEATHER = {
    "clear": WeatherState("clear", 1.0, 1.0),
    "rain": WeatherState("rain", 0.8, 1.4),
    "fog": WeatherState("fog", 0.4, 1.0),
    "storm": WeatherState("storm", 0.3, 1.6)
}

@dataclass
class HazardZone:
    cells: frozenset[Coord]
    expires_tick: int

@dataclass
class Obstacle:
    cell: Coord
    expires_tick: int

@dataclass
class DynamicsManager: # This class will be instantiated by the model
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
        out: set[Coord] = set()
        for hazard in self.hazards:
            out |= hazard.cells
        for obstacles in self.obstacles:
            out.add(obstacles.cell)
        return out
    
    def sensor_scale(self) -> float:
        return WEATHER[self.weather].sensor_scale if self.weather_enabled else 1.0
    
    def traction_scale(self) -> float:
        return WEATHER[self.weather].traction_scale if self.weather_enabled else 1.0
    
    def step(self, now: int) -> None:
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