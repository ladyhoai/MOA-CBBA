"""Excavation tasks t_j: a target cell l_j and a target volume V_j."""

from __future__ import annotations
 
from dataclasses import dataclass, field
 
Coord = tuple[int, int]
 
# A chunk smaller than this is not worth the extra approach trip.
# NOTE: keep this comfortably above the smallest payload capacity in the
# fleet. If chunks shrink to <= min(C_i) then n_ij = 1 for every robot
# and the capacity heterogeneity becomes invisible to the allocator --
# the same failure mode as pinning n = 1 in costs.py.
MIN_CHUNK = 1.5
 
 
@dataclass
class Task:
    """One chunk. Still exactly one robot per Task -- ST-SR is intact."""
 
    task_id: int
    cell: Coord            # l_j (shared with siblings)
    volume: float          # V_j for THIS chunk, not the whole site
    site_id: int = 0       # which excavation site this chunk belongs to
    site_volume: float = 0.0   # total across all chunks at the site
    remaining: float = field(init=False)
    future_remaining: float = field(init=False)
    assigned_to: int | None = None   # robot unique_id, or None
    completed_tick: int | None = None
 
    def __post_init__(self) -> None:
        self.remaining = self.volume
        self.future_remaining = self.volume
 
    @property
    def done(self) -> bool:
        return self.remaining <= 1e-9
 
 
def chunk_count(volume: float, max_sharers: int, free_work_cells: int,
                min_chunk: float = MIN_CHUNK) -> int:
    """How many robots may share this site.
 
    Capped by three things, and the middle one is the one people forget:
      - max_sharers      : experiment parameter
      - free_work_cells  : robots never share a cell, and work_candidates
                           yields at most 9 positions around a target --
                           fewer once bedrock, dump blocks, hazards and
                           obstacles are subtracted. Over-splitting past
                           this just creates chunks nobody can reach.
      - volume/min_chunk : do not split into slivers.
    """
    if max_sharers <= 1 or volume <= 0.0:
        return 1
    by_volume = int(volume // min_chunk)
    return max(1, min(max_sharers, free_work_cells, by_volume))
 
 
class TaskRegistry:
    """Holds all chunks; the source of truth behind the BAM x_ij."""
 
    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._sites: dict[int, list[int]] = {}   # site_id -> chunk ids
        self._next_id = 0
        self._next_site = 0
 
    # ---------------------------------------------------------------- #
    def add(self, cell: Coord, volume: float, n_chunks: int = 1) -> list[Task]:
        """Create one site as `n_chunks` equal-volume chunks.
 
        Always returns a list, even for n_chunks=1, so callers do not
        branch. n_chunks=1 reproduces the pre-decomposition behaviour
        exactly, which is what the ablation baseline needs.
        """
        n_chunks = max(1, int(n_chunks))
        site_id = self._next_site
        self._next_site += 1
 
        per = volume / n_chunks
        chunks: list[Task] = []
        for _ in range(n_chunks):
            t = Task(self._next_id, cell, per,
                     site_id=site_id, site_volume=volume)
            self._tasks[t.task_id] = t
            chunks.append(t)
            self._next_id += 1
        self._sites[site_id] = [t.task_id for t in chunks]
        return chunks
 
    def get(self, task_id: int) -> Task:
        return self._tasks[task_id]
 
    # ---------------------------------------------------------------- #
    # site-level views (chunks are what the allocator sees; sites are
    # what the operator, the GUI and the terrain care about)
    # ---------------------------------------------------------------- #
    def siblings(self, task_id: int) -> list[Task]:
        """Other chunks of the same site, excluding this one."""
        t = self._tasks[task_id]
        return [self._tasks[i] for i in self._sites[t.site_id] if i != task_id]
 
    def site_chunks(self, site_id: int) -> list[Task]:
        return [self._tasks[i] for i in self._sites[site_id]]
 
    def site_remaining(self, site_id: int) -> float:
        return sum(t.remaining for t in self.site_chunks(site_id))
 
    def site_done(self, site_id: int) -> bool:
        return all(t.done for t in self.site_chunks(site_id))
 
    def site_completed_tick(self, site_id: int) -> int | None:
        """When the LAST chunk of the site finished, or None."""
        ticks = [t.completed_tick for t in self.site_chunks(site_id)]
        return max(ticks) if ticks and all(x is not None for x in ticks) else None
 
    def sharers(self, site_id: int) -> int:
        """How many robots are currently working this site."""
        return sum(1 for t in self.site_chunks(site_id)
                   if t.assigned_to is not None and not t.done)
 
    def at_cell(self, cell: Coord) -> list[Task]:
        return [t for t in self._tasks.values() if t.cell == cell]
 
    @property
    def sites(self) -> list[int]:
        return list(self._sites)
 
    @property
    def n_sites(self) -> int:
        return len(self._sites)
 
    # ---------------------------------------------------------------- #
    @property
    def all(self) -> list[Task]:
        return list(self._tasks.values())
 
    @property
    def pending(self) -> list[Task]:
        """Unfinished and unassigned -- the open column set of the BAM."""
        return [t for t in self._tasks.values()
                if not t.done and t.assigned_to is None]
 
    @property
    def unfinished(self) -> list[Task]:
        return [t for t in self._tasks.values() if not t.done]
 
    @property
    def isAllTaskDone(self) -> list[Task]:
        return [t for t in self._tasks.values()
                if t.done and t.completed_tick is not None]
