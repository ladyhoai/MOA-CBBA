"""Excavation tasks t_j: a target cell l_j and a target volume V_j.

ONE TASK PER SITE, BUT POSSIBLY SEVERAL ROBOTS PER TASK. Static
decomposition (pre-splitting a site into chunks) is gone -- a site is
one indivisible task. What replaced it is genuine concurrent sharing:
`assignees` is a SET, so k robots can work the same task at the same
time and the volume drains k times faster.

That is a deliberate move from ST-SR to ST-MR in the Gerkey & Mataric
taxonomy, and it should be stated as such in the write-up. It is also
opt-in per allocator: CBBA and CBPAE filter on "is anybody on this
task", so they keep putting exactly one robot on each task and remain
ST-SR. Only MOA-CBBA fills more than one seat.

New-reader primer: a Task is just "a pile of dirt at a grid cell, with a
volume that shrinks as robots dig it". TaskRegistry is the master list
of every pile in the current run -- allocators ask it "what work is
still available" (`pending` / `unfinished` / `seats_open`) and robots
ask it "give me task #7" (`get`). ST-SR/ST-MR is shorthand from robotics
task-allocation theory: Single-Task-robot vs Multi-Robot, i.e. "can one
job have more than one robot on it at once".
"""

from __future__ import annotations

from dataclasses import dataclass, field

Coord = tuple[int, int]


@dataclass
class Task:
    """One excavation site. Exactly one robot per Task."""

    task_id: int
    cell: Coord            # l_j
    volume: float          # V_j, the full size of the pile at creation
    # `remaining` has init=False: it isn't passed in by the caller, it's
    # computed once in __post_init__ (below) and then drained tick by
    # tick as robots dig. `volume` stays fixed as the original total, so
    # "how much has been dug so far" is always `volume - remaining`.
    remaining: float = field(init=False)
    assignees: set[int] = field(default_factory=set)   # robot_ids at work
    completed_tick: int | None = None   # tick this task was fully delivered, or None

    def __post_init__(self) -> None:
        # Runs automatically right after the auto-generated __init__.
        self.remaining = self.volume

    @property
    def done(self) -> bool:
        """True once the pile is dug out. NOTE: "dug out" is not the same
        as "delivered" -- the last robot may still be hauling. The
        stronger condition (dug AND hauled AND stamped) is
        TaskRegistry.all_done, which is what ends a run. See
        robot._stamp_if_complete for why the two differ.

        1e-9 rather than == 0.0 because `remaining` is repeatedly
        decremented by floating-point dig amounts and will not land
        exactly on zero.

        READ BY: nearly everything -- robot._dig_tick, the allocators'
        prune/task-list filters, and TaskRegistry's own queries below.
        """
        return self.remaining <= 1e-9

    @property
    def assigned_to(self) -> int | None:
        """Read-only compatibility view: the lowest-id robot on this
        task, or None if nobody is. Every ST-SR call site asks one of
        two questions -- "is anyone on it" or "is it me" -- and both
        still answer correctly, because those allocators never seat more
        than one robot. Writes go through add_assignee / drop_assignee,
        so nothing can silently overwrite a co-worker."""
        return min(self.assignees) if self.assignees else None

    @property
    def sharers(self) -> int:
        """How many robots are currently seated on this task (k in the
        MOA-CBBA sharing math: the volume drains k times faster, and each
        robot is priced for roughly V/k of it -- see
        MOACBBAAgent.pathCost)."""
        return len(self.assignees)

    def add_assignee(self, robot_id: int) -> None:
        """Seat a robot on this task. CALLED BY: robot.assign, the single
        place a robot commits to a task."""
        self.assignees.add(robot_id)

    def drop_assignee(self, robot_id: int) -> None:
        """Free a robot's seat. `discard` (not `remove`) so dropping a
        robot that was never seated is a no-op rather than an error.

        CALLED BY: robot._finish_task (finished it) and
        robot._release_task (gave it up unfinished)."""
        self.assignees.discard(robot_id)


class TaskRegistry:
    """Holds all tasks; the source of truth behind the BAM x_ij.

    "BAM" = Binary Assignment Matrix, the x_ij of Eq. (1): x_ij = 1 when
    robot i is assigned task j. Rather than storing that matrix
    explicitly, it is derived -- each Task's `assignees` set is one
    column of it. One registry per model, as model.tasks.
    """

    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 0

    # ---------------------------------------------------------------- #
    def add(self, cell: Coord, volume: float) -> Task:
        """Create and register a task, auto-assigning the next id.

        CALLED BY: model.ExcavationModel.__init__ during world setup,
        once per requested task."""
        t = Task(self._next_id, cell, volume)
        self._tasks[t.task_id] = t
        self._next_id += 1
        return t

    def get(self, task_id: int) -> Task:
        """Look up a task by id. The most-called method in the codebase:
        allocators and robots pass task IDS around (they are cheap to put
        in a message) and resolve them to objects here."""
        return self._tasks[task_id]

    def at_cell(self, cell: Coord) -> list[Task]:
        """Every task located at a grid cell. A list because the data
        model does not forbid two tasks sharing a cell, though the model
        never creates them that way."""
        return [t for t in self._tasks.values() if t.cell == cell]

    # ---------------------------------------------------------------- #
    @property
    def all(self) -> list[Task]:
        """Every task ever created, finished or not."""
        return list(self._tasks.values())

    @property
    def pending(self) -> list[Task]:
        """Unfinished with NOBODY on it -- the open column set of the BAM
        for the single-assignment allocators.

        CALLED BY: GreedyAllocator.allocate. CBBA/CBPAE build equivalent
        filters inline; MOA-CBBA uses seats_open instead, since it can
        seat more than one robot per task."""
        return [t for t in self._tasks.values()
                if not t.done and not t.assignees]

    def seats_open(self, robot_id: int, max_sharers: int = 1) -> list[Task]:
        """Tasks this robot may bid on when several robots per task are
        allowed: unfinished, and either it is already seated or a seat is
        free. `pending` is the max_sharers=1 case of this.

        Including tasks the robot is ALREADY on is what lets MOA-CBBA
        express an en-route switch as an ordinary auction outcome rather
        than a special case.

        CALLED BY: nothing, currently. MOA-CBBA used to build its
        candidate lists here and no longer does: this method answers from
        the registry, i.e. from fleet-wide occupancy no robot could have
        observed, so it has been replaced by belief.WorldBelief.candidates
        which applies the SAME predicate to what one robot has sensed and
        been told. Kept as the reference definition of the predicate, and
        because a centralised allocator would legitimately want it."""
        return [t for t in self._tasks.values()
                if not t.done and (robot_id in t.assignees
                                   or len(t.assignees) < max_sharers)]

    @property
    def unfinished(self) -> list[Task]:
        """Tasks with volume still in the ground, regardless of who is on
        them. The allocators' "is there any work left at all" check."""
        return [t for t in self._tasks.values() if not t.done]

    @property
    def all_done(self) -> bool:
        """Every task finished AND stamped -- the simulation's stopping
        condition.

        Deliberately stricter than `all(t.done)`: completed_tick is only
        stamped once the last robot leaves the task, which means the
        material has actually reached a dump site rather than merely
        being out of the ground. Without that, a run could stop while
        robots were still mid-haul and the reported makespan would be
        short by a full dump trip. See robot._stamp_if_complete.

        CALLED BY: model.step (sets running=False), model.run (loop
        condition) and batch30.run_one."""
        return all(t.done and t.completed_tick is not None
                   for t in self._tasks.values())