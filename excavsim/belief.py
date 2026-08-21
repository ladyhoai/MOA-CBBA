"""Per-robot beliefs about the fleet's DYNAMIC state (Phase 6.5).

WHY THIS MODULE EXISTS. Phase 3 gave each robot a private map of what is
IMPASSABLE (robot._map / known_blocked) and routed all planning and
bidding through it. Everything else an allocator needs to make a
decision -- how much soil is left in a pile, whether a pile is finished,
who is currently digging it, how big the other machines are -- was still
read straight off the shared TaskRegistry, i.e. off the simulator's
ground truth. A robot cannot know those things without being told, so
reading them made the auction only half decentralised: local bid tables
exchanged by radio, sitting on top of omniscient world state.

WorldBelief closes that gap. It is the task-state analogue of
known_blocked: one instance per robot, updated from two sources only --

  FIRST-HAND   what this robot can see right now: the pile it is
               standing in (exact), any pile inside its sensor disc, and
               any machine inside that disc that is parked on a pile.
  GOSSIP       what peers have told it, merged by timestamp in exactly
               the same way MOACBBAAgent.resolveConflicts merges bids.

WHAT IS STILL COMMON KNOWLEDGE, and why that is legitimate: the task SET
(ids, cells l_j and original volumes V_j), terrain type, bedrock and
dump locations. All of it is fixed at setup and never changes -- tasks
are created once in ExcavationModel.__init__ and none are ever added --
so it is a survey handed to the fleet before work starts, which is how
the proposal frames the environment. What changes DURING the run is
exactly what this module makes local:

    remaining volume  |  done  |  who is seated  |  peer capacities

MERGE RULE, per key: newer stamp wins; equal stamps break
deterministically (lower remaining / lower task id) so two robots cannot
disagree forever. Note this is NOT a monotone min-register even though
piles normally only shrink: robot._release_task puts an abandoned
hopper's soil BACK into task.remaining (the conservation guard in
robot.py), so remaining can rise and a stampless "take the smaller"
merge would latch a value that is no longer true.

OMNISCIENT FALLBACK. Every accessor reads through to ground truth when
model.sensing_enabled is False, exactly as known_blocked does. That keeps
the two halves of partial observability on ONE switch: sensing off means
a robot sees every hazard AND every pile the instant it changes, which is
the pre-Phase-3 behaviour and the ablation baseline. Sensing on (the
default) means both are local and the fleet has to talk.

CONSEQUENCE WORTH REPORTING. Beliefs lag, so a robot can bid on a pile
another robot has already emptied, drive there, and find nothing. That is
not a bug -- it is the cost of decentralisation, and it is the thing a
comm_range / packet_loss sweep is supposed to measure.

New-reader primer: think of this as each robot's private notebook about
the job site. It writes down what it sees, copies pages from anyone it
can radio, and plans from the notebook -- never from the site office's
master ledger.
"""

from __future__ import annotations

EPS = 1e-9
NOWHERE = -1        # station value for "this robot is not on any task"


class WorldBelief:
    """One robot's notebook. Owned by MOACBBAAgent, never shared."""

    def __init__(self) -> None:
        # task_id  -> (remaining volume, stamp)
        self.volume: dict[int, tuple[float, int]] = {}
        # robot_id -> (task_id or NOWHERE, stamp)
        self.station: dict[int, tuple[int, int]] = {}
        # robot_id -> hopper capacity C_k (static, so no stamp needed)
        self.capacity: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def omniscient(model) -> bool:
        """True when the run has partial observability switched off, in
        which case every accessor below reads ground truth.

        Mirrors robot.known_blocked's own fallback, deliberately: one
        switch controls both halves of what a robot does not know.

        Also read by MOACBBAAgent.broadcast, which SKIPS the belief
        payload entirely in this mode: nothing reads the tables when
        every accessor short-circuits to ground truth, so gossiping them
        would burn bandwidth and -- because a merge counts as "something
        changed" -- add auction rounds that cannot alter a single bid.
        """
        return not getattr(model, "sensing_enabled", False)

    # ------------------------------------------------------------------ #
    # first-hand observation
    # ------------------------------------------------------------------ #
    def observe(self, model, robot, now: int) -> None:
        """Write down what this robot can see for itself, stamped `now`.

        FOUR SOURCES, in descending order of certainty:

          1. ITSELF. Which pile it is standing in is not a belief, and
             neither is its own hopper size. Always recorded.
          2. THE PILE IT IS DIGGING. A robot in the hole knows exactly
             how much is left in it.
          3. PILES INSIDE THE SENSOR DISC. Same disc test as robot.sense
             -- (tx-cx)^2 + (ty-cy)^2 <= r^2, compared squared to avoid a
             square root -- so a robot driving past a site reads its size
             off the ground.
          4. MACHINES INSIDE THE DISC that are parked ON a pile
             (Chebyshev distance <= 1 of a task cell, the same
             neighbourhood work_candidates uses). Seeing a machine in a
             hole is seeing it work.

        Source 4 is CONFIRM-ONLY: it can add "robot k is seated on j",
        never clear one. A machine hauling to a dump is nowhere near its
        task cell, so treating "not next to a pile" as "seated nowhere"
        would invent free seats every time a co-worker drove off to
        unload -- and a free seat that is not free is exactly the error
        that puts two robots on one work cell. Clearing a station is left
        to the owning robot's own broadcast, which is authoritative.

        Stamps come from the allocator's round clock, which is monotone,
        so a robot's own fresh entry about itself always beats a peer's
        relayed copy of an older one (stamps are assigned at the origin
        and preserved through every relay).

        CALLED BY: MOACBBAAllocator.allocate, once per robot per tick,
        before the auction rounds begin.
        """
        me = robot.robot_id
        self.capacity[me] = robot.spec.capacity
        self.station[me] = (robot.task_id if robot.task_id is not None
                            else NOWHERE, now)
        if robot.task_id is not None:
            task = model.tasks.get(robot.task_id)
            self.volume[robot.task_id] = (task.remaining, now)

        if self.omniscient(model):
            return                      # accessors read ground truth anyway

        r = robot.sensor_radius
        if r <= 0.0:
            return
        cx, cy = robot.cell.coordinate
        rr = r * r
        for task in model.tasks.all:
            tx, ty = task.cell
            if (tx - cx) ** 2 + (ty - cy) ** 2 <= rr:
                self.volume[task.task_id] = (task.remaining, now)

        for other in robot.visible_robots():
            ox, oy = other.cell.coordinate
            self.capacity[other.robot_id] = other.spec.capacity
            for task in model.tasks.all:
                tx, ty = task.cell
                if max(abs(tx - ox), abs(ty - oy)) <= 1:
                    self.station[other.robot_id] = (task.task_id, now)
                    break

    # ------------------------------------------------------------------ #
    # accessors -- everything the allocator is allowed to ask
    # ------------------------------------------------------------------ #
    def remaining(self, model, task) -> float:
        """Believed remaining volume of a pile.

        The PRIOR, used when nothing has been seen or heard, is the
        surveyed original volume V_j: the fleet was told how big every
        pile was before work started, so an unheard-of task looks
        untouched rather than empty. That errs towards bidding on a pile
        that may already be gone, which costs a wasted drive -- the
        opposite error (assuming it is finished) would strand real work
        with nobody bidding on it.

        CALLED BY: pathCost, finishIfJoined, finishIfLeft,
        capacityAffinity, seats_for and _execute's switch test -- i.e.
        every place MOA-CBBA used to read task.remaining.
        """
        if self.omniscient(model):
            return task.remaining
        entry = self.volume.get(task.task_id)
        return task.volume if entry is None else entry[0]

    def is_done(self, model, task) -> bool:
        """Believed finished. Same 1e-9 threshold as Task.done, applied
        to the believed volume rather than the true one."""
        if self.omniscient(model):
            return task.done
        return self.remaining(model, task) <= EPS

    def seated_on(self, model, task) -> set[int]:
        """Robots this one believes are currently digging `task`.

        The local stand-in for task.assignees. Derived from the station
        table rather than stored per task, because each robot is
        authoritative about exactly one row of it -- its own -- so there
        is never a conflicting claim about the same machine to resolve.
        """
        if self.omniscient(model):
            return set(task.assignees)
        j = task.task_id
        return {k for k, (t, _s) in self.station.items() if t == j}

    def sharers(self, model, task) -> int:
        """|seated_on|, the believed k in the V/k sharing math."""
        return len(self.seated_on(model, task))

    def fleet_capacity(self, model, robot) -> float:
        """C_max over the machines this robot knows about, for
        capacityAffinity's normaliser.

        Peer capacities arrive by gossip and by sight (a machine's size
        is visible), so early in a run this is just the robot's own
        hopper and the affinity term is near-neutral -- which is the
        right behaviour for a robot that has not met anyone yet.
        """
        if self.omniscient(model):
            return max(r.spec.capacity for r in model.robots)
        known = max(self.capacity.values()) if self.capacity else 0.0
        return max(known, robot.spec.capacity)

    def largest_open(self, model, tasks) -> float:
        """V_max over `tasks` by believed volume -- the other half of
        capacityAffinity's normaliser. 0.0 for an empty list, which
        capacityAffinity reads as "no discount"."""
        vals = [self.remaining(model, t) for t in tasks]
        return max(vals) if vals else 0.0

    def candidates(self, model, robot, max_sharers: int) -> list:
        """Tasks this robot BELIEVES it may bid on -- the local
        replacement for TaskRegistry.seats_open.

        Same predicate as seats_open (unfinished, and either I am on it
        or a seat looks free), evaluated against beliefs instead of the
        registry. Two robots can therefore disagree about whether a seat
        is free, which is normal in a decentralised auction and is
        resolved the way every other disagreement is: by gossip, and
        finally by robot.assign refusing to seat a machine that has
        nowhere to stand.

        CALLED BY: MOACBBAAllocator.allocate, per robot per tick.
        """
        me = robot.robot_id
        out = []
        for task in model.tasks.all:
            if self.is_done(model, task):
                continue
            seated = self.seated_on(model, task)
            if me in seated or len(seated) < max_sharers:
                out.append(task)
        return out

    # ------------------------------------------------------------------ #
    # gossip
    # ------------------------------------------------------------------ #
    def payload(self) -> dict:
        """This notebook, flattened for broadcast.

        Keys are stringified because the payload is a plain dict that
        gets deep-copied through CommNetwork, matching how
        MOACBBAAgent.broadcast serialises its bid table.

        The WHOLE table goes out, not just first-hand rows: relaying is
        what makes a range-limited single-hop network reach the far side
        of the site, exactly as it does for bids.
        """
        return {"vol": {str(j): [v, s] for j, (v, s) in self.volume.items()},
                "sta": {str(i): [t, s] for i, (t, s) in self.station.items()},
                "cap": {str(i): c for i, c in self.capacity.items()}}

    def merge(self, payload: dict, now: int) -> bool:
        """Fold a peer's notebook into mine. RETURNS True if anything
        changed.

        NOTE the caller (MOACBBAAgent.resolveConflicts) deliberately
        IGNORES that return value for the convergence test -- see the
        comment there. It is kept because it is the natural thing for a
        merge to report and is useful for instrumentation.

        Per-key rule, the same shape as the bid merge:

            adopt theirs  iff  I have no entry
                          OR   their stamp is newer
                          OR   stamps tie AND their value wins the
                               deterministic tie-break

        Tie-breaks: lower remaining for volumes, lower task id for
        stations. Both are arbitrary but AGREED, which is all that is
        needed to stop two robots swapping the same key back and forth.

        Capacities are static, so first writer wins and no stamp is kept.
        """
        changed = False
        for key, entry in payload.get("vol", {}).items():
            j = int(key)
            vol, stamp = float(entry[0]), int(entry[1])
            mine = self.volume.get(j)
            if mine is None or stamp > mine[1] or \
                    (stamp == mine[1] and vol < mine[0] - EPS):
                self.volume[j] = (vol, stamp)
                changed = True
        for key, entry in payload.get("sta", {}).items():
            i = int(key)
            where, stamp = int(entry[0]), int(entry[1])
            mine = self.station.get(i)
            if mine is None or stamp > mine[1] or \
                    (stamp == mine[1] and where < mine[0]):
                self.station[i] = (where, stamp)
                changed = True
        for key, cap in payload.get("cap", {}).items():
            i = int(key)
            if i not in self.capacity:
                self.capacity[i] = float(cap)
                changed = True
        return changed
