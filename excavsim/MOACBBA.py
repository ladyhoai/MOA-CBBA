"""MOA-CBBA: multi-objective, capacity-aware, sharing, switchable CBBA.

Everything specific to this allocator lives in this file. The only
changes elsewhere are the data model that concurrent sharing requires
(Task.assignees is a set; ExcavatorRobot.assign seats rather than
claims) -- both of which leave CBBA and CBPAE behaving exactly as before,
because those allocators only ever fill one seat.

FOUR MECHANISMS.

1. COST-CONVENTION BIDDING. CBBA maximises a non-negative discounted
   reward and admits a task only if `gain > EPS and gain > y_ij`. Both
   thresholds are failure modes here: a task can be REJECTED outright
   rather than merely ranked last, which was measured as a reachable
   task sitting unclaimed for 62 ticks with the whole fleet idle. Cost
   minimisation has no admission threshold. bidding.py already carries
   the comparator (_cheaper_bid, Das et al. Sec. 3.1).

2. MAKESPAN AS A MARGINAL COST. J(x) = w1*max_i(sum_j tau_ij) +
   w2*sum E_ij. The max is a global coupling no local score can see, so
   both baselines optimise the energy half by construction and the
   makespan half by accident. Robots gossip c_k (projected completion)
   and bid
       w1 * [max(c_i after, max_k c_k) - max(c_i before, max_k c_k)]
     + w2 * (energy added)
   A robot with slack pays nothing in makespan terms, so it keeps
   winning until it becomes the bottleneck, at which point every further
   task costs full price. Self-limiting, unlike a flattened discount.

3. CAPACITY AFFINITY. Big machines should prefer big remaining volumes.
   NOTE: some of this is already in the physics -- n_ij = ceil(V_j/C_i)
   means a small hopper pays (2n-1)*d_dump on a large task while a large
   one pays a single round trip, and that is in both tau_ij and
   energy_ij. The term here is therefore a RANKING preference on top,
   not a re-statement of the same effect, and it is applied only to the
   marginal cost used to order candidates -- never to leg_cost. That
   distinction matters: the bid == uncontended execution invariant is
   measured against leg_cost, so this cannot break it. capacity_affinity
   = 0.0 removes the term entirely for the ablation.

4. CONCURRENT SHARING AND EN-ROUTE SWITCHING. Several robots may hold
   seats on one task; the volume then drains in parallel. Seats are
   capped by policy (max_sharers), by the free work cells physically
   available around the target, and by a minimum share so nobody is
   seated for a sliver. Separately, a robot travelling to a task -- and
   only travelling; can_abandon is stage TO_TASK with an empty hopper,
   Das et al. Sec. 3.7.4 -- may switch to a cheaper task it discovers,
   but only by WINNING it in the auction, and only if the improvement
   clears a hysteresis margin so two near-equal robots do not trade a
   task back and forth every tick.

CONSENSUS. Single-winner y/z cannot express k seats, so the state is
per (task, robot): bids[(j, i)] = (cost, stamp). The winners of task j
are the K cheapest live entries. Merging takes the newer stamp per key,
which is what makes a RELEASE survive contact with a peer that has not
heard it yet -- without stamps the peer echoes the dead claim back and
the robot re-adopts a task it has dropped. That exact bug has now been
hit twice in this codebase (CBPAE.release, and the first draft of this
file), so it is worth naming: min-cost consensus with no clock cannot
distinguish new information from a stale echo.
"""

from __future__ import annotations

from .comms import network_diameter
from .bidding import EPS, INF, NOCOST, leg_cost, residual_cost
from .pathfinding import work_candidates

RELEASED = NOCOST          # a live bid is anything cheaper than this

# A robot must stay on a task it just took for at least this many ticks
# before it may switch away again. Pure lockout, and it is NOT the same
# guard as switch_margin: the margin compares two costs at one instant,
# and both costs move every tick as the robot walks, so a margin alone
# cannot stop a cycle. Measured symptom without it: one robot abandoned
# and re-took the same task on twelve consecutive ticks, its own bid
# falling 168.4 -> 165.9 as it approached, and never arrived.
SWITCH_LOCKOUT = 15

# Defaults -- every one of these is an ablation axis.
MAX_SHARERS = 3            # 1 reproduces single-robot-per-task
MIN_SHARE = 1.2            # volume a seat must be worth having
CAPACITY_AFFINITY = 0.25   # 0.0 removes mechanism 3
SWITCH_MARGIN = 0.20       # en-route switch must be this much cheaper

# OFF BY DEFAULT ON MEASURED EVIDENCE. Paired 8-seed comparison, all
# else identical: en-route switching was worse on 7 seeds, tied on 1 and
# better on 0, mean makespan +15.0 ticks (215.6 vs 200.6) and mean energy
# +4.6 (84.8 vs 80.2). It loses on BOTH objectives at once.
#
# The reason is that a switch throws away sunk approach travel twice
# over: the switching robot discards the distance it already covered
# toward its old task, and the seat it vacates then has to be refilled
# by some robot starting from further away than it was. The lockout and
# the hysteresis margin stop the pathological per-tick churn (which was
# real -- twelve abandon/re-assign cycles on consecutive ticks) but they
# cannot recover that sunk cost, because the bid compares REMAINING cost
# from here and is blind to what has already been spent getting here.
#
# Kept, not deleted: it is a clean ablation axis, and it may well pay off
# in a regime with more tasks per robot or heavier dynamics, where the
# information a robot gains en route is worth more. Enable with
# MOACBBAAllocator(enable_switching=True).
ENABLE_SWITCHING = False
# How many tasks a robot may add to its bundle in ONE auction round.
#
# The problem this fixes: createBundle used to fill the whole bundle
# (up to L_t = 6 tasks) before a single message was exchanged. So in
# round 1 every robot decided its entire schedule from the SAME stale
# c_k -- and with N robots there is always exactly one bottleneck and
# N-1 robots who all correctly compute "extra work is free for me"
# (measured: 3 of 4 robots in 99.5% of ticks). All of them then took
# work on that basis, and after they did, several had passed the
# bottleneck they were measuring themselves against. The work was not
# free after all; they just could not know it yet.
#
# Adding one task per round forces a broadcast between every decision,
# so the second task is chosen against a c_k that already reflects the
# first. Robots still cannot un-take work -- createBundle only grows,
# and only releaseOutbid shrinks -- so the fix is to stop them
# over-committing rather than to let them back out.
#
# None = old behaviour (fill the bundle in one round).
MAX_ADDS_PER_ROUND = 1

# Route around where a sensed obstacle is ABOUT to be, not just where it
# was seen. Obstacles step with probability obstacle_move_probability
# every tick, so the eight cells around one are a coin flip on being
# blocked by the time a robot arrives -- and being blocked costs
# STUCK_LIMIT waiting ticks plus a re-plan, while stepping one cell wide
# costs at most one extra move.
#
# Deliberately a PREFERENCE with a fallback, not a wall: _plan_leg tries
# the halo first and re-plans without it if that leaves no route. And
# deliberately short-horizon (HALO_HORIZON in robot.py): the prediction
# is only good for the next few ticks, after which the obstacle has
# wandered somewhere unrelated.
#
# NOTE the honest cost: leg_cost prices bids on known_blocked() WITHOUT
# the halo, while execution plans WITH it, so a halo detour is a length
# the bid did not charge for. The gap is small -- a halo route is
# typically one or two steps longer -- and it is a deliberate trade
# against the wait it avoids, but it does widen bid-vs-actual.
OBSTACLE_HALO = True

ROUND_CEILING = 200


class MOACBBAAgent:
    """Per-robot auction state. Cost convention: LOWER wins."""

    def __init__(self):
        self.bundle: list[int] = []
        self.path: list[int] = []
        self.bids: dict[tuple[int, int], tuple[float, int]] = {}
        self.completions: dict[int, float] = {}
        self.myCompletion = 0.0
        self.task_list: list = []
        self._now = 0
        self.switched_tick = -10 ** 9      # last en-route switch

    # ---------------- bid table ------------------------------------- #
    def _live(self, j) -> list[tuple[float, int]]:
        """(cost, robot) for every robot with a live bid on j, cheapest
        first, ties on the lower robot id."""
        out = [(c, i) for (t, i), (c, s) in self.bids.items()
               if t == j and c < RELEASED]
        out.sort()
        return out

    def winners(self, j, seats: int) -> list[int]:
        return [i for _c, i in self._live(j)[:max(0, seats)]]

    def seated(self, j, me, seats: int) -> bool:
        return me in self.winners(j, seats)

    def worst_seated(self, j, seats: int) -> float:
        """Cost of the marginal seat: what a newcomer has to beat."""
        live = self._live(j)
        return live[seats - 1][0] if len(live) >= seats else RELEASED

    def place(self, j, me, cost) -> None:
        self.bids[(j, me)] = (cost, self._now)

    def release(self, j, me) -> None:
        self.bids[(j, me)] = (RELEASED, self._now)

    # ---------------- costing --------------------------------------- #
    def pathCost(self, model, robot, path, sharers_fn=None) -> tuple[float, float]:
        """(completion time, energy) for executing `path` in order.

        SHARING CORRECTION. leg_cost prices task.remaining as if this
        robot digs the whole thing alone. With k robots seated each digs
        roughly V/k, so the share is passed INTO leg_cost rather than
        divided out of its result.

        The earlier version divided the returned tau by k and left the
        energy alone, on the reasoning that k robots moving a third of
        the soil each spend what one robot spends moving all of it. That
        is true of the FLEET and false of each ROBOT, which is what a bid
        prices: energy_ij's dig term is BETA*V*H/rho and its haul term
        carries n = ceil(V/C), both proportional to the volume THIS robot
        moves. The measured consequence was every shared task coming in
        at x0.2-x0.6 of its energy bid, i.e. w2 silently inflated on
        exactly the tasks this allocator is built around.

        Dividing the totals would have been wrong too: the approach leg
        is paid in full however little is dug, and n = ceil(V/C) is a
        step function. Scaling the volume gets both right.
        """
        tau0, e0 = residual_cost(model, robot)
        t, e = float(tau0), float(e0)
        startPos = robot.dump_cell if robot.task_id is not None else None
        for task_id in path:
            if task_id == robot.task_id:
                continue
            task = model.tasks.get(task_id)
            k = 1 if sharers_fn is None else max(1, sharers_fn(task))
            tau, en, dump = leg_cost(model, robot, task, startPos,
                                     volume=task.remaining / k)
            if dump is None:
                return INF, INF
            e += en
            t += tau
            startPos = dump
        return t, e

    def marginalCost(self, model, robot, trial, base_t, base_e,
                     others_c, affinity, sharers_fn=None) -> float:
        t, e = self.pathCost(model, robot, trial, sharers_fn)
        if t >= INF:
            return RELEASED
        before = base_t if base_t > others_c else others_c
        after = t if t > others_c else others_c
        cost = model.w1 * (after - before) + model.w2 * (e - base_e)
        return cost * affinity

    def capacityAffinity(self, robot, task, c_max, v_max, kappa) -> float:
        """Multiplier in (1-kappa, 1]. Falls as capacity and remaining
        volume both rise, so the big machine is drawn to the big pile.
        Normalised by the fleet's largest hopper and the largest open
        task so the term cannot dominate the physical cost."""
        if kappa <= 0.0 or c_max <= 0.0 or v_max <= 0.0:
            return 1.0
        match = (robot.spec.capacity / c_max) * (task.remaining / v_max)
        return 1.0 - kappa * min(1.0, max(0.0, match))

    # ---------------- bundle ---------------------------------------- #
    def createBundle(self, model, robot, seats_fn, c_max, v_max,
                     kappa, limit, max_adds=None) -> None:
        me = robot.robot_id
        others_c = self.othersCompletion(me)
        # Expected sharers: what the task will look like once seated,
        # counting myself if I am not already on it. Using the CURRENT
        # occupancy instead would price the first robot's bid as solo and
        # then never revise it.
        def expected(task):
            return self.expectedSharers(task, me, seats_fn(task))
        added = 0
        while len(self.bundle) < limit:
            if max_adds is not None and added >= max_adds:
                break              # broadcast, hear the others, then continue
            base_t, base_e = self.pathCost(model, robot, self.path, expected)
            if base_t >= INF:
                break
            best = None
            for task in self.task_list:
                j = task.task_id
                if j in self.bundle:
                    continue
                seats = seats_fn(task)
                if seats <= 0:
                    continue
                aff = self.capacityAffinity(robot, task, c_max, v_max, kappa)
                cheapest = None
                for n in range(len(self.path) + 1):
                    trial = self.path[:n] + [j] + self.path[n:]
                    c = self.marginalCost(model, robot, trial, base_t,
                                          base_e, others_c, aff, expected)
                    if c >= RELEASED:
                        continue
                    if cheapest is None or c < cheapest[0] - EPS:
                        cheapest = (c, n)
                if cheapest is None:
                    continue
                # a seat is winnable if it is free, or if I beat the
                # robot currently holding the marginal one
                incumbent = self.worst_seated(j, seats)
                if not (self.seated(j, me, seats)
                        or cheapest[0] < incumbent - EPS):
                    continue
                if best is None or cheapest[0] < best[0] - EPS:
                    best = (cheapest[0], j, cheapest[1])
            if best is None:
                break
            cost, j, n = best
            self.bundle.append(j)
            self.path.insert(n, j)
            self.place(j, me, cost)
            added += 1

    def releaseOutbid(self, robot, seats_fn, model) -> None:
        """Losing a seat invalidates every marginal cost after it, so the
        tail goes too (Choi et al. Eq. 6)."""
        me = robot.robot_id
        cut = None
        for idx, j in enumerate(self.bundle):
            if j == robot.task_id:
                continue
            seats = seats_fn(model.tasks.get(j))
            if not self.seated(j, me, seats):
                cut = idx
                break
        if cut is None:
            return
        for j in self.bundle[cut:]:
            if j == robot.task_id:
                continue
            if j in self.path:
                self.path.remove(j)
            self.release(j, me)
        tail_lock = [robot.task_id] if robot.task_id in self.bundle[cut:] else []
        self.bundle = self.bundle[:cut] + tail_lock

    def prune(self, model, robot) -> None:
        keep = []
        for j in self.bundle:
            task = model.tasks.get(j)
            if task.done:
                self.release(j, robot.robot_id)
                continue
            keep.append(j)
        for j in [x for x in self.bundle if x not in keep]:
            if j in self.path:
                self.path.remove(j)
        self.bundle = keep

    # ---------------- gossip ---------------------------------------- #
    def expectedSharers(self, task, me, seats: int) -> int:
        """How many robots will be digging this task once the auction
        settles -- read off the BID TABLE, not off current occupancy.

        THE BUG THIS REPLACES had two halves, and the first is worse.

        (a) INCONSISTENCY. The old estimate was

                n = len(assignees) + (0 if me in assignees else 1)

            which gives a DIFFERENT answer to different robots at the
            same instant. For a task with one robot on it, the incumbent
            priced it solo (V/1) while a challenger priced it shared
            (V/2) -- so the two were comparing bids computed on volumes
            that differ by a factor of two. That is not forecast error,
            it is an unfair comparison, and it decides who wins a seat.

        (b) FORECAST ERROR. task.assignees only changes in _execute(),
            AFTER the whole auction has converged. So during bidding
            every robot sees pre-auction occupancy and none of them can
            see that two more are about to sit down. Measured: tasks
            whose sharer count stayed put came in at x1.05 of bid; tasks
            whose count changed came in at x0.87, and they were the
            majority (28 of 52).

        The bid table fixes both. `winners(j, seats)` is exactly the set
        that _execute will seat, it is agreed across robots by consensus,
        and it updates every round -- so all bidders price the same task
        on the same volume, and that volume is what execution will
        actually see.

        Union with assignees because a robot already digging holds its
        seat through a LOCKED_COST entry, and with `me` because asking
        "what would this task cost me" presumes joining it.
        """
        j = task.task_id
        who = set(self.winners(j, seats)) | set(task.assignees)
        who.add(me)
        return max(1, min(len(who), seats))

    def othersCompletion(self, me) -> float:
        vals = [c for k, c in self.completions.items() if k != me]
        return max(vals) if vals else 0.0

    def broadcast(self, robot, now, sharers_fn=None) -> None:
        self._now = now
        self.myCompletion, _e = self.pathCost(robot.model, robot, self.path,
                                              sharers_fn)
        robot.send({"sender": robot.robot_id,
                    "bids": {f"{j}:{i}": v for (j, i), v in self.bids.items()},
                    "c": self.myCompletion})

    def resolveConflicts(self, robot, messages, now) -> bool:
        self._now = max(self._now, now)
        changed = False
        for msg in messages:
            k = msg.payload.get("sender")
            if k is None or k == robot.robot_id:
                continue
            c = msg.payload.get("c")
            if c is not None:
                self.completions[k] = float(c)
            for key, (cost_k, stamp_k) in msg.payload.get("bids", {}).items():
                j_s, i_s = key.split(":")
                key2 = (int(j_s), int(i_s))
                mine = self.bids.get(key2)
                # newer wins; equal stamps break on the cheaper value,
                # then deterministically so two robots cannot disagree
                if mine is None or stamp_k > mine[1] or \
                        (stamp_k == mine[1] and cost_k < mine[0] - EPS):
                    self.bids[key2] = (cost_k, stamp_k)
                    changed = True
        return changed


class MOACBBAAllocator:
    """Multi-objective adaptive CBBA."""

    name = "moa-cbba"
    AGENT_ATTR = "MOACBBA"

    def __init__(self, obstacle_halo: bool = OBSTACLE_HALO,
                 max_sharers: int = MAX_SHARERS,
                 min_share: float = MIN_SHARE,
                 capacity_affinity: float = CAPACITY_AFFINITY,
                 switch_margin: float = SWITCH_MARGIN,
                 enable_switching: bool = ENABLE_SWITCHING,
                 max_adds_per_round: int | None = MAX_ADDS_PER_ROUND) -> None:
        self.max_adds_per_round = max_adds_per_round
        self.obstacle_halo = bool(obstacle_halo)
        self.max_sharers = int(max_sharers)
        self.min_share = float(min_share)
        self.capacity_affinity = float(capacity_affinity)
        self.switch_margin = float(switch_margin)
        self.enable_switching = bool(enable_switching)
        self.last_round = 0
        self.converged = False
        self._clock = 0
        self.n_switches = 0
        self.max_sharers_seen = 0

    def _agent(self, robot):
        return getattr(robot, self.AGENT_ATTR)

    # ---------------------------------------------------------------- #
    def seats_for(self, model, task) -> int:
        """How many robots this task is worth, and can physically hold.

        Three caps: policy, the volume (a seat must be worth min_share),
        and the free work cells around the target -- work_candidates
        yields at most 9, and bedrock, dumps, hazards and parked robots
        all subtract. Seating more robots than there are cells to stand
        in just produces collisions and STUCK_LIMIT waits."""
        if self.max_sharers <= 1:
            return 1
        by_volume = int(task.remaining // self.min_share)
        free = len(work_candidates([task.cell], model.grid.width,
                                   model.grid.height, model.blocked_cells()))
        return max(1, min(self.max_sharers, by_volume, free))

    # ---------------------------------------------------------------- #
    def allocate(self, model) -> None:
        # Announce the routing policy every call rather than once at
        # construction: SolaraViz swaps allocators on a live model, so a
        # flag set in __init__ would outlive the allocator that wanted it.
        model.obstacle_halo_enabled = self.obstacle_halo

        robots = list(model.robots)
        open_tasks = model.tasks.unfinished
        if not robots or not open_tasks:
            return

        c_max = max(r.spec.capacity for r in robots)
        v_max = max(t.remaining for t in open_tasks)
        seats_fn = lambda t: self.seats_for(model, t)   # noqa: E731

        for r in robots:
            agent = self._agent(r)
            agent.prune(model, r)
            # A robot may bid on any unfinished task with a free seat,
            # plus the one it is already working. Crucially this includes
            # tasks it is merely TRAVELLING to -- that is what makes an
            # en-route switch expressible as an ordinary auction result
            # rather than a special case.
            agent.task_list = model.tasks.seats_open(r.robot_id,
                                                     self.max_sharers)
            r.receive_all()

        L_t = max(r.bundle_limit for r in robots)
        n_min = min(len(open_tasks) * self.max_sharers, len(robots) * L_t)
        max_rounds = min(ROUND_CEILING,
                         max(2, n_min * network_diameter(model)))

        self.converged = False
        for rnd in range(1, max_rounds + 1):
            self._clock += 1
            for robot in robots:
                agent = self._agent(robot)
                agent.createBundle(model, robot, seats_fn, c_max, v_max,
                                   self.capacity_affinity, robot.bundle_limit,
                                   self.max_adds_per_round)
                # c_i must be computed on the SAME volume basis as the
                # bids, or the completion time a robot advertises is not
                # the completion time its own bids were built from --
                # every peer then load-balances against a schedule that
                # does not exist.
                agent.broadcast(
                    robot, self._clock,
                    lambda t: agent.expectedSharers(t, robot.robot_id,
                                                    seats_fn(t)))
            model.comms.flush_and_deliver(model.tick)
            changed = [self._agent(r).resolveConflicts(r, r.receive_all(),
                                                       self._clock)
                       for r in robots]
            for r in robots:
                self._agent(r).releaseOutbid(r, seats_fn, model)
            self.last_round = rnd
            if not any(changed):
                self.converged = True
                break

        self._execute(model, robots, seats_fn)

    # ---------------------------------------------------------------- #
    def _execute(self, model, robots, seats_fn) -> None:
        for robot in robots:
            agent = self._agent(robot)
            me = robot.robot_id

            # --- en-route switching --------------------------------- #
            # Only in execution phase 1 with an empty hopper (Das et al.
            # Sec. 3.7.4), only for a materially cheaper alternative, and
            # only if this robot has not switched recently.
            if (self.enable_switching
                    and robot.task_id is not None and robot.can_abandon
                    and model.tick - agent.switched_tick >= SWITCH_LOCKOUT):
                # Both sides priced HERE, from the same position, on the
                # same basis. Reading the stored bid for the current task
                # compared a value computed at assignment time against
                # rivals repriced this tick, so the comparison drifted in
                # favour of switching a little more every tick.
                def expected(task):
                    return agent.expectedSharers(task, me, seats_fn(task))

                cur_task = model.tasks.get(robot.task_id)
                cur_k = max(1, expected(cur_task))
                cur_tau, cur_e, cur_q = leg_cost(
                    model, robot, cur_task,
                    volume=cur_task.remaining / cur_k)
                current = (INF if cur_q is None
                           else model.w1 * cur_tau + model.w2 * cur_e)
                better = None
                for j in agent.path:
                    if j == robot.task_id:
                        continue
                    task = model.tasks.get(j)
                    if not agent.seated(j, me, seats_fn(task)):
                        continue
                    if len(task.assignees) >= seats_fn(task):
                        continue            # no seat actually free
                    k = max(1, expected(task))
                    tau, en, q = leg_cost(model, robot, task,
                                          volume=task.remaining / k)
                    if q is None:
                        continue
                    cost = model.w1 * tau + model.w2 * en
                    if cost < current * (1.0 - self.switch_margin) \
                            and (better is None or cost < better[0]):
                        better = (cost, j)
                if better is not None:
                    # Look before leaping: abandon_task() was previously
                    # called BEFORE the target was checked, so a switch
                    # that could not complete still dropped the current
                    # task and the robot restarted it from scratch next
                    # tick -- the churn loop, twelve ticks running.
                    old = robot.task_id
                    robot.abandon_task()
                    if robot.assign(better[1]):
                        agent.release(old, me)
                        agent.switched_tick = model.tick
                        self.n_switches += 1
                    elif not robot.assign(old):
                        agent.release(old, me)   # could not go back either

            if robot.task_id is not None:
                continue
            for task_id in list(agent.path):
                task = model.tasks.get(task_id)
                if not agent.seated(task_id, me, seats_fn(task)):
                    continue
                # Seat cap is enforced HERE, against the registry, not
                # against the local bid table. Mid-auction the robots'
                # views differ, so k robots can each believe they hold
                # one of the k cheapest seats; only the task itself knows
                # how many are actually sitting down.
                if len(task.assignees) >= seats_fn(task):
                    continue
                if robot.assign(task_id):
                    break
                # assign() refused (no free work cell, or no reachable
                # dump). Drop the seat as well as the path entry, or the
                # bid keeps blocking a robot that could have taken it.
                agent.path.remove(task_id)
                if task_id in agent.bundle:
                    agent.bundle.remove(task_id)
                agent.release(task_id, me)

        self.max_sharers_seen = max(
            [self.max_sharers_seen] + [t.sharers for t in model.tasks.all])


def register() -> None:
    """Add this allocator to the shared registry.

    Deferred into a function and called from model.py rather than run at
    import time, so this module never imports allocation.py at module
    scope. allocation.py must not import this one either -- that pair of
    top-level imports is exactly the cycle that made startup order
    matter and broke `solara run app.py`.
    """
    from .allocation import ALLOCATORS
    ALLOCATORS[MOACBBAAllocator.name] = MOACBBAAllocator