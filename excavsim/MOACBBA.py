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

New-reader primer: MOA-CBBA is this project's own extension of CBBA
(allocation.py), built to fix two things a plain reward-maximising
auction handles badly for this domain: it doesn't naturally balance
WORKLOAD (one robot can end up doing far more than others, since
finishing late costs a robot nothing extra), and it can't put more than
one robot on the same task at once. Read the four "MECHANISMS" above in
order -- each is a self-contained fix (cost-based bidding, makespan-
aware pricing, big-robot/big-task matching, sharing + switching) and
each can be turned off individually (see the constants just below) to
measure what it's actually buying. If you already understand
allocation.py's CBBAAgent, MOACBBAAgent is the same bundle-building idea
with a different price tag.
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
                           # (legacy "margin" criterion only -- see
                           # SWITCH_CRITERION below)

# Which test decides an en-route switch.
#
#   "rtlff"  -- Parker & Gini's Real-Time Latest Finishing First rule
#               (AAMAS 2014, Sec. 6): move a robot from task i to task j
#               only if task i STILL FINISHES BEFORE task j would, even
#               after losing this robot:
#
#                   ct-_i  <  ct+_j
#
#               where ct-_i is i's completion time with one fewer digger
#               and ct+_j is j's with one more, INCLUDING the transferring
#               robot's travel time. See finishIfLeft / finishIfJoined.
#
#   "margin" -- the original rule: switch when the candidate is cheaper
#               for ME by more than SWITCH_MARGIN. Kept for the ablation.
#
# Why the change. "margin" is a UNILATERAL test: it asks whether j is a
# better deal for this robot and never asks what happens to i once the
# robot walks away. That is a greed test, not an allocation test, and it
# is why switching measured worse on both objectives -- a robot abandons
# a half-approached pile because a nearer one looks cheap, and the pile
# it left has to be re-approached from scratch by somebody further away.
# The cost of the abandoned approach is real and nobody pays it in the
# comparison.
#
# Parker & Gini hit this directly and say so. They considered the obvious
# makespan-improving rule, max(ct_i, ct_j) >= max(ct-_i, ct+_j), and
# rejected it because it "can cause assignment thrashing, especially when
# there is noise or an error in the growth function" -- and our growth
# function IS noisy, since bids are priced on each robot's own sensed map
# under moving hazards and changing weather. Their greedy rule is
# self-damping instead: after a transfer the two completion times are
# approximately equal, so a further transfer between the same pair is
# unlikely. That property is why "rtlff" needs no SWITCH_MARGIN and, in
# principle, no SWITCH_LOCKOUT (the lockout is retained as a belt-and-
# braces bound while the rule is unproven in this domain).
SWITCH_CRITERION = "rtlff"

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
ENABLE_SWITCHING = True
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

# Re-price bundle entries the robot already holds, every round.
#
# The problem this fixes: createBundle skips any task already in the
# bundle (`if j in self.bundle: continue`), so a bid was placed ONCE and
# then never revisited. Its value stayed frozen at the world state and
# robot position of whenever it was inserted -- while the terrain, the
# weather, the robot's own known_blocked map and every peer's gossiped
# completion time all moved on around it. A held claim was therefore the
# one thing in the auction immune to new information, which is precisely
# backwards: it is the claim the robot is most committed to.
#
# Two consequences were visible. First, reallocation could only ever be
# failure-driven -- a task changed hands when it became IMPOSSIBLE (the
# UNREACHABLE_PATIENCE release in robot.py), never when it merely became
# a worse idea than someone else's. Second, myCompletion is built from
# pathCost over the same held entries, so a robot broadcast a c_i
# assembled from stale estimates, and every peer load-balanced its
# makespan term against a schedule that had drifted.
#
# Re-pricing closes both through machinery that already exists: a robot
# whose cost has risen simply loses the seat to one whose has not, via
# the ordinary worst_seated / releaseOutbid path. Nothing new is
# committed and no execution is interrupted -- the task under execution
# is exempt (see repriceHeld).
#
# This is also the property Das et al. Sec. 3.7.1 argues is responsible
# for CBPAE's advantage ("multi-round dynamic bidding with improved bids
# in each round"): as residual_cost on the current task shrinks, a bid on
# the NEXT task genuinely improves. Without re-pricing, MOA-CBBA had the
# parallel auction-and-execution STRUCTURE but not that behaviour.
#
# Cost: one extra pathCost per held task per round, and more messages
# (every re-place bumps the stamp). REPRICE_EPS is what keeps that
# bounded -- see below. False = old behaviour, for the ablation.
REPRICE_HELD = True

# Relative change a re-priced bid must show before it is actually
# re-placed. Every place() bumps the entry's stamp and so wins the next
# merge and travels to every peer; re-placing on floating-point noise
# would generate constant stamp churn and message traffic without moving
# a single allocation. It also damps oscillation: MOA-CBBA has no
# Lemma 4 clamp (its score is not monotone in t, so Choi et al.'s
# condition (32) cannot hold anyway), and bids that may now move in BOTH
# directions are exactly the case that clamp was guarding against, so the
# hysteresis is doing real work rather than saving a few bytes.
REPRICE_EPS = 0.02

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
        first, ties on the lower robot id.

        "Live" means cheaper than RELEASED (+inf) -- a release is
        recorded as an infinite-cost bid rather than a deletion, so that
        it carries a timestamp and can win a merge against a peer still
        echoing the old claim. Sorting tuples sorts by cost then robot
        id, giving the deterministic tie-break every robot agrees on.

        CALLED BY: winners and worst_seated.
        """
        out = [(c, i) for (t, i), (c, s) in self.bids.items()
               if t == j and c < RELEASED]
        out.sort()
        return out

    def winners(self, j, seats: int) -> list[int]:
        """The `seats` cheapest live bidders on task j -- who I believe
        will be digging it. This replaces CBBA's single z_ij winner and
        is what lets several robots hold one task."""
        return [i for _c, i in self._live(j)[:max(0, seats)]]

    def seated(self, j, me, seats: int) -> bool:
        """Am I among the winners of task j?"""
        return me in self.winners(j, seats)

    def worst_seated(self, j, seats: int) -> float:
        """Cost of the marginal seat: what a newcomer has to beat.

        MATH: the cost of the k-th cheapest live bid, where k = seats.
        If fewer than `seats` robots have bid, a seat is still free and
        this returns RELEASED (+inf), so any finite bid wins it.

        This is the multi-seat generalisation of CBBA's y_ij threshold:
        instead of "beat the single winner", it is "beat the cheapest
        loser who currently has a seat".

        CALLED BY: createBundle, to test whether a seat is winnable.
        """
        live = self._live(j)
        return live[seats - 1][0] if len(live) >= seats else RELEASED

    def place(self, j, me, cost) -> None:
        """Record my bid on task j, stamped with the current auction
        clock. The stamp is what lets resolveConflicts tell new
        information from a stale echo."""
        self.bids[(j, me)] = (cost, self._now)

    def release(self, j, me) -> None:
        """Give up my seat on task j -- recorded as an infinite-cost bid
        with a FRESH stamp, not as a deletion. A deletion would simply be
        re-learned from the next peer who had not heard yet; a stamped
        release wins the merge and sticks."""
        self.bids[(j, me)] = (RELEASED, self._now)

    # ---------------- costing --------------------------------------- #
    def pathCost(self, model, robot, path, sharers_fn=None) -> tuple[float, float]:
        """(completion time, energy) for executing `path` in order.

        MATH -- walk the path accumulating both quantities, starting from
        whatever the robot already owes on its current task:

            t, e  =  residual_cost(robot)            (starting offset)
            for each task j in path:
                k        = expected sharers of j
                tau, E   = leg_cost(j, from startPos, volume = V_j / k)
                t += tau ;  e += E
                startPos = q*_j     (next leg starts at this dump)

        The returned `t` is this robot's projected COMPLETION TIME c_i,
        which is exactly what gets gossiped to peers and used as the
        makespan term in marginalCost. Threading startPos through is what
        makes the cost order-dependent, as in CBBA's pathScoreCBBA.

        Returns (INF, INF) if any leg is infeasible.

        CALLED BY: createBundle (base cost and, via marginalCost, every
        trial insertion) and broadcast (to advertise c_i).

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
        """What adding a task to my schedule costs THE FLEET, not just me.

        MECHANISM 2, the heart of MOA-CBBA. MATH:

            before = max(base_t,  others_c)     fleet finish time now
            after  = max(t_trial, others_c)     fleet finish time if I
                                                take this task
            cost   = w1*(after - before) + w2*(e_trial - base_e)
            return   cost * affinity

        where others_c = max completion time over all OTHER robots
        (gossiped, see othersCompletion) and t_trial/e_trial come from
        pathCost on the candidate path.

        WHY THE max: J(x)'s makespan term is a max over robots, so the
        fleet only finishes later if I become the bottleneck. Concretely:

          - If I have slack (base_t and t_trial both below others_c),
            then before == after == others_c and the makespan term is
            ZERO. Extra work is free to me, so I keep winning tasks.
          - Once my schedule passes others_c, every further task costs
            its full duration in makespan terms, and I stop winning.

        That is self-limiting load balancing, and it is exactly what a
        purely local score cannot express -- a robot cannot know it is
        the bottleneck without knowing everyone else's completion time.
        Compare CBBA's pathScoreCBBA, which discounts by arrival time and
        so only ever approximates this.

        `affinity` is mechanism 3 (capacityAffinity), a multiplier in
        (1-kappa, 1] applied to the whole cost -- a ranking preference
        only, never applied to leg_cost, so the bid == execution
        invariant is untouched.

        Cost convention: LOWER WINS. Infeasible returns RELEASED (+inf).

        CALLED BY: createBundle, once per candidate task per insertion
        position.
        """
        t, e = self.pathCost(model, robot, trial, sharers_fn)
        if t >= INF:
            return RELEASED
        before = base_t if base_t > others_c else others_c
        after = t if t > others_c else others_c
        cost = model.w1 * (after - before) + model.w2 * (e - base_e)
        return cost * affinity

    def capacityAffinity(self, robot, task, c_max, v_max, kappa) -> float:
        """MECHANISM 3: make big machines prefer big piles.

        MATH -- a discount multiplier applied to the marginal cost:

            match = (C_i / C_max) * (V_j / V_max)      in [0, 1]
            return  1 - kappa * clamp(match, 0, 1)     in (1-kappa, 1]

        Both factors are NORMALISED -- capacity against the fleet's
        largest hopper, volume against the largest open task -- so the
        product is a dimensionless "how well matched are these two"
        score, and the discount can never exceed kappa (0.25 by default).
        A large robot considering a large task gets up to 25% off; a
        small robot on a small task gets no discount at all, but neither
        does it get a penalty.

        NOTE this is a RANKING preference layered on top of physics that
        already partly does this: n_ij = ceil(V_j/C_i) means a small
        hopper pays (2n-1)*d_dump on a big task while a large one pays a
        single round trip, and that is already inside tau_ij and
        energy_ij. So this term sharpens an existing effect rather than
        inventing one. kappa = 0.0 removes it entirely for the ablation.

        CALLED BY: createBundle, once per candidate task.
        """
        if kappa <= 0.0 or c_max <= 0.0 or v_max <= 0.0:
            return 1.0
        match = (robot.spec.capacity / c_max) * (task.remaining / v_max)
        return 1.0 - kappa * min(1.0, max(0.0, match))

    # ---------------- bundle ---------------------------------------- #
    def createBundle(self, model, robot, seats_fn, c_max, v_max,
                     kappa, limit, max_adds=None) -> None:
        """Grow this robot's bundle -- MOA-CBBA's phase 1.

        Structurally the same greedy insertion loop as CBBA's
        createBundle, with three differences:

          COST NOT REWARD    -- picks the task with the LOWEST
              marginalCost (which already contains the makespan term and
              capacity affinity) rather than the highest reward. There is
              no admission threshold: CBBA's `gain > 0 and gain > y_ij`
              can REJECT a task outright, which was measured leaving a
              reachable task unclaimed for 62 ticks with the fleet idle.
              A cost-minimiser always ranks; it never refuses.
          SEATS NOT WINNERS  -- a task is winnable if a seat is free OR
              this robot beats the marginal seat-holder (worst_seated),
              rather than beating a single winner.
          ONE ADD PER ROUND  -- with max_adds=1 the loop stops after one
              insertion so a broadcast happens before the next decision.
              See the MAX_ADDS_PER_ROUND note above: filling the whole
              bundle from one stale snapshot of c_k had N-1 robots all
              correctly computing "extra work is free for me" and all
              acting on it simultaneously.

        As in CBBA, every insertion POSITION is tried and the cheapest
        kept, so ordering is optimised rather than appended to.

        CALLED BY: MOACBBAAllocator.allocate, once per robot per round.
        """
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

    def finishIfJoined(self, model, robot, task, seats) -> float:
        """ct+_j -- when task j would finish if THIS robot joined it.

        Parker & Gini's ct+ (AAMAS 2014, Sec. 6), the receiving side of
        the RT-LFF transfer test. Priced through leg_cost from the
        robot's CURRENT cell, so the approach leg it would have to drive
        is inside the number: their rule requires the transferring
        agent's travel delay to be charged to the receiving task, which
        is exactly what makes the test refuse a distant "cheap" pile.

        MATH:  ct+_j = now + tau_ij( V_j / k )  with k = expectedSharers
        including this robot. The robot is the LAST to arrive, so its own
        finish time is the task's finish time to a good approximation.

        Returns INF when the task cannot be priced, which the caller must
        treat as "not a candidate" rather than as a late finish -- see
        the guard in _execute.
        """
        k = self.expectedSharers(task, robot.robot_id, seats)
        tau, _e, q = leg_cost(model, robot, task,
                              volume=task.remaining / k)
        if q is None:
            return INF
        return model.tick + tau

    def finishIfLeft(self, model, robot, task) -> float:
        """ct-_i -- when this robot's CURRENT task would finish if it
        walked away now.

        The giving side of the RT-LFF test, and the term the old margin
        criterion had no notion of at all.

        TWO CASES.

        Co-workers remain. The diggers left behind absorb this robot's
        share, so the dig phase stretches by k/(k-1) for k current
        sharers. Their finish times are taken from the GOSSIPED
        completion times (self.completions) rather than recomputed --
        pricing another robot's leg would need that robot's known_blocked
        map, which this robot does not have and must not have. The
        estimate is therefore a belief that lags by a round, in exactly
        the same way marginalCost's others_c does.

        Nobody remains. The pile goes back in the pool and waits for
        somebody to be free, so it finishes no sooner than the fleet's
        current bottleneck plus the whole job done solo:

            ct-_i = othersCompletion + tau_ij( V_j )

        That is deliberately pessimistic. Abandoning a task nobody else
        is on is the move that produced the churn loop, and the RT-LFF
        test should have to clear a high bar before making it.

        Returns INF when nothing has been heard from the fleet yet, so a
        robot with no gossip never switches -- the safe default, since
        with no completion times the makespan term is uninformed.
        """
        me = robot.robot_id
        others = [r for r in task.assignees if r != me]
        if others:
            k = max(1, len(task.assignees))
            base = max((self.completions.get(r, 0.0) for r in others),
                       default=0.0)
            if base <= model.tick:
                return INF          # no usable belief about them
            return model.tick + (base - model.tick) * k / max(1, k - 1)

        others_c = self.othersCompletion(me)
        if others_c <= 0.0:
            return INF              # nothing heard: refuse to switch
        tau, _e, q = leg_cost(model, robot, task, volume=task.remaining)
        if q is None:
            return INF
        return others_c + tau

    def repriceHeld(self, model, robot, seats_fn, c_max, v_max,
                    kappa) -> bool:
        """Re-value every task already in the bundle against the CURRENT
        world, and re-place the bid where it has materially moved.

        The counterpart of createBundle: that method prices tasks the
        robot does NOT hold, this one prices the tasks it does. Together
        they mean no bid in the auction is older than one round. See the
        REPRICE_HELD note at the top of this module for why holding a
        frozen bid was the wrong default.

        WHAT A HELD BID MEANS. createBundle sets a bid to the marginal
        cost of INSERTING j into the path at its cheapest position, given
        the bundle prefix at that moment (Choi et al. Eq. 3). That
        definition is not re-computable later: the prefix has changed and
        the path order it produced is not stored per-entry. What is
        computable, and what this uses, is the LEAVE-ONE-OUT marginal --

            base   = pathCost(path with j removed)
            cost   = marginalCost(path as it stands, against base)

        i.e. "what is j costing me, in this schedule, right now". For the
        most recently added task the two coincide exactly; for earlier
        entries they differ, because a later task may have absorbed some
        of j's travel. Leave-one-out is the more honest of the two for
        re-pricing anyway: it answers what the robot would actually save
        by giving j up, which is the question a re-price exists to ask.

        EXEMPTIONS, both deliberate:
          - robot.task_id is skipped. The robot is physically in that
            hole; releaseOutbid already refuses to release it, so
            re-pricing could only ever move its bid without being able to
            act on the result. Worse, a raised bid there could hand the
            seat to a challenger in the bid table while the registry
            check in _execute blocks the challenger from actually taking
            it -- the phantom-seat case. Left alone on purpose.
          - a task whose counterfactual path cannot be priced (base is
            infinite) keeps its existing bid rather than being released:
            an unpriceable COUNTERFACTUAL says nothing about whether j
            itself is still feasible.

        A task that has become genuinely infeasible (marginalCost returns
        RELEASED) is released here. releaseOutbid then sees the robot is
        no longer seated and cuts it along with its tail, which is the
        correct Eq. (6) cascade -- every bid after it was priced as an
        addition to a schedule that no longer exists.

        RETURNS True if any bid moved, so the caller can keep the round
        loop alive: a re-priced bid is new information that still has to
        reach the fleet, and converging on the strength of "no merges
        changed anything" would strand it.

        CALLED BY: MOACBBAAllocator.allocate, once per robot per round,
        immediately before createBundle.
        """
        me = robot.robot_id
        others_c = self.othersCompletion(me)

        def expected(task):
            return self.expectedSharers(task, me, seats_fn(task))

        moved = False
        for j in list(self.bundle):
            if j == robot.task_id or j not in self.path:
                continue
            task = model.tasks.get(j)
            without = [t for t in self.path if t != j]
            base_t, base_e = self.pathCost(model, robot, without, expected)
            if base_t >= INF:
                continue                 # counterfactual unpriceable: leave it
            aff = self.capacityAffinity(robot, task, c_max, v_max, kappa)
            cost = self.marginalCost(model, robot, self.path, base_t,
                                     base_e, others_c, aff, expected)
            old = self.bids.get((j, me), (RELEASED, -1))[0]
            if cost >= RELEASED:
                if old < RELEASED:
                    self.release(j, me)
                    moved = True
                continue
            # Hysteresis: only a material move is worth a fresh stamp and
            # a broadcast to every peer.
            if old >= RELEASED or \
                    abs(cost - old) > REPRICE_EPS * max(1.0, abs(old)):
                self.place(j, me, cost)
                moved = True
        return moved

    def releaseOutbid(self, robot, seats_fn, model) -> None:
        """Losing a seat invalidates every marginal cost after it, so the
        tail goes too (Choi et al. Eq. 6).

        The MOA-CBBA counterpart of CBBAAgent._releaseOutbid, with the
        same reasoning -- every bid after the lost one was priced as a
        marginal addition to a schedule that no longer exists -- but
        testing seat membership (`seated`) rather than single ownership.

        The task currently under execution is exempt and is preserved
        even if it falls inside the released tail: the robot is
        physically in that hole.

        CALLED BY: MOACBBAAllocator.allocate, after each round's
        consensus merge.
        """
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
        """Drop finished tasks from the bundle and path, releasing their
        seats.

        Bundles persist across ticks, so without this they accumulate ids
        of tasks that no longer exist. Note this releases the bid as well
        as removing the entry -- a live bid on a finished task would keep
        occupying a seat in every robot's view of the bid table.

        CALLED BY: MOACBBAAllocator.allocate, per robot, before bidding.
        """
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
        """c_max over every OTHER robot -- the fleet's current bottleneck
        finish time, excluding me.

        MATH:  others_c = max over k != me of c_k

        This is the number marginalCost compares my schedule against to
        decide whether taking another task actually delays the fleet.
        Values arrive by gossip (resolveConflicts stores each peer's
        broadcast c), so it is a BELIEF that may lag reality by a round.
        Zero when nothing has been heard yet, which makes every robot
        look like the bottleneck and therefore bid conservatively.

        CALLED BY: createBundle, once per round.
        """
        vals = [c for k, c in self.completions.items() if k != me]
        return max(vals) if vals else 0.0

    def broadcast(self, robot, now, sharers_fn=None) -> None:
        """Send my whole bid table plus my projected completion time c_i.

        Two payload fields:
          "bids" -- every (task, robot) -> (cost, stamp) entry I hold,
                    keyed as "j:i" strings because the payload is a plain
                    dict that gets deep-copied through the network.
          "c"    -- my projected completion time, recomputed here so it
                    reflects the bundle I just built.

        c_i MUST be computed on the same volume basis as the bids
        (hence sharers_fn), or peers load-balance against a schedule that
        does not exist.

        CALLED BY: MOACBBAAllocator.allocate, per robot per round.
        """
        self._now = now
        self.myCompletion, _e = self.pathCost(robot.model, robot, self.path,
                                              sharers_fn)
        robot.send({"sender": robot.robot_id,
                    "bids": {f"{j}:{i}": v for (j, i), v in self.bids.items()},
                    "c": self.myCompletion})

    def resolveConflicts(self, robot, messages, now) -> bool:
        """Merge peers' bid tables into mine -- MOA-CBBA's phase 2.

        Far simpler than CBBA's Table I, because the state is richer: a
        per-(task, robot) table with timestamps needs no case analysis,
        only a merge rule per key:

            adopt theirs  iff  I have no entry
                          OR   stamp_k > stamp_mine        (newer wins)
                          OR   stamp_k == stamp_mine
                               AND cost_k < cost_mine      (deterministic
                                                            tie-break)

        The timestamp is what makes a RELEASE survive contact with a peer
        that has not heard it yet -- without it the peer echoes the dead
        claim straight back and the robot re-adopts a task it dropped.
        The equal-stamp tie-break on cost keeps two robots from
        disagreeing about the same key forever.

        Peer completion times c_k are absorbed here too, feeding
        othersCompletion and therefore the makespan term of every
        subsequent bid.

        RETURNS True if anything changed (drives the convergence check).

        CALLED BY: MOACBBAAllocator.allocate, per robot per round.
        """
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
                 switch_criterion: str = SWITCH_CRITERION,
                 enable_switching: bool = ENABLE_SWITCHING,
                 max_adds_per_round: int | None = MAX_ADDS_PER_ROUND,
                 reprice_held: bool = REPRICE_HELD) -> None:
        self.max_adds_per_round = max_adds_per_round
        self.reprice_held = bool(reprice_held)
        self.n_repriced = 0        # rounds in which some bid moved
        self.obstacle_halo = bool(obstacle_halo)
        self.max_sharers = int(max_sharers)
        self.min_share = float(min_share)
        self.capacity_affinity = float(capacity_affinity)
        self.switch_margin = float(switch_margin)
        if switch_criterion not in ("rtlff", "margin"):
            raise ValueError("switch_criterion must be 'rtlff' or 'margin', "
                             f"got {switch_criterion!r}")
        self.switch_criterion = switch_criterion
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

        MECHANISM 4's seat budget. MATH -- the minimum of three caps:

            seats = max(1, min( max_sharers,                  policy
                                floor(V_j / min_share),       economics
                                |work_candidates(l_j)| ))     physics

        Each cap answers a different objection to piling robots onto one
        task: policy is the configured ceiling (1 reproduces
        single-robot-per-task exactly); the volume cap stops a robot
        being seated for a sliver of work not worth the drive; and the
        work-cell cap is hard physics -- there are at most 9 cells
        adjacent to a target, fewer once bedrock, dumps, hazards and
        parked robots are subtracted, and seating more robots than there
        are places to stand just produces collisions and STUCK_LIMIT
        waits.

        CALLED BY: MOACBBAAllocator.allocate, wrapped as `seats_fn` and
        passed down into createBundle, releaseOutbid and _execute.
        """
        if self.max_sharers <= 1:
            return 1
        by_volume = int(task.remaining // self.min_share)
        free = len(work_candidates([task.cell], model.grid.width,
                                   model.grid.height, model.blocked_cells()))
        return max(1, min(self.max_sharers, by_volume, free))

    # ---------------------------------------------------------------- #
    def allocate(self, model) -> None:
        """Run one full MOA-CBBA auction and seat whoever won.

        Same four-stage shape as CBBAAllocator.allocate -- setup, rounds
        of (bid, broadcast, deliver, resolve), then execute -- with these
        differences:

          - Candidate lists come from tasks.seats_open, so they include
            tasks the robot is already travelling to (which is what makes
            an en-route switch expressible as an ordinary auction result)
            and tasks that already have other robots on them.
          - releaseOutbid runs after every round's merge.
          - The round budget uses N_min = min(|tasks| * max_sharers,
            |robots| * L_t), scaled by max_sharers because there are that
            many more seats to settle.
          - There is no _trigger short-circuit: MOA-CBBA re-auctions
            every tick, since gossiped completion times change
            continuously as robots work.
          - Execution is delegated to _execute, which handles seat caps
            and optional en-route switching.

        CALLED BY: model.step, once per tick, via model.allocator.
        """
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
            repriced = []
            for robot in robots:
                agent = self._agent(robot)
                # Re-value what this robot already holds BEFORE deciding
                # what to add: the marginal cost of a new task is
                # measured against the existing path, so that path has to
                # be priced against the current world first.
                if self.reprice_held:
                    repriced.append(
                        agent.repriceHeld(model, robot, seats_fn, c_max,
                                          v_max, self.capacity_affinity))
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
            if any(repriced):
                self.n_repriced += 1
            # A re-priced bid is new information that has not reached the
            # fleet yet, so it counts as a change: converging on "no
            # merge altered anything" alone would strand it for a tick.
            if not (any(changed) or any(repriced)):
                self.converged = True
                break

        self._execute(model, robots, seats_fn)

    # ---------------------------------------------------------------- #
    def _execute(self, model, robots, seats_fn) -> None:
        """Turn the settled auction into actual assignments.

        TWO PHASES per robot:

        1. EN-ROUTE SWITCHING (mechanism 4, OFF by default). A robot
           still merely travelling to a task -- can_abandon, i.e. stage
           TO_TASK with an empty hopper, Das et al. Sec. 3.7.4 -- may
           swap to a cheaper task it has since won. Guarded three ways:

               model.tick - switched_tick >= SWITCH_LOCKOUT   (15 ticks)
               cost_new < cost_current * (1 - switch_margin)  (20%)
               and both costs re-priced HERE, from the same position

           The lockout and margin are not redundant: the margin compares
           two costs at one instant, but both move every tick as the
           robot walks, so a margin alone cannot stop a cycle. See the
           ENABLE_SWITCHING note above for why this is disabled by
           default (measured worse on both objectives).

        2. SEATING. An idle robot walks its path and takes the first task
           where it holds a seat AND the task has room. The seat cap is
           enforced against the REGISTRY (task.assignees), not the local
           bid table: mid-auction robots' views differ, so k robots can
           each believe they hold one of the k cheapest seats, and only
           the task itself knows how many have actually sat down.

        If assign() refuses (no free work cell, no reachable dump), the
        task is dropped from the bundle AND the seat released -- keeping
        the bid alive would block a robot that could have taken it.

        CALLED BY: allocate, as its final stage.
        """
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
                better = None

                if self.switch_criterion == "rtlff":
                    # Parker & Gini RT-LFF: give up the current task only
                    # if it still finishes FIRST without me. Among the
                    # candidates that clear that bar, take the nearest --
                    # their Algorithm RT-LFF picks argmin travel time, not
                    # the latest-finishing target, so a switch never buys
                    # balance with a long drive.
                    ct_minus = agent.finishIfLeft(model, robot, cur_task)
                    for j in agent.path:
                        if j == robot.task_id:
                            continue
                        task = model.tasks.get(j)
                        seats = seats_fn(task)
                        if not agent.seated(j, me, seats):
                            continue
                        if len(task.assignees) >= seats:
                            continue        # no seat actually free
                        ct_plus = agent.finishIfJoined(model, robot, task,
                                                       seats)
                        if ct_plus >= INF:
                            continue        # unpriceable, NOT "finishes late"
                        if ct_minus < ct_plus \
                                and (better is None or ct_plus < better[0]):
                            better = (ct_plus, j)
                else:
                    # Legacy unilateral margin test, kept for the ablation.
                    # Both sides priced HERE, from the same position, on
                    # the same basis. Reading the stored bid for the
                    # current task compared a value computed at assignment
                    # time against rivals repriced this tick, so the
                    # comparison drifted in favour of switching a little
                    # more every tick.
                    cur_k = max(1, expected(cur_task))
                    cur_tau, cur_e, cur_q = leg_cost(
                        model, robot, cur_task,
                        volume=cur_task.remaining / cur_k)
                    current = (INF if cur_q is None
                               else model.w1 * cur_tau + model.w2 * cur_e)
                    for j in agent.path:
                        if j == robot.task_id:
                            continue
                        task = model.tasks.get(j)
                        if not agent.seated(j, me, seats_fn(task)):
                            continue
                        if len(task.assignees) >= seats_fn(task):
                            continue        # no seat actually free
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