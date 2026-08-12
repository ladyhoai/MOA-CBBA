"""Pluggable task allocators.

Every allocator implements `allocate(model)`: inspect robots and tasks,
then call `robot.assign(task_id)` for each new pairing. This is the seam
where CBBA, CBPAE and MOA-CBBA slot in -- the model and metrics code
never change when you swap algorithms, which is what makes the
three-way comparison clean in batch runs.

CBBA here follows Choi, Brunet & How (2009). Four things previously
diverged from the paper and are fixed:

  1. The s vector (Eq. 5) was broadcast but never merged, so every
     s_im stayed at -1 and the `newer(m)` clause of Table I was
     effectively hardcoded True. Harmless at D = 1, wrong the moment
     comm_range is finite -- i.e. exactly when Phase 6 turns on.
  2. The score was not DMG and marginal gains could go negative, which
     breaks the convergence proof. The score now discounts by the
     accrued objective, making every gain strictly positive (Sec. VI-C),
     and Lemma 4's clamp c_ij(t) = min(c_ij(t), c_ij(t-1)) is applied
     as the belt-and-braces guarantee for non-DMG scores.
  3. Bundles were wiped every call and only path[0] was ever assigned,
     which is the L_t = 1 case -- i.e. CBAA, not CBBA (Sec. IV-B).
     Bundles now persist across ticks. Set robot.bundle_limit = 1 to get
     the CBAA behaviour back as an ablation.
  4. MAX_ROUNDS was an arbitrary 20; Theorem 1 bounds convergence at
     N_min * D.
  5. The s vector was written in two different clocks -- broadcast
     stamped the round index, mergeTimestamps stamped model.tick -- so
     newer(m) compared incomparable numbers. One monotone counter now
     serves both, incremented once per auction round for the whole run.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path

from .bidding import (EPS, INF, NOBID, Stage, _better_bid, bid_value,
                      leg_cost)
from .comms import network_diameter
from .CBPAE import CBPAEAllocator

# A task under execution cannot be taken away mid-dig, so its owner
# advertises an unbeatable (but finite -- inf poisons the arithmetic)
# winning bid. This is the CBBA-side equivalent of the EXEC status in
# CBPAE's execution-status vector.
LOCKED_BID = 1e9

# Time-discounted reward, Choi et al. Eq. (11).
LAMBDA = 0.99
TASK_REWARD = 10.0

############### helper functions for debugging and formatting ###############
CBBA_VERBOSE = int(os.environ.get("CBBA_VERBOSE", "1"))
CBBA_LOG_FILE = os.environ.get("CBBA_LOG_FILE", "cbba_debug.log")

_log = logging.getLogger("cbba")
_log.setLevel(logging.DEBUG)
_log.propagate = False  # keep it out of the root/stderr handler

if not _log.handlers:
    Path(CBBA_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    _handler = logging.FileHandler(CBBA_LOG_FILE, mode="w", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                            datefmt="%H:%M:%S"))
    _log.addHandler(_handler)


def _dbg(level, *args):
    if CBBA_VERBOSE >= level:
        _log.debug(" ".join(str(a) for a in args))


def _fmt_num(v):
    if v == float("inf"):
        return "   +inf"
    if v == float("-inf"):
        return "   -inf"
    return f"{v:7.3f}"


def _fmt_map(d):
    if not d:
        return "{}"
    items = []
    for k in sorted(d.keys()):
        v = d[k]
        items.append(f"{k}: {_fmt_num(v) if isinstance(v, float) else v}")
    return "{" + ", ".join(items) + "}"


def _check_consistency(tag, bundle, path, where):
    """The invariant CBBA relies on: path is a permutation of bundle."""
    ok = True
    if len(set(path)) != len(path):
        dupes = sorted({t for t in path if path.count(t) > 1})
        _dbg(1, f"{tag} !! DUPLICATE task ids in path {where}: {dupes}")
        ok = False
    if len(set(bundle)) != len(bundle):
        dupes = sorted({t for t in bundle if bundle.count(t) > 1})
        _dbg(1, f"{tag} !! DUPLICATE task ids in bundle {where}: {dupes}")
        ok = False
    if set(bundle) != set(path):
        _dbg(1, f"{tag} !! bundle/path MISMATCH {where}")
        _dbg(1, f"{tag}      in bundle, not in path: {sorted(set(bundle) - set(path))}")
        _dbg(1, f"{tag}      in path, not in bundle: {sorted(set(path) - set(bundle))}")
        ok = False
    return ok




#########################

class Allocator:
    name = "base"

    def allocate(self, model) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class GreedyAllocator(Allocator):
    """Each idle robot takes the cheapest pending task, in shuffled
    order. Cost convention: lower bid wins."""

    name = "greedy"

    def allocate(self, model) -> None:
        idle = [r for r in model.robots if r.task_id is None]
        model.random.shuffle(idle)
        for robot in idle:
            pending = model.tasks.pending
            if not pending:
                return
            bids = []
            for t in pending:
                v, q = bid_value(model, robot, t, model.w1, model.w2)
                if v != INF and q is not None:
                    bids.append((v, t))
            for _, t in sorted(bids, key=lambda vt: vt[0]):
                if robot.assign(t.task_id):
                    break


# ------------------------------------------------------------------ #
# CBBA scoring
# ------------------------------------------------------------------ #
def pathScoreCBBA(model, robot, path, lam: float = LAMBDA,
                  task_reward: float = TASK_REWARD) -> float:
    """S_i^{p_i}, Choi et al. Eq. (11), instantiated for Eq. (1) of the
    proposal.

    The paper discounts by ARRIVAL TIME alone. We discount by the
    accrued objective w1*tau + w2*E, so that the reward CBBA maximises
    is a monotone transform of the J(x) the experiment reports. The old
    version hardcoded its own task_reward / energy_weight and ignored
    model.w1/w2 entirely, so CBBA and CBPAE were optimising different
    objectives and the comparison between them meant nothing.

    Two properties matter and both survive the change:
      - NON-NEGATIVE: every term is task_reward * lam**(cost) > 0, so
        Sec. IV-A's assumption c_ij[b_i] >= 0 holds and inserting a task
        at the end of the path always yields a strictly positive gain.
        Marginal gains can no longer go negative.
      - DMG (Eq. 7): inserting a task can only increase the accrued cost
        at every later task in the path -- the Eq. (12) triangle-
        inequality argument, with cost in place of time -- so each
        existing term can only shrink. Gains therefore diminish.
    """
    clock = 0.0
    score = 0.0
    startPos = None
    for task_id in path:
        task = model.tasks.get(task_id)
        tau, e, dumpCoord = leg_cost(model, robot, task, startPos)
        if dumpCoord is None:
            return NOBID          # infeasible path scores nothing
        clock += model.w1 * tau + model.w2 * e
        score += task_reward * (lam ** clock)
        startPos = dumpCoord      # next leg starts where this one dumped
    return score


class CBBAAgent:
    """Consensus-Based Bundle Algorithm (Choi et al., 2009).

    State per Sec. IV-A: bundle b_i, path p_i, winning bids y_i,
    winning agents z_i, timestamps s_i.
    """

    def __init__(self):
        self.path = []
        self.task_list = []
        self.bundle = []            # b_i
        self.winningAgentList = {}  # z_i
        self.winningBidList = {}    # y_i
        self.timeStamp: dict[int, int] = {}  # s_i
        self._score_memo: dict[int, float] = {}   # Lemma 4 clamp

    def _bid(self, task_id) -> float:
        return self.winningBidList.get(task_id, NOBID)

    def _winner(self, task_id):
        return self.winningAgentList.get(task_id, None)

    def _stamp(self, agent_id) -> int:
        return self.timeStamp.get(agent_id, -1)

    # -------------------------------------------------------------- #
    # communication
    # -------------------------------------------------------------- #
    def broadcast(self, robot, now: int) -> None:
        """Broadcast (y, z, s) to neighbours."""
        self.timeStamp[robot.robot_id] = now
        robot.send({
            "sender": robot.robot_id,
            "y": dict(self.winningBidList),
            "z": dict(self.winningAgentList),
            "s": dict(self.timeStamp),
        })

    def mergeTimestamps(self, sender_id: int, other_s: dict, now: int) -> None:
        """Eq. (5): s_ik = tau_r for a direct neighbour k, and
        max over neighbours m of s_mk for everyone else.

        This was the missing half of the consensus. Without it every
        s_im stayed at its -1 default, `newer(m)` in the decision table
        was True whenever the sender knew anything at all about m, and
        roughly half of Table I collapsed into a single branch. It costs
        nothing at D = 1 and is load-bearing at any finite comm_range.
        """
        self.timeStamp[sender_id] = now
        for m, s in other_s.items():
            if m == sender_id:
                continue
            if s > self.timeStamp.get(m, -1):
                self.timeStamp[m] = s

    # -------------------------------------------------------------- #
    # Table I
    # -------------------------------------------------------------- #
    def _CBBADecisionTable(self, i, k, j, y_k, z_k, s_k) -> str:
        """Table I: receiver i's action on task j after hearing from k.

        Returns "update", "reset" or "leave" (the default).
        """
        zk = z_k.get(j)          # who the sender thinks won j
        zi = self._winner(j)     # who I think won j
        ykj = y_k.get(j, NOBID)  # the bid of agent k on task j
        yij = self._bid(j)       # my own bid on task j

        def newer(m) -> bool:
            """s_km > s_im: the sender's information about m is fresher."""
            return s_k.get(m, -1) > self._stamp(m)

        def better() -> bool:
            """y_kj beats y_ij (higher reward, tie-break on robot id)."""
            return _better_bid(ykj, zk, yij, zi)

        # ---- sender (k) thinks z_kj = k (it won the task itself) ----- #
        if zk == k:
            if zi == i:
                return "update" if better() else "leave"
            if zi == k:
                return "update"
            if zi is None:
                return "update"
            return "update" if (newer(zi) or better()) else "leave"   # zi = m

        # ---- sender thinks z_kj = i (I won it) ----------------------- #
        if zk == i:
            if zi == i:
                return "leave"
            if zi == k:
                return "reset"
            if zi is None:
                return "leave"
            return "reset" if newer(zi) else "leave"                  # zi = m

        # ---- sender thinks nobody won it ---------------------------- #
        if zk is None:
            if zi == i:
                return "leave"
            if zi == k:
                return "update"
            if zi is None:
                return "leave"
            return "update" if newer(zi) else "leave"                 # zi = m

        # ---- sender thinks z_kj = m, some third agent --------------- #
        m = zk
        if zi == i:
            return "update" if (newer(m) and better()) else "leave"
        if zi == k:
            return "update" if newer(m) else "reset"
        if zi is None:
            return "update" if newer(m) else "leave"
        if zi == m:
            return "update" if newer(m) else "leave"

        # ---- zi = n, a fourth agent --------------------------------- #
        n = zi
        if newer(m) and newer(n):
            return "update"
        if newer(m) and better():
            return "update"
        if newer(n) and self._stamp(m) > s_k.get(m, -1):
            return "reset"
        return "leave"

    # -------------------------------------------------------------- #
    # Eq. (6)
    # -------------------------------------------------------------- #
    def _releaseOutbid(self, i, locked=None) -> bool:
        """Eq. (6): if a task is outbid, everything added to the bundle
        after it must be released too, because removing b_{i,n_bar}
        changes the marginal score of every ensuing task.

        `locked` is the task currently under execution; it can never be
        released (the robot is physically standing in the hole), so the
        scan starts after it.
        """
        n_bar = None
        for n, task_id in enumerate(self.bundle):
            if task_id == locked:
                continue
            if self._winner(task_id) != i:
                n_bar = n
                break
        if n_bar is None:
            return False

        # b_{i,n_bar} keeps its (losing) y/z; everything strictly after
        # it that I still hold is reset -- Eq. (6) exactly.
        for task_id in self.bundle[n_bar + 1:]:
            if self._winner(task_id) == i:
                self.winningBidList[task_id] = NOBID
                self.winningAgentList[task_id] = None

        for task_id in self.bundle[n_bar:]:
            if task_id in self.path:
                self.path.remove(task_id)

        del self.bundle[n_bar:]
        return True

    # -------------------------------------------------------------- #
    # Phase 1: bundle construction (Algorithm 3)
    # -------------------------------------------------------------- #
    def _releaseStaleLocks(self, robot) -> None:
        """Give back the unbeatable bid once the lock is over.

        createBundle advertises y = LOCKED_BID (1e9) for the task under
        execution, and NOTHING ever took it back. On the normal path that
        is invisible: the task finishes, prune drops it as done and pops
        y/z with it. But a task that is RELEASED rather than finished --
        abandon_task, dropIfUnreachable, or robot.py's
        UNREACHABLE_PATIENCE -- is not done and is not assigned to
        anyone, so prune keeps it and the 1e9 survives.

        Every robot then fails `c_ij > y_ij` against 1e9 forever, so no
        robot can ever bid on that chunk again -- including the owner,
        which simply re-assigns it to itself off its own path next round.
        With the patience release in place that is a ~20-tick livelock on
        a chunk the owner has already proved it cannot reach.

        Resetting y/z here is enough: the following _releaseOutbid sees
        that the robot is no longer the winner and truncates the bundle
        from that point, which is Eq. (6) doing exactly its job.
        """
        for task_id, bid in list(self.winningBidList.items()):
            if bid != LOCKED_BID:
                continue
            if self.winningAgentList.get(task_id) != robot.robot_id:
                continue          # somebody else's lock; consensus owns it
            if robot.task_id == task_id:
                continue          # still in the hole: the lock is real
            self.winningBidList[task_id] = NOBID
            self.winningAgentList[task_id] = None

    def _pathScore(self, model, robot, path) -> float:
        """Seam for subclasses. Algorithm 3 below calls this rather than
        pathScoreCBBA directly, so a variant can change what a path is
        WORTH without reimplementing how bundles are built."""
        return pathScoreCBBA(model, robot, path)

    def _clamped_gain(self, task_id: int, raw: float) -> float:
        """Lemma 4: c_ij(t) = min(c_ij_raw(t), c_ij(t-1)).

        The paper proves CBBA converges to a conflict-free assignment
        within N_min*D iterations for ANY scoring scheme, DMG or not,
        provided the scores are made monotonically non-increasing over
        iterations this way. Reset once per auction episode, since the
        proof is over the iterations of one episode.
        """
        prev = self._score_memo.get(task_id, INF)
        c = min(raw, prev)
        self._score_memo[task_id] = c
        return c

    def createBundle(self, model, robot):
        """Algorithm 3. Operates on self.bundle / self.path directly --
        the old signature took y/z/bundle as arguments and then
        reassigned the members anyway, which is what the note in the
        original file was complaining about."""
        tag = f"[R{robot.robot_id}]"
        locked = robot.task_id
        known_ids = [t.task_id for t in self.task_list]

        _dbg(1, "=" * 78)
        _dbg(1, f"{tag} createBundle ENTRY  L_t={robot.bundle_limit}")
        _dbg(1, f"{tag}   bundle in : {self.bundle}")
        _dbg(1, f"{tag}   path   in : {self.path}")
        _dbg(1, f"{tag}   y         : {_fmt_map(self.winningBidList)}")
        _dbg(1, f"{tag}   z         : {_fmt_map(self.winningAgentList)}")
        _dbg(1, f"{tag}   task_list : {known_ids}")
        _check_consistency(tag, self.bundle, self.path, "on entry")

        iteration = 0
        while len(self.bundle) < robot.bundle_limit:
            inBundle = set(self.bundle)
            candidates = [t for t in self.task_list
                          if t.task_id not in inBundle]
            if not candidates:
                _dbg(1, f"{tag} STOP: no tasks left outside the bundle")
                break

            currentPathScore = self._pathScore(model, robot, self.path)

            _dbg(1, "-" * 78)
            _dbg(1, f"{tag} iter {iteration} | bundle={self.bundle} "
                    f"path={self.path} | S(path)={_fmt_num(currentPathScore)}")

            bestGain = 0.0     # h_ij also requires c_ij > 0 (Sec. IV-A)
            Ji = None
            JiPathLocation = None

            for cand in candidates:
                bestBidTask = NOBID
                bestTaskLocation = None
                for trialPosition in range(len(self.path) + 1):
                    trialPath = (self.path[:trialPosition]
                                 + [cand.task_id]
                                 + self.path[trialPosition:])
                    trialBid = self._pathScore(model, robot, trialPath)
                    if bestTaskLocation is None or trialBid > bestBidTask:
                        bestBidTask = trialBid
                        bestTaskLocation = trialPosition

                if bestTaskLocation is None:
                    continue
                gain = self._clamped_gain(cand.task_id,
                                          bestBidTask - currentPathScore)
                y_ij = self._bid(cand.task_id)

                # Eq. (4): h_ij = I(c_ij > y_ij), and c_ij >= 0
                if gain <= EPS or gain <= y_ij + EPS:
                    continue
                if gain > bestGain:
                    bestGain = gain
                    Ji = cand.task_id
                    JiPathLocation = bestTaskLocation

            if Ji is None:
                _dbg(1, f"{tag} STOP: no task beat its incumbent bid "
                        f"({len(self.bundle)}/{robot.bundle_limit})")
                break

            self.bundle.append(Ji)
            self.path.insert(JiPathLocation, Ji)
            prev_winner = self.winningAgentList.get(Ji)
            self.winningBidList[Ji] = bestGain
            self.winningAgentList[Ji] = robot.robot_id

            _dbg(1, f"{tag} ADD task {Ji} at path pos {JiPathLocation} "
                    f"| c={_fmt_num(bestGain)} | took it from z={prev_winner}")
            _check_consistency(tag, self.bundle, self.path, f"after add {Ji}")

            iteration += 1
            if iteration > len(self.task_list) + robot.bundle_limit + 1:
                _dbg(1, f"{tag} !! ABORT: while-loop is not terminating")
                break

        if locked is not None:
            self.winningBidList[locked] = LOCKED_BID
            self.winningAgentList[locked] = robot.robot_id

        _dbg(1, f"{tag} createBundle EXIT bundle={self.bundle} "
                f"path={self.path}")
        _check_consistency(tag, self.bundle, self.path, "on exit")
        return self.winningAgentList, self.winningBidList, self.bundle

    # -------------------------------------------------------------- #
    # Phase 2: conflict resolution (Algorithm 2 + Table I)
    # -------------------------------------------------------------- #
    def resolveConflicts(self, robot, receivedMessages, now: int) -> bool:
        """Returns True if anything changed, i.e. another round is
        needed before the fleet has converged.

        `now` is the auction clock, and it MUST be the same one
        broadcast() stamps with. It used to read model.tick here while
        broadcast stamped the round index, so s_i[me] counted 1, 2, 3...
        and s_i[neighbour] counted 85, 86... -- Eq. (5) comparing two
        different units. newer(m) was then meaningless, which matters in
        every third-agent branch of Table I, i.e. precisely when
        comm_range is finite and the fleet is not a complete graph."""
        i = robot.robot_id
        changed = False

        for msg in receivedMessages:
            payload = msg.payload
            k = payload.get("sender")
            if k is None or k == i:
                continue

            otherWinningBidList = payload.get("y", {})
            otherWinningAgentList = payload.get("z", {})
            otherTimestamps = payload.get("s", {})

            taskIDs = (set(self.winningBidList) | set(otherWinningBidList)
                       | set(self.winningAgentList)
                       | set(otherWinningAgentList))
            for taskID in taskIDs:
                if self.winningBidList.get(taskID) == LOCKED_BID \
                        and self.winningAgentList.get(taskID) == i:
                    continue        # my task is under execution: untouchable

                action = self._CBBADecisionTable(
                    i, k, taskID, otherWinningBidList,
                    otherWinningAgentList, otherTimestamps)

                if action == "update":
                    new_y = otherWinningBidList.get(taskID, NOBID)
                    new_z = otherWinningAgentList.get(taskID)
                    if new_z != self._winner(taskID) \
                            or abs(new_y - self._bid(taskID)) > EPS:
                        self.winningBidList[taskID] = new_y
                        self.winningAgentList[taskID] = new_z
                        changed = True

                elif action == "reset":
                    if self._winner(taskID) is not None \
                            or self._bid(taskID) != NOBID:
                        self.winningBidList[taskID] = NOBID
                        self.winningAgentList[taskID] = None
                        changed = True
                # "leave": do nothing

            # Eq. (5). Done after the table so the decisions above see
            # the sender's s_k against my PRE-merge s_i.
            self.mergeTimestamps(k, otherTimestamps, now)

        if self._releaseOutbid(i, locked=robot.task_id):
            changed = True

        _dbg(2, f"[R{i}] resolve -> bundle={self.bundle} path={self.path}")
        return changed

    # -------------------------------------------------------------- #
    def prune(self, model, robot) -> None:
        """Drop finished tasks, and tasks another robot is executing,
        from the bundle and path. With persistent bundles this is what
        keeps them from accumulating stale ids forever."""
        self._releaseStaleLocks(robot)
        keep = []
        for task_id in self.bundle:
            task = model.tasks.get(task_id)
            if task.done:
                continue
            if task.assigned_to is not None and task.assigned_to != robot.robot_id:
                continue
            keep.append(task_id)
        dropped = set(self.bundle) - set(keep)
        if dropped:
            self.bundle = keep
            self.path = [t for t in self.path if t not in dropped]
            for task_id in dropped:
                self.winningBidList.pop(task_id, None)
                self.winningAgentList.pop(task_id, None)


class CBBAAllocator(Allocator):
    """Centralised driver for a decentralised algorithm: the simulator
    steps every robot's local CBBA instance in lockstep. All the
    decision-making stays inside CBBAAgent."""

    name = "cbba"
    ROUND_CEILING = 200      # hard stop; the real bound is N_min * D
    # Which per-robot agent instance this allocator drives. Subclasses
    # point at their own attribute so two allocators never share state.
    AGENT_ATTR = "CBBA"

    def _agent(self, robot):
        return getattr(robot, self.AGENT_ATTR)

    def __init__(self) -> None:
        self.last_round = 0
        self.converged = False
        self._signature = None
        # Monotone across the WHOLE run, not per auction: round indices
        # restart at 1 every tick, so they cannot order information that
        # was learned in an earlier tick.
        self._clock = 0

    def _trigger(self, model, robots, open_tasks):
        """Re-auction only on a change in the situation: a robot freed
        up, a task finished, or the terrain moved (Algorithm 1 lines
        3-7). Holding a converged assignment between changes is the
        point of a time-extended allocation -- re-running the auction
        every tick would both burn A* calls and churn bundles that
        nothing has invalidated."""
        sig = (frozenset(r.robot_id for r in robots if r.task_id is None),
               frozenset(t.task_id for t in open_tasks),
               bool(model.changed_cells))
        if sig == self._signature and self.converged:
            return False
        self._signature = sig
        return True

    def allocate(self, model) -> None:
        robots = model.robots
        open_tasks = [t for t in model.tasks.unfinished]
        if not robots or not open_tasks:
            return
        if not self._trigger(model, robots, open_tasks):
            return

        for r in robots:
            agent = self._agent(r)
            agent.prune(model, r)
            # A robot may bid on any unfinished task that nobody else is
            # executing; its own current task stays visible so it keeps
            # winning it.
            agent.task_list = [
                t for t in open_tasks
                if t.assigned_to is None or t.assigned_to == r.robot_id]
            agent._score_memo = {}
            r.receive_all()          # drop anything stale in the inbox

        # Theorem 1: convergence within N_min * D iterations.
        L_t = max(r.bundle_limit for r in robots)
        n_min = min(len(open_tasks), len(robots) * L_t)
        # Floor of 2: convergence is DEFINED as a round in which nothing
        # changed, so a ceiling of 1 can never report it. Late in a run
        # n_min collapses to 1 (one open chunk), max_rounds became 1, and
        # every auction logged "did NOT converge" -- which also defeats
        # _trigger, since it only short-circuits on a converged result.
        max_rounds = min(self.ROUND_CEILING,
                         max(2, n_min * network_diameter(model)))

        _dbg(1, f"\n[AUCTION @ tick {model.tick}] {len(robots)} robots, "
                f"{len(open_tasks)} open tasks, max_rounds={max_rounds}")

        self.converged = False
        self.last_round = 0
        for rnd in range(1, max_rounds + 1):
            self._clock += 1
            for robot in robots:
                agent = self._agent(robot)
                agent.createBundle(model, robot)
                agent.broadcast(robot, self._clock)
            model.comms.flush_and_deliver(model.tick)
            changed = [self._agent(r).resolveConflicts(
                           r, r.receive_all(), self._clock)
                       for r in robots]
            self.last_round = rnd
            if not any(changed):
                self.converged = True
                break

        _dbg(1, f"[AUCTION] {'converged' if self.converged else 'HIT BOUND'} "
                f"after {self.last_round} round(s)")

        # Execute: an idle robot starts the first task in its path that
        # is still available. The rest of the bundle is kept, not thrown
        # away -- that is the whole point of a bundle algorithm.
        for robot in robots:
            if robot.task_id is not None:
                continue
            agent = self._agent(robot)
            for task_id in list(agent.path):
                if agent._winner(task_id) != robot.robot_id:
                    continue
                if robot.assign(task_id):
                    _dbg(1, f"[AUCTION]   R{robot.robot_id} starts {task_id}, "
                            f"bundle={agent.bundle}")
                    break
                # assign() refused: no reachable dig cell, or no dump
                # reachable from it. Dropping it from bundle/path while
                # LEAVING z pointing at this robot advertised a claim it
                # had already given up, and y_ij then blocked every other
                # robot from bidding on it.
                agent.path.remove(task_id)
                if task_id in agent.bundle:
                    agent.bundle.remove(task_id)
                agent.winningBidList[task_id] = NOBID
                agent.winningAgentList[task_id] = None


# MOACBBAAllocator registers ITSELF into this dict on import (see the
# bottom of MOACBBA.py). It subclasses CBBAAgent/CBBAAllocator, so this
# module cannot import it at the top, and a bottom-of-file import would
# fail whenever MOACBBA happened to be imported first. robot.py imports
# MOACBBAAgent, so the registration always happens before any model is
# built.
ALLOCATORS = {a.name: a for a in (GreedyAllocator, CBBAAllocator,
                                  CBPAEAllocator)}