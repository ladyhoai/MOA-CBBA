"""Consensus Based Parallel Auction and Execution (Das et al., 2015).

The defining feature of CBPAE, and the thing the previous version did
not have, is that a robot bids on its NEXT task while EXECUTING its
current one, and the bid improves as that execution progresses (Sec.
3.7.1, Eq. 7). Everything else in the paper -- the silent-auction
framing, the bidding window, the drop/reallocate machinery -- exists to
support that. Four things were missing or wrong and are fixed here:

  1. Eq. (7). Bids now carry the residual effort of the current task
     (bidding.residual_cost), so they strictly improve round over round.
     Previously the bid was computed from a fixed start position and
     never moved, so the "multi-round dynamic bidding" was static.
  2. Eq. (12), HT. A robot may overbid ANY task whose incumbent bid it
     beats, whoever holds it -- that is the point of a silent auction.
     The old ownership filter (`skip if someone else won it`) made the
     first bid permanent.
  3. The five task vectors of Sec. 3.5 (B, A, TB, TD, E) and the six
     consensus actions of Table 4. Without bid time and drop time,
     Tables 5 and 6 collapse into plain max-consensus; without the
     execution-status vector there is no Release and no DROP.
  4. The bidding window and message counter (Sec. 3.7.3), which is what
     the paper relies on to guarantee that the same task is never
     assigned to two robots.

Domain notes (record these in the report as deliberate scope choices):
  - Task PRIORITIES (l_m, L_FT) do not exist in excavation; every chunk
    is equal-priority, so Eq. (9)'s priority filter is a no-op and
    Eq. (10) is omitted.
  - SKILLS and EXPERTISE (lambda_np, beta_nj, Eqs. 6 and 11) collapse to
    a single "excavate" skill that every robot has, so beta == 1 for
    every (robot, task) pair. Heterogeneity lives in RobotSpec instead
    and enters through the cost model, not through a skill vector.
  - The paper drops a task when an EMERGENCY task appears. Excavation
    has no emergencies, but Phase 4 dynamics give the same event a
    different name: a hazard zone can grow over the route to a target
    and make it unreachable. That triggers the drop path here, so the
    DROP status and drop-time vector do real work in this domain.

Bid convention: LOWER IS BETTER (Sec. 3.1). Tables 5 and 6 are written
as b_k < b_n, so transcribing them faithfully requires it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .bidding import (EPS, INF, NOCOST, Stage, _cheaper_bid, leg_cost,
                      residual_cost)

if TYPE_CHECKING:
    from .model import ExcavationModel
    from .robot import ExcavatorRobot

# --- Table 3: possible values of the task vectors ------------------- #
NBID = None      # a_{n,m}: no bid on the task yet
NDRP = -1        # td_{n,m}: execution has not been dropped
NOTB = -1        # tb_{n,m}: no bid time yet

NALC = "NALC"    # not allocated yet
EXEC = "EXEC"    # being executed currently
FNSH = "FNSH"    # execution finished
DROP = "DROP"    # was allocated, robot dropped the execution

# Sec. 3.7.3: the bidding window is a fixed interval after the current
# highest bid was placed. Its job is to let the fleet converge before
# anyone commits, so that two robots never start the same task.
BID_WINDOW = 2


class CBPAEAgent:
    """One robot's local copy of the five task vectors (Sec. 3.5)."""

    def __init__(self) -> None:
        self.B: dict[int, float] = {}   # bid value   b_{n,m}
        self.A: dict[int, int] = {}     # allocation  a_{n,m}
        self.TB: dict[int, int] = {}    # bid time    tb_{n,m}
        self.TD: dict[int, int] = {}    # drop time   td_{n,m}
        self.E: dict[int, str] = {}     # exec status e_{n,m}

        self.bidTask = None        # tBid(t)
        self.prevBidTask = None    # tBid(t-1)
        self.execTask = None       # tExec(t)
        self.prevExecTask = None   # tExec(t-1)

        # Sec. 3.7.2/3.7.3: messages processed without losing the bid.
        self.msg_count = 0
        # what tExec(t) was won for, so a task that has become far more
        # expensive than it was bid at can be recognised (dropIfTooCostly)
        self.execBid = NOCOST

    # -------------------------------------------------------------- #
    # accessors with Table 3 defaults
    # -------------------------------------------------------------- #
    def _bid(self, j) -> float:
        return self.B.get(j, NOCOST)

    def _winner(self, j):
        return self.A.get(j, NBID)

    def _tb(self, j) -> int:
        return self.TB.get(j, NOTB)

    def _td(self, j) -> int:
        return self.TD.get(j, NDRP)

    def _status(self, j) -> str:
        return self.E.get(j, NALC)

    def _entry(self, j) -> tuple:
        """The six values broadcast per task (Fig. 3)."""
        return (self._winner(j), self._bid(j), self._status(j),
                self._tb(j), self._td(j))

    # -------------------------------------------------------------- #
    # Sec. 3.7.1: bidding
    # -------------------------------------------------------------- #
    def freeTasks(self, model) -> list:
        """FT, Eq. (8): tasks whose execution status is NALC or DROP."""
        return [t for t in model.tasks.all
                if not t.done and self._status(t.task_id) in (NALC, DROP)]

    def biddableTasks(self, model, robot) -> list:
        """BT, Eq. (9). The priority filter (l_j <= L_FT) is a no-op in
        this domain and beta_nj == 1 for every pair, so the only real
        constraint left is reachability."""
        out = []
        startPos = self._bidStartPosition(robot)
        for t in self.freeTasks(model):
            tau, e, q = leg_cost(model, robot, t, startPos)
            if q is not None:
                out.append((t, tau, e))
        return out

    @staticmethod
    def _bidStartPosition(robot):
        """Where the robot will be when it can start the next task: at
        its dump site if it is mid-task, otherwise where it stands."""
        if robot.task_id is not None and robot.dump_cell is not None:
            return robot.dump_cell
        return None

    def computeBid(self, model, robot, now: int):
        """One iteration of Fig. 2.

        v_n,m = (residual effort on the current task) + (effort for m),
        which is the excavation instance of Eq. (7): the skills common
        to both tasks are the haul-and-dig cycle, and only the UNFINISHED
        part of the current task counts. As execution progresses the
        first term shrinks, so the bid improves every round -- the
        mechanism the whole paper is built on.
        """
        self.prevBidTask = self.bidTask
        r_tau, r_e = residual_cost(model, robot)
        residual = model.w1 * r_tau + model.w2 * r_e

        # HT, Eq. (12): every task whose incumbent bid I can beat.
        # Ownership is deliberately NOT a filter here.
        best, best_v = None, INF
        for t, tau, e in self.biddableTasks(model, robot):
            j = t.task_id
            v = residual + model.w1 * tau + model.w2 * e
            if self._winner(j) != robot.robot_id:
                if not _cheaper_bid(v, robot.robot_id,
                                    self._bid(j), self._winner(j)):
                    continue
            if v < best_v - EPS:
                best, best_v = j, v

        self.bidTask = best

        # Fig. 2: if tBid(t) != tBid(t-1), release the old bid first.
        if self.prevBidTask is not None and self.prevBidTask != self.bidTask:
            self.release(self.prevBidTask, robot.robot_id, now)
        if self.bidTask is not None:
            self.placeBid(self.bidTask, best_v, robot.robot_id, now)
        return self.bidTask, best_v

    def placeBid(self, j, value, me, now) -> None:
        """Record my bid on j (Table 3 fields b, a, tb).

        Two things are deliberately NOT reset when I merely IMPROVE a bid
        I already hold, and both of them are load-bearing for Eq. (7).

        `msg_count` counts messages processed WITHOUT LOSING the bid
        (Sec. 3.7.2). Improving your own bid loses nothing. Zeroing it on
        every rewrite made the paper's central mechanism self-defeating:
        residual_cost shrinks every round while a robot executes, so a
        mid-execution bidder rewrote its bid every round, zeroed the
        counter every round, and could not reach tryAssign's quorum until
        it went idle and its bid finally went static. The visible symptom
        is a ~3-tick dead gap after every completion before the next
        assignment -- paid once per chunk, by every robot, all run.

        `tb` opens the bidding window, which exists to give rivals time to
        contest (Sec. 3.7.3). A strictly better bid from the same holder
        gives them nothing new to contest, so restarting the window on an
        improvement only delays the commit. A WORSE bid can be beaten, so
        that one does re-open it.
        """
        if self._winner(j) == me and abs(self._bid(j) - value) <= EPS:
            return                       # unchanged: no vector edit
        fresh_claim = self._winner(j) != me
        worse = (not fresh_claim) and value > self._bid(j) + EPS
        self.B[j] = value
        self.A[j] = me
        if fresh_claim or worse or self._tb(j) == NOTB:
            self.TB[j] = now
        if self._status(j) not in (NALC, DROP):
            self.E[j] = NALC
        if fresh_claim or worse:
            self.msg_count = 0

    def release(self, j, me, now: int | None = None) -> None:
        """Table 4, Release: b = NBID, a = NBID, e = NALC.

        The release must carry a FRESH bid time. Table 6's NALC/NALC row
        adopts a sender's `a_k is NBID` only when tb_k >= tb_n, so a
        release broadcast with the ORIGINAL bid's timestamp loses every
        race against a receiver that has since bumped tb_n through
        updatebidtime. The releasing robot then walks away while every
        other robot goes on believing it still holds the task at a bid
        nobody can beat. The chunk is never bid on again, no assign is
        ever attempted, and the fleet is in perfect agreement about a
        fiction -- which is the one failure mode Tables 5-7 cannot fix,
        because there is no disagreement left to resolve.

        `now` is optional only so that an external caller cannot break;
        every internal call site passes it.
        """
        if self._winner(j) != me:
            return
        self.B[j] = NOCOST
        self.A[j] = NBID
        self.E[j] = NALC
        if now is not None:
            self.TB[j] = now
        if self.bidTask == j:
            self.bidTask = None

    # -------------------------------------------------------------- #
    # Sec. 3.7.2: communication
    # -------------------------------------------------------------- #
    def broadcast(self, robot, now: int) -> None:
        """Fig. 3: exactly four tasks -- tBid(t), tBid(t-1), tExec(t),
        tExec(t-1) -- so the message size is constant in the number of
        robots and tasks. That constancy is the paper's bandwidth
        argument (Sec. 4.2) and it only holds if we broadcast these four
        and nothing else."""
        four = [self.bidTask, self.prevBidTask,
                self.execTask, self.prevExecTask]
        tasks = {j: self._entry(j) for j in four if j is not None}
        robot.send({"sender": robot.robot_id, "tasks": tasks})

    def consensus(self, robot, messages, now: int) -> bool:
        """Apply Tables 5, 6 and 7 to every task in every message."""
        n = robot.robot_id
        changed = False
        for msg in messages:
            payload = msg.payload
            k = payload.get("sender")
            if k is None or k == n:
                continue
            for j, entry in payload.get("tasks", {}).items():
                a_k, b_k, e_k, tb_k, td_k = entry
                if self.execTask == j and self._status(j) == EXEC:
                    actions = self._parallelExecutionRule(n, a_k, e_k)
                elif self.bidTask == j:
                    actions = self._bidTaskRule(n, j, a_k, b_k, e_k,
                                                tb_k, td_k)
                else:
                    actions = self._otherTaskRule(n, j, a_k, b_k, e_k,
                                                  tb_k, td_k)
                if self._apply(actions, robot, j, entry, now):
                    changed = True
            self.msg_count += 1
        return changed

    # ---- Table 5: consensus rules for the Bid Task ---------------- #
    def _bidTaskRule(self, n, j, a_k, b_k, e_k, tb_k, td_k) -> tuple:
        e_n = self._status(j)
        b_n, tb_n, td_n = self._bid(j), self._tb(j), self._td(j)

        if e_n == NALC:
            if e_k == NALC:
                if a_k is NBID:
                    return ("updatebidtime",) if tb_k > tb_n else ()
                if a_k == n:
                    return ()                                  # Leave
                if b_k < b_n - EPS:
                    return ("release", "update")
                if abs(b_k - b_n) <= EPS and tb_k > tb_n:
                    return ("release", "update") if a_k < n else ("updatebidtime",)
                if b_k > b_n + EPS and tb_k > tb_n:
                    return ("updatebidtime",)
                return ()
            if e_k == DROP:
                return ("release", "update")
            if e_k in (EXEC, FNSH) and a_k is not NBID:
                return ("release", "update")
            return ()

        if e_n == DROP:
            if e_k == DROP:
                if a_k is NBID:
                    if td_k > td_n:
                        return ("updatetime",)
                    return ("updatebidtime",) if tb_k > tb_n else ()
                if a_k == n:
                    return ()
                if b_k < b_n - EPS:
                    return ("release", "update")
                if abs(b_k - b_n) <= EPS and tb_k > tb_n:
                    if a_k < n:
                        return ("release", "update")
                    return ("updatetime",) if td_k > td_n else ("updatebidtime",)
                if b_k > b_n + EPS and tb_k > tb_n:
                    return ("updatetime",) if td_k > td_n else ("updatebidtime",)
                return ()
            if e_k == EXEC and a_k is not NBID:
                return ("release", "update") if tb_k > td_n else ()
            if e_k == FNSH and a_k is not NBID:
                return ("release", "update")
            return ()
        return ()

    # ---- Table 6: all other tasks --------------------------------- #
    def _otherTaskRule(self, n, j, a_k, b_k, e_k, tb_k, td_k) -> tuple:
        e_n = self._status(j)
        a_n, tb_n, td_n = self._winner(j), self._tb(j), self._td(j)
        UPD = ("update",)

        if e_k == EXEC and a_k is not NBID:
            if e_n == EXEC:
                if a_k != a_n and (tb_k > tb_n or td_k > td_n):
                    return UPD
                if a_n is not NBID and a_k < a_n:
                    return UPD
                return ()
            if e_n == NALC:
                return UPD
            if e_n == DROP:
                return UPD if tb_k >= td_n else ()
            return ()

        if e_k == FNSH and a_k is not NBID:
            return UPD                       # EXEC / NALC / DROP alike

        if e_k == NALC:
            if e_n != NALC:
                return ()
            if a_k is NBID:
                return UPD if tb_k >= tb_n else ()
            if a_k == n:
                return ()                                      # Leave
            return UPD if tb_k > tb_n else ()

        if e_k == DROP:
            if e_n == EXEC:
                return UPD if td_k > td_n else ()
            if e_n == NALC:
                return UPD
            if e_n == DROP:
                if a_k is NBID:
                    return UPD if td_k > td_n else ()
                if a_k == n:
                    return ()
                if td_k > td_n:
                    return UPD
                return UPD if (td_k == td_n and tb_k > tb_n) else ()
            return ()
        return ()

    # ---- Table 7: parallel execution detection -------------------- #
    def _parallelExecutionRule(self, n, a_k, e_k) -> tuple:
        """Two robots executing the same task -- only possible after a
        communication dropout. The higher-indexed robot backs off."""
        if e_k == FNSH:
            return ("stopexecution", "update")
        if e_k == EXEC and a_k is not NBID and a_k < n:
            return ("stopexecution", "update")
        return ()

    # ---- Table 4: the six actions --------------------------------- #
    def _apply(self, actions, robot, j, entry, now) -> bool:
        if not actions:
            return False                                        # Leave
        a_k, b_k, e_k, tb_k, td_k = entry
        n = robot.robot_id
        changed = False

        if "stopexecution" in actions:
            if robot.task_id == j:
                robot.abandon_task()
                self.prevExecTask, self.execTask = j, None
            changed = True

        if "release" in actions:
            self.release(j, n, now)
            self.msg_count = 0
            changed = True

        if "update" in actions:
            if (self._winner(j) != a_k or self._status(j) != e_k
                    or abs(self._bid(j) - b_k) > EPS):
                changed = True
            self.A[j], self.B[j] = a_k, b_k
            self.E[j], self.TB[j], self.TD[j] = e_k, tb_k, td_k
            if self.bidTask == j and a_k != n:
                self.bidTask = None
                self.msg_count = 0

        if "updatebidtime" in actions:
            self.TB[j] = now
            changed = True

        if "updatetime" in actions:
            self.TB[j] = now
            self.TD[j] = td_k
            changed = True

        return changed

    # -------------------------------------------------------------- #
    # execution status bookkeeping
    # -------------------------------------------------------------- #
    def syncExecution(self, model, robot, now: int) -> None:
        """Keep the E vector in step with what the robot is physically
        doing. Replaces the old `task.future_remaining -= capacity`
        hack, which was never restored when a task finished or was
        abandoned and so leaked capacity out of the pool."""
        for t in model.tasks.all:
            if t.done and self._status(t.task_id) != FNSH:
                self.E[t.task_id] = FNSH
                if self.bidTask == t.task_id:
                    self.bidTask = None

        if robot.task_id is not None:
            j = robot.task_id
            if self.execTask != j:
                self.prevExecTask = self.execTask
                self.execTask = j
            self.E[j] = EXEC
            self.A[j] = robot.robot_id
        elif self.execTask is not None:
            j = self.execTask
            self.prevExecTask = j
            if model.tasks.get(j).done:
                self.E[j] = FNSH
            else:
                # I stopped executing WITHOUT finishing. Leaving e = EXEC
                # here is fatal and silent: the chunk is unassigned in the
                # registry but excluded from freeTasks forever, so no bid
                # is placed, tryAssign is never reached, and no ASSIGN-FAIL
                # is ever logged. Worse, prevExecTask keeps BROADCASTING
                # that EXEC, and Table 6 (e_k == EXEC over e_n == NALC ->
                # Update) spreads it to the whole fleet. Every robot then
                # agrees the chunk is being worked and consensus defends
                # the fiction indefinitely.
                #
                # robot.py reaches this state without ever calling
                # abandon_task(): _go_dump() calls _finish_task() when no
                # dump is reachable, and _dig_tick() does the same for
                # non-diggable terrain. Both clear task_id with volume
                # still in the ground. Table 4 calls that a Drop, so this
                # records a Drop -- with a drop time, which is what lets
                # Tables 5 and 6 propagate it.
                self.E[j] = DROP
                self.A[j] = NBID
                self.B[j] = NOCOST
                self.TD[j] = now
            self.execBid = NOCOST
            self.execTask = None

        self._reconcile(model, robot, now)

    def _reconcile(self, model, robot, now: int) -> None:
        """Repair local vectors against what the robot can actually see.

        Tables 5-7 resolve DISAGREEMENT. A vector that is wrong the same
        way on every robot is converged, and consensus will defend it for
        the rest of the run. These two rules are the ground truth: an
        empty dig site is visible from the site, and tryAssign already
        reads task.assigned_to directly, so this adds no omniscience that
        was not in the file already.
        """
        for t in model.tasks.unfinished:
            j = t.task_id
            if t.assigned_to is None:
                if self._status(j) == EXEC:      # nobody is working it
                    self.E[j] = NALC
                    self.A[j] = NBID
                    self.B[j] = NOCOST
                    self.TB[j] = now
            elif self._status(j) != EXEC:        # somebody demonstrably is
                self.E[j] = EXEC
                self.A[j] = t.assigned_to
                self.TB[j] = now
                if self.bidTask == j:
                    self.bidTask = None

    def dropIfUnreachable(self, model, robot, now: int) -> bool:
        """Sec. 3.7.4: a robot may abandon a task, but only during the
        first phase of execution, so the task state is untouched and it
        stays reallocatable.

        The paper's trigger is a new emergency task. Ours is Phase 4:
        a hazard or obstacle has made the target unreachable, so holding
        the task just blocks it for everyone."""
        if not robot.can_abandon:
            return False
        task = model.tasks.get(robot.task_id)
        _, _, q = leg_cost(model, robot, task)
        if q is not None:
            return False
        j = robot.task_id
        robot.abandon_task()
        self.E[j] = DROP
        self.A[j] = NBID
        self.B[j] = NOCOST
        self.TD[j] = now
        self.prevExecTask, self.execTask = j, None
        return True

    def dropIfTooCostly(self, model, robot, now: int,
                        ratio: float | None) -> bool:
        """Phase-4 generalisation of dropIfUnreachable. OFF by default.

        dropIfUnreachable only fires when the target has become
        IMPOSSIBLE. A hazard that merely forces a long detour leaves a
        robot crawling towards a chunk it would never have won at the new
        price, with the chunk reserved and everyone else idle. Checked in
        execution phase 1 only (via can_abandon), so the task state is
        untouched and it stays reallocatable -- the Sec. 3.7.4 condition.

        This is a SCOPE DECISION, not a bug fix. The paper's trigger is an
        emergency task and excavation has no priorities, so `ratio=None`
        reproduces the strict-paper behaviour exactly. If you switch it
        on, say so in the report and quote the ratio you used.
        """
        if ratio is None or not robot.can_abandon:
            return False
        if self.execBid >= NOCOST or self.execBid <= EPS:
            return False
        j = robot.task_id
        tau, e, q = leg_cost(model, robot, model.tasks.get(j))
        if q is None:
            return False          # unreachable is dropIfUnreachable's job
        if model.w1 * tau + model.w2 * e <= self.execBid * ratio:
            return False
        robot.abandon_task()
        self.E[j] = DROP
        self.A[j] = NBID
        self.B[j] = NOCOST
        self.TD[j] = now
        self.prevExecTask, self.execTask = j, None
        self.execBid = NOCOST
        return True

    # -------------------------------------------------------------- #
    # Sec. 3.7.3: task assignment
    # -------------------------------------------------------------- #
    def tryAssign(self, model, robot, now: int) -> bool:
        """Assign only after the bidding window has elapsed AND enough
        messages have been processed without losing the bid. This is
        what makes parallel allocation of one task impossible; without
        it two robots can both commit in the same tick."""
        j = self.bidTask
        if j is None or robot.task_id is not None:
            return False
        if self._winner(j) != robot.robot_id:
            return False
        if self._status(j) not in (NALC, DROP):
            return False
        task = model.tasks.get(j)
        if task.done or task.assigned_to is not None:
            self.E[j] = FNSH if task.done else EXEC
            if not task.done:
                # a stands for "who holds it", and that is now demonstrably
                # not me; leaving a = me here leaks a claim nobody can beat
                self.A[j] = task.assigned_to
            self.bidTask = None
            return False

        if now - self._tb(j) < BID_WINDOW:
            return False
        # Sec. 3.7.2: the counter is "messages from neighbouring robots
        # processed without losing the current bid", so the quorum is
        # the CURRENT neighbour count. A robot with no neighbours has
        # nobody to reach consensus with and its quorum is zero -- with
        # a floor of 1 it waits for a message that can never arrive, and
        # the whole fleet starves the moment comm_range isolates it.
        if self.msg_count < len(model.comms.neighbors(robot)):
            return False

        if not robot.assign(j):
            # Someone else physically got there first -- visible on
            # arrival, not telepathy, and the same premise as the
            # paper's parallel-execution detection.
            if not task.done and task.assigned_to is not None:
                self.E[j] = EXEC
                self.A[j] = task.assigned_to
            self.release(j, robot.robot_id, now)
            self.bidTask = None
            return False
        self.E[j] = EXEC
        self.execBid = self._bid(j)   # the price I won at, for dropIfTooCostly
        self.prevExecTask = self.execTask
        self.execTask = j
        self.bidTask = None
        self.msg_count = 0
        return True


class CBPAEAllocator:
    """Das et al. (2015): bidding continues during task execution."""

    name = "cbpae"

    def __init__(self, drop_cost_ratio: float | None = None) -> None:
        self.round = 0
        # None == strict paper behaviour; see dropIfTooCostly
        self.drop_cost_ratio = drop_cost_ratio

    def allocate(self, model: "ExcavationModel") -> None:
        self.round += 1
        now = self.round

        for r in model.robots:
            r.CBPAE.syncExecution(model, r, now)
            r.CBPAE.dropIfUnreachable(model, r, now)
            r.CBPAE.dropIfTooCostly(model, r, now, self.drop_cost_ratio)
            r.CBPAE.computeBid(model, r, now)
            r.CBPAE.broadcast(r, now)

        model.comms.flush_and_deliver(model.tick)

        for r in model.robots:
            r.CBPAE.consensus(r, r.receive_all(), now)

        for r in model.robots:
            r.CBPAE.tryAssign(model, r, now)