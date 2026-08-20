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

New-reader primer: CBPAE's headline idea, and its main difference from
CBBA, is that a robot doesn't wait until it's idle to look for its next
job -- it bids on a NEXT task while still finishing its CURRENT one, and
that bid gets cheaper (more competitive) every round as the current task
gets closer to done (`residual_cost` in bidding.py is what shrinks). The
five per-task fields tracked in CBPAEAgent (bid value B, who holds it A,
bid time TB, drop time TD, execution status E) are exactly what the
paper calls the task vectors; `_bidTaskRule` / `_otherTaskRule` /
`_parallelExecutionRule` are its consensus Tables 5/6/7 -- the rules two
robots use to agree on the outcome after hearing from each other, the
same role Table I plays for CBBA in allocation.py.
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
        """FT, Eq. (8): tasks I believe are available to bid on.

        MATH:  FT = { t : not done AND e_t in {NALC, DROP} }

        i.e. unfinished tasks whose execution status is "not allocated
        yet" or "was allocated but dropped". Tasks I believe are EXEC
        (someone is digging) or FNSH (finished) are excluded.

        Note this filters on BELIEF (my E vector), not on ground truth --
        which is the whole point of a decentralised algorithm, and also
        why _reconcile exists to repair beliefs that have gone wrong in
        the same way on every robot.

        CALLED BY: biddableTasks.
        """
        return [t for t in model.tasks.all
                if not t.done and self._status(t.task_id) in (NALC, DROP)]

    def biddableTasks(self, model, robot) -> list:
        """BT, Eq. (9): the free tasks this robot can actually reach,
        each with its priced (tau, E).

        MATH: BT = { (t, tau, E) : t in FT and t is reachable }, where
        tau/E come from bidding.leg_cost priced FROM the position this
        robot will be in when it can start (see _bidStartPosition).

        In the original paper Eq. (9) also filters on task priority
        (l_j <= L_FT) and robot skill (beta_nj), but excavation has no
        priorities and every robot has the single "excavate" skill, so
        both collapse to no-ops and reachability is the only real
        constraint left.

        CALLED BY: computeBid.
        """
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
        its dump site if it is mid-task, otherwise where it stands.

        This is what makes bid-while-executing physically honest. A robot
        mid-task will finish at its dump cell q*, so the NEXT task's
        approach leg must be measured from there, not from where the
        robot happens to be standing right now.

        CALLED BY: biddableTasks.
        """
        if robot.task_id is not None and robot.dump_cell is not None:
            return robot.dump_cell
        return None

    def computeBid(self, model, robot, now: int):
        """Pick the single best task to bid on this round, and bid on it.

        MATH -- Eq. (7). For each biddable task m:

            v_m = residual + w1*tau_m + w2*E_m
            residual = w1*tau_res + w2*E_res     (bidding.residual_cost)

        `residual` is the effort still owed on the CURRENT task and is
        the same for every candidate, so it does not change the ranking
        WITHIN this robot -- but it does change how this robot compares
        against OTHER robots, which is the point. A robot nearly finished
        bids close to its raw task cost; a robot that just started bids
        much higher. And because residual shrinks every tick as work
        progresses, the same robot's bid on the same task improves round
        over round -- the mechanism the whole paper is built on.

        The robot then keeps the cheapest task it can actually WIN: for
        tasks it does not already hold, it must beat the incumbent bid
        (_cheaper_bid, Eq. 12's HT). Ownership is deliberately NOT a
        filter -- in a silent auction you may outbid anyone, and the old
        ownership check made whoever bid first the permanent holder.

        Finally (Fig. 2) if this round's choice differs from last
        round's, the previous bid is RELEASED before the new one is
        placed, so a robot never holds two claims at once.

        RETURNS (task_id or None, bid value).

        CALLED BY: CBPAEAllocator.allocate, once per robot per tick.
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
        """Reconcile my task vectors against everything I just heard.

        For each task in each message, ONE of three rule tables applies,
        chosen by what that task is to me:

          Table 7 (_parallelExecutionRule) -- I am executing it too.
              Only possible after a comms dropout; the higher-indexed
              robot backs off.
          Table 5 (_bidTaskRule)           -- it is the task I am
              bidding on. The contested case, so the rules are the most
              detailed: bid value first, then bid/drop times as
              tie-breakers.
          Table 6 (_otherTaskRule)         -- anything else. I have no
              stake, so these rules are essentially "believe the
              better-informed robot", ordered by execution status.

        Each table returns a tuple of ACTION NAMES, which _apply then
        carries out (Table 4's six actions: Leave, Update, Release,
        Stop Execution, Update Bid Time, Update Time).

        Also increments msg_count per message processed -- the quorum
        counter tryAssign uses to decide the fleet has had a fair chance
        to contest this bid.

        RETURNS True if anything changed.

        CALLED BY: CBPAEAllocator.allocate, after messages are delivered.
        """
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
        """Table 5: what to do about the task I am bidding on, having
        heard robot k's view of it.

        Arguments are the sender's five values for this task:
        a_k (who k thinks holds it), b_k (at what bid), e_k (execution
        status), tb_k (bid time), td_k (drop time). My own equivalents
        are read from my vectors as a_n/b_n/e_n/tb_n/td_n.

        STRUCTURE: split first by MY execution status (NALC or DROP --
        those are the only two a bid task can have), then by the
        SENDER's, then by bid comparison:

            b_k < b_n            -> the sender's bid is better; release
                                    mine and adopt theirs
            b_k = b_n, tb_k > tb_n -> tie on value, so break on robot
                                    index (a_k < n) as Table 5 specifies
            b_k > b_n, tb_k > tb_n -> mine is better but theirs is
                                    fresher; just bump my bid time

        The e_k in (EXEC, FNSH) rows are the important ones for
        correctness: if somebody is already digging or has finished the
        task I am bidding on, my bid is void and I adopt their view.

        Returns a tuple of action names for _apply.
        """
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
        """Table 6: what to do about a task I have no stake in.

        Simpler than Table 5 because there is nothing of mine to defend:
        the question is only "is the sender better informed than I am?",
        and the answer is driven by execution status and timestamps
        rather than by bid values.

        The ordering of cases encodes an information hierarchy --
        FNSH (finished) is unconditionally believed, EXEC (someone is
        digging) beats NALC (nobody is), and DROP is believed when its
        drop time is fresher than what I hold. Timestamps break the
        remaining ambiguity.

        This table is how information about distant tasks spreads
        through the fleet: robots relay what they were told about tasks
        they will never touch.

        Returns ("update",) or () -- Table 6 has no release/stop actions,
        since I hold nothing to release.
        """
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
        """Carry out the action names a rule table returned (Table 4).

        The six actions, and what each writes:
          Leave           -- (empty tuple) nothing changes.
          stopexecution   -- abandon the task I am executing, but ONLY if
                             can_abandon allows it (Sec. 3.7.4).
          release         -- give up my claim: b = NOCOST, a = NBID,
                             e = NALC, and reset the message counter.
          update          -- adopt the sender's five values wholesale.
          updatebidtime   -- keep my values but refresh tb to now.
          updatetime      -- refresh tb to now AND adopt the sender's td.

        Several can apply at once (e.g. ("release", "update")), and the
        order here matters: release clears my claim before update
        installs the sender's.

        RETURNS True if anything actually changed, which propagates up to
        consensus and then to the allocator's convergence check.
        """
        if not actions:
            return False                                        # Leave
        a_k, b_k, e_k, tb_k, td_k = entry
        n = robot.robot_id
        changed = False

        if "stopexecution" in actions:
            # Sec. 3.7.4: a robot may stop ONLY in the first phase of
            # execution, so that the task state is unchanged and the task
            # stays reallocatable. Table 7 was calling abandon_task()
            # with no such check, so a robot in TO_DUMP with a full
            # hopper would drop the task and keep the soil. Every other
            # drop path in this file checks can_abandon; this one did
            # not. If it cannot stop yet it keeps executing and Table 7
            # will fire again next round, once it is empty.
            if robot.task_id == j and robot.can_abandon:
                robot.abandon_task()
                self.prevExecTask, self.execTask = j, None
                changed = True
            elif robot.task_id != j:
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
        """Keep the E (execution status) vector in step with what the
        robot is physically doing.

        The bridge between BELIEF (the task vectors) and REALITY (what
        robot.py actually did last tick). Three jobs:
          1. Mark every finished task FNSH.
          2. If I am executing something, record it as EXEC and claim it.
          3. If I STOPPED executing without finishing, record a DROP with
             a drop time -- see the long comment inline, this is the case
             that used to poison the whole fleet's beliefs.

        Then hands off to _reconcile for the repairs consensus cannot
        make on its own.

        CALLED BY: CBPAEAllocator.allocate, first thing each tick, before
        any bidding.

        Replaces the old `task.future_remaining -= capacity` hack, which
        was never restored when a task finished or was abandoned and so
        leaked capacity out of the pool.
        """
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

        THE TWO REPAIRS:
          - I believe a task is EXEC but nobody is actually on it
            -> reset it to NALC so it can be bid on again.
          - I believe a task is free but somebody demonstrably IS on it
            -> mark it EXEC and drop my bid on it.

        CALLED BY: syncExecution, at the end.
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
        the task just blocks it for everyone.

        TEST: re-price the current task with leg_cost; a None dump cell
        means no route exists any more. If so, abandon it and record a
        DROP with a drop time, so Tables 5/6 can propagate the news.

        CALLED BY: CBPAEAllocator.allocate, every tick before bidding.
        """
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

        MATH -- the drop test:

            drop  iff  w1*tau_now + w2*E_now  >  execBid * ratio

        where execBid is the price this task was WON at (recorded by
        tryAssign) and the left side is what it would cost re-priced from
        here, right now. So ratio = 2.0 means "drop it if it has become
        more than twice as expensive as I bid".

        CALLED BY: CBPAEAllocator.allocate, every tick. Returns False
        immediately when drop_cost_ratio is None (the default).
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
        """Commit to my bid task -- but only once it is safe to do so.

        THE TWO-PART SAFETY GATE (Sec. 3.7.3), which is what makes it
        impossible for two robots to start the same task:

            now - tb_j  >=  BID_WINDOW          (time has passed)
            msg_count   >=  |current neighbours| (everyone has replied)

        The first gives rivals a fixed window to contest the bid. The
        second requires having actually processed a message from every
        robot within radio range WITHOUT losing the bid in the meantime
        (msg_count resets to 0 whenever a claim is lost -- see placeBid
        and _apply). Together they mean: I waited, everyone who could
        object has spoken, and nobody outbid me.

        Note the quorum is the CURRENT neighbour count, so an isolated
        robot has a quorum of zero and can commit immediately. A floor of
        1 would make it wait forever for a message that can never arrive.

        Before committing it also re-checks ground truth (is the task
        already done or taken?) and handles assign() refusing.

        On success: status EXEC, remember the winning price for
        dropIfTooCostly, and reset the counter.

        CALLED BY: CBPAEAllocator.allocate, as the final stage of a tick.
        """
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
        """One CBPAE round -- exactly one per tick, no inner loop.

        FOUR STAGES, in order:

          1. PER ROBOT: sync execution status against reality, drop the
             current task if it has become unreachable (or too costly),
             compute this round's bid, and broadcast the four-task
             message.
          2. DELIVER the messages.
          3. PER ROBOT: run consensus (Tables 5/6/7) on what arrived.
          4. PER ROBOT: try to commit, subject to the bidding window and
             message quorum.

        CONTRAST WITH CBBA: CBBA runs many rounds inside a single tick
        and stops when nothing changes. CBPAE runs ONE round per tick and
        never checks for convergence, because it does not need to -- the
        bidding window in tryAssign spans several ticks, so the fleet
        gets its rounds of agreement anyway, and meanwhile robots are
        executing rather than waiting. That is the "parallel auction AND
        execution" of the name.

        `now` is the round counter, which doubles as the algorithm's
        clock for every bid/drop timestamp.

        CALLED BY: model.step, once per tick, via model.allocator.
        """
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