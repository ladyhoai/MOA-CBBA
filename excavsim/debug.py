"""Runtime introspection for excavsim: event log, invariant checks and
bid-vs-execution accounting.

Nothing in here changes simulation behaviour. It attaches to a model
INSTANCE by wrapping bound methods, so:
  - none of the existing modules need editing,
  - a fresh model (GUI Reset, batch_run) is simply attached again,
  - a run with no monitor attached is bit-identical to before, because
    the monitor consumes no RNG draws.

Two ways in:

    from excavsim.debug import attach
    mon = attach(model)          # idempotent; returns model._debug
    ...
    mon.events                   # ring buffer of what happened
    mon.checks                   # invariant failures, recomputed each tick
    mon.summary()                # headline numbers for a dashboard

or headless, which is usually faster than clicking through the GUI:

    python -m excavsim.debug --ticks 300 --allocator cbba --seed 42

WHY THESE CHECKS. The three failure modes this simulation actually has
are (a) a task reserved by a robot that is no longer working it, which
removes it from `pending` forever, so the fleet goes quiet with work
left on the board; (b) a robot whose planned leg came back empty
because A* could not route around a hazard/obstacle/robot, which parks
it in TO_TASK for the rest of the run; and (c) allocator disagreement,
where two robots each believe they won the same chunk. None of these
raise, and none of them look different from "the simulation is just
slow" on the map. They each have a named check below.
"""

from __future__ import annotations

import functools
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

from .bidding import Stage, leg_cost
from .CBPAE import DROP, NALC
from .pathfinding import chebyshev

# --- event levels ---------------------------------------------------- #
TRACE, INFO, WARN, ERROR = 0, 1, 2, 3
LEVEL_NAME = {TRACE: "trace", INFO: "info", WARN: "warn", ERROR: "error"}

# how many frozen ticks before we call it a stall
STALL_TICKS = 25

# digits are normalised out before deduping repeated check messages
_NUM_RE = re.compile(r"\d+")


def _bundle_agent(robot):
    """Whichever bundle-based agent is actually in play. A moa-cbba run
    leaves robot.CBBA empty, so reading it unconditionally reported an
    untouched bundle and every CBBA check passed vacuously."""
    for attr in ("MOACBBA", "CBBA"):
        agent = getattr(robot, attr, None)
        if agent is None:
            continue
        if (getattr(agent, "bundle", None) or getattr(agent, "path", None)
                or getattr(agent, "winningAgentList", None)
                or getattr(agent, "bids", None)):
            return agent
    return getattr(robot, "CBBA", None)


@dataclass
class Event:
    tick: int
    level: int
    kind: str
    text: str
    robot: int | None = None
    task: int | None = None

    def __str__(self) -> str:
        who = "" if self.robot is None else f" R{self.robot}"
        return f"[{self.tick:>4}]{who} {self.kind}: {self.text}"


@dataclass
class _Prediction:
    """What the bid promised, recorded at assignment time."""
    tick: int
    energy: float
    tau: float
    e: float
    dist_at_bid: float = 0.0
    sharers: int = 1


@dataclass
class DebugMonitor:
    model: object
    max_events: int = 800

    events: deque = field(default_factory=lambda: deque(maxlen=800))
    checks: list = field(default_factory=list)      # (level, text) this tick
    tick_ms: deque = field(default_factory=lambda: deque(maxlen=200))
    alloc_ms: deque = field(default_factory=lambda: deque(maxlen=200))
    counters: dict = field(default_factory=dict)    # robot_id -> counters
    predictions: dict = field(default_factory=dict)  # (rid, tid) -> _Prediction
    accuracy: deque = field(default_factory=lambda: deque(maxlen=200))
    changed_cells_last: int = 0
    leg_cache_peak: int = 0
    last_progress_tick: int = 0
    _progress_sig: tuple = ()
    _idle_streak: int = 0
    _snapshot: dict = field(default_factory=dict)
    _pending_finish: dict = field(default_factory=dict)
    _check_seen: dict = field(default_factory=dict)
    _abandon_mark: dict = field(default_factory=dict)
    soil_total: float = -1.0        # ground + hoppers + delivered
    _attached_at: int = 0
    _errors: list = field(default_factory=list)     # monitor's own failures
    enabled: bool = True

    # An idle robot next to a pending chunk is normal for a tick or two:
    # the robot finishes mid-tick and the allocator only runs at the top
    # of the next one. Only a streak means the allocator is stuck.
    idle_grace: int = 3
    # window over which repeated abandons count as churn
    churn_window: int = 20
    # ticks before the same check message is worth logging again
    check_cooldown: int = 40

    # ---------------------------------------------------------------- #
    # logging
    # ---------------------------------------------------------------- #
    def log(self, kind: str, text: str, level: int = INFO,
            robot: int | None = None, task: int | None = None) -> None:
        self.events.append(Event(getattr(self.model, "tick", -1), level,
                                 kind, text, robot, task))

    def recent(self, n: int = 30, min_level: int = INFO,
               robot: int | None = None) -> list[Event]:
        """Newest first, which is the order you actually read a log in."""
        out = [e for e in self.events
               if e.level >= min_level
               and (robot is None or e.robot == robot)]
        return out[-n:][::-1]

    def bump(self, robot_id: int, key: str, n: int = 1) -> None:
        c = self.counters.setdefault(robot_id, {})
        c[key] = c.get(key, 0) + n

    def count(self, robot_id: int, key: str) -> int:
        return self.counters.get(robot_id, {}).get(key, 0)

    # ---------------------------------------------------------------- #
    # summary for the dashboard header
    # ---------------------------------------------------------------- #
    def summary(self) -> dict:
        m = self.model
        idle = sum(1 for r in m.robots if r.task_id is None)
        pending = len(m.tasks.pending)
        unfinished = len(m.tasks.unfinished)
        reserved = sum(1 for t in m.tasks.unfinished
                       if t.assigned_to is not None)
        return {
            "tick": m.tick,
            "idle_robots": idle,
            "pending_tasks": pending,
            "reserved_tasks": reserved,
            "unfinished_tasks": unfinished,
            "stalled_for": m.tick - self.last_progress_tick,
            "warnings": sum(1 for lv, _ in self.checks if lv >= WARN),
            "tick_ms": (sum(self.tick_ms) / len(self.tick_ms)) if self.tick_ms else 0.0,
            "alloc_ms": (sum(self.alloc_ms) / len(self.alloc_ms)) if self.alloc_ms else 0.0,
            "alloc_share": (sum(self.alloc_ms) / sum(self.tick_ms) * 100.0
                            if self.tick_ms and sum(self.tick_ms) > 0 else 0.0),
            "changed_cells": self.changed_cells_last,
            "monitor_errors": len(self._errors),
        }

    def allocator_state(self) -> dict:
        """Whatever the current allocator exposes, without assuming which
        one is plugged in."""
        a = getattr(self.model, "allocator", None)
        out = {"allocator": type(a).__name__ if a else "—",
               "name": getattr(a, "name", "—")}
        for key in ("last_round", "converged", "round", "ROUND_CEILING"):
            if hasattr(a, key):
                out[key] = getattr(a, key)
        out["max_bundle_seen"] = getattr(self.model, "_max_bundle_seen", "—")
        out["leg_cache_peak"] = self.leg_cache_peak
        return out

    def accuracy_summary(self) -> dict:
        """bid == uncontended execution is the invariant costs.py is built
        around; this is the measured version of it. Ratios below 1.0 mean
        the bid OVER-charged, above 1.0 that execution cost more than the
        bid promised (contention, re-routes, hazards)."""
        if not self.accuracy:
            return {}
        taus = [a["tau_ratio"] for a in self.accuracy if a["tau_ratio"]]
        es = [a["e_ratio"] for a in self.accuracy if a["e_ratio"]]
        return {
            "n": len(self.accuracy),
            "tau_ratio_mean": sum(taus) / len(taus) if taus else 0.0,
            "tau_ratio_max": max(taus) if taus else 0.0,
            # min matters as much as max: a bid that came in UNDER cost
            # breaks the invariant in the direction nobody checks, and
            # mean+max hid the x0.77 outlier completely
            "tau_ratio_min": min(taus) if taus else 0.0,
            "e_ratio_mean": sum(es) / len(es) if es else 0.0,
            "e_ratio_max": max(es) if es else 0.0,
            "e_ratio_min": min(es) if es else 0.0,
        }

    # ---------------------------------------------------------------- #
    # invariant checks — recomputed every tick, cheap by design
    # ---------------------------------------------------------------- #
    def run_checks(self) -> list[tuple[int, str]]:
        m = self.model
        out: list[tuple[int, str]] = []
        add = out.append

        # 1. BAM consistency, both directions. An orphaned reservation is
        #    invisible on the map and permanently removes the chunk from
        #    tasks.pending, so the fleet idles with work outstanding.
        by_robot = {r.robot_id: r for r in m.robots}
        for t in m.tasks.unfinished:
            for rid in sorted(t.assignees):
                r = by_robot.get(rid)
                if r is None:
                    add((ERROR, f"T{t.task_id} seats unknown robot {rid}"))
                elif r.task_id != t.task_id:
                    add((ERROR, f"T{t.task_id} seats R{rid} but that robot is "
                                f"working "
                                f"{'nothing' if r.task_id is None else f'T{r.task_id}'}"
                                f" — the seat is stranded and the task can "
                                f"never fill it again"))
        for r in m.robots:
            if r.task_id is None:
                continue
            t = m.tasks.get(r.task_id)
            if r.robot_id not in t.assignees:
                add((ERROR, f"R{r.robot_id} is executing T{t.task_id} but is "
                            f"not among its assignees {sorted(t.assignees)}"))

        # 2. one robot per cell (the physics rule robot.py documents)
        seen: dict = {}
        for r in m.robots:
            c = r.cell.coordinate
            if c in seen:
                add((ERROR, f"R{seen[c]} and R{r.robot_id} share cell {c}"))
            seen[c] = r.robot_id

        # 3. robot standing somewhere permanently impassable
        static = m._static_blocked()
        for r in m.robots:
            if r.cell.coordinate in static:
                add((ERROR, f"R{r.robot_id} is inside a bedrock/dump cell "
                            f"{r.cell.coordinate}"))

        # 4. committed but going nowhere: empty path, not yet at the
        #    destination. This is what "the robot just sits there" is.
        for r in m.robots:
            if r.stage in (Stage.TO_TASK, Stage.TO_DUMP) and not r._path:
                dest = r.work_cell if r.stage is Stage.TO_TASK else r.dump_cell
                if dest is not None and r.cell.coordinate != dest:
                    add((WARN, f"R{r.robot_id} is {r.stage.name} with an empty "
                               f"path at {r.cell.coordinate}, target {dest} "
                               f"unreachable"))

        # 5. digging out of range (robot.py re-plans, so this is a hint,
        #    not a fault — but a repeated one means thrash)
        for r in m.robots:
            if r.stage is Stage.DIG and r.task_id is not None:
                t = m.tasks.get(r.task_id)
                if chebyshev(r.cell.coordinate, t.cell) > 1:
                    add((INFO, f"R{r.robot_id} is DIG but {chebyshev(r.cell.coordinate, t.cell)} "
                               f"cells from T{t.task_id} — will re-plan"))

        # 6. flat battery. Nothing in the model stops a robot at 0, so
        #    this silently invalidates any energy-budget conclusion.
        for r in m.robots:
            if r.battery <= 1e-9:
                add((WARN, f"R{r.robot_id} battery is flat "
                           f"({r.energy_used:.1f} spent, cap {r.spec.battery:.0f}) "
                           f"— it keeps working anyway"))

        # 7. two robots aiming at the same dig position
        cells: dict = {}
        for r in m.robots:
            if r.work_cell is None:
                continue
            if r.work_cell in cells:
                # INFO, not WARN. Measured: forcing _reroute to avoid
                # co-workers' claimed cells drove this condition from 334
                # robot-ticks to 0 across 8 seeds and changed total wait
                # ticks by one (314 -> 313) and mean makespan not at all
                # (215.6 -> 217.0, sd 64). Two robots converging on one
                # dig cell resolve it themselves within a tick or two, so
                # reporting it at warning level was crying wolf -- and it
                # fires constantly once several robots share a task.
                add((INFO, f"R{cells[r.work_cell]} and R{r.robot_id} both "
                           f"target work cell {r.work_cell} (benign: they "
                           f"re-route within a tick or two)"))
            cells[r.work_cell] = r.robot_id

        # 8. allocation stall: idle robots and unreserved work, for
        #    longer than the one-tick handover gap explains
        idle = [r.robot_id for r in m.robots if r.task_id is None]
        pending = m.tasks.pending
        if idle and pending and self._idle_streak > self.idle_grace \
                and m.tick > self._attached_at + self.idle_grace:
            # "Pending" means unassigned in the REGISTRY. It does not mean
            # unclaimed: both allocators reserve tasks in local vectors
            # that task.assigned_to knows nothing about. Reporting those
            # as "the allocator is handing out nothing" was simply wrong,
            # and it is wrong in two different ways per allocator:
            #   CBBA  — a bundle claim held by a busy robot blocks every
            #           other robot from bidding. That is the hoarding
            #           defect, and it stays a warning.
            #   CBPAE — reserving your next task while executing IS the
            #           algorithm (Eq. 7). Informational, not a fault.
            held = self._reservation_holders(pending)
            unheld = [t.task_id for t in pending if t.task_id not in held]
            if unheld:
                add((WARN, f"{len(unheld)} chunk(s) {unheld} are unassigned "
                           f"and unreserved after {self._idle_streak} ticks "
                           f"with robots {idle} idle — nothing is bidding"))
            for j, (rid, kind) in held.items():
                where = ("in its bundle" if kind == "cbba"
                         else "as its next bid")
                add((INFO, f"T{j} is unassigned but R{rid} holds it {where} "
                           f"while working T{m.robots[rid].task_id}, so "
                           f"robots {idle} stay idle. Both allocators are "
                           f"time-extended — winning work you will execute "
                           f"LATER is the algorithm (CBBA bundles, CBPAE "
                           f"Eq. 7), not a fault. It is a utilisation cost "
                           f"worth reporting, not a stall to fix."))

        # 9. global stall: nothing moved, nothing dug, nothing changed
        #    stage. Soil alone is a bad proxy — a long haul to the dump
        #    legitimately moves no soil for dozens of ticks.
        stalled = m.tick - self.last_progress_tick
        if stalled >= STALL_TICKS and m.tasks.unfinished:
            add((ERROR, f"the whole fleet has been frozen for {stalled} ticks "
                        f"({len(m.tasks.unfinished)} chunks left) — no motion, "
                        f"no digging, no stage change"))

        # 10. CBBA internals: path must be a permutation of the bundle,
        #     and no two robots may claim the same chunk.
        claims: dict = {}
        for r in m.robots:
            cb = _bundle_agent(r)
            if cb is None or getattr(cb, "winningAgentList", None) is None:
                continue          # cost-convention agent: no y/z to check
            if set(cb.bundle) != set(cb.path):
                add((ERROR, f"R{r.robot_id} CBBA bundle/path mismatch: "
                            f"b={cb.bundle} p={cb.path}"))
            if len(set(cb.path)) != len(cb.path):
                add((ERROR, f"R{r.robot_id} CBBA path has duplicates: {cb.path}"))
            if len(cb.bundle) > r.bundle_limit:
                add((WARN, f"R{r.robot_id} bundle is over L_t "
                           f"({len(cb.bundle)} > {r.bundle_limit})"))
            for tid, winner in cb.winningAgentList.items():
                if winner == r.robot_id:
                    claims.setdefault(tid, []).append(r.robot_id)
        for tid, who in claims.items():
            if len(who) > 1:
                add((ERROR, f"T{tid} is claimed as won by robots {who} "
                            f"— consensus has not converged"))

        # 11a. The CBPAE failure consensus CANNOT repair: every robot
        #      agrees, and every robot is wrong. freeTasks() only offers
        #      chunks whose local status is NALC or DROP, so once every E
        #      vector says EXEC or FNSH for a chunk that nobody is working,
        #      no bid is ever placed, tryAssign is never reached, and no
        #      ASSIGN-FAIL is logged. Silent, permanent, and invisible to
        #      the disagreement count below, because there is nothing left
        #      to disagree about.
        cbpae = [r for r in m.robots if getattr(r, "CBPAE", None)]
        if cbpae:
            for t in m.tasks.unfinished:
                if t.assigned_to is not None:
                    continue
                if all(r.CBPAE._status(t.task_id) not in (NALC, DROP)
                       for r in cbpae):
                    states = sorted({r.CBPAE._status(t.task_id) for r in cbpae})
                    add((ERROR, f"T{t.task_id} is unassigned but every robot's "
                                f"E vector says {'/'.join(states)}, so it is "
                                f"outside freeTasks for the whole fleet — "
                                f"nothing will ever bid on it again"))

        # 11b. CBPAE: how far apart are the local A vectors? Disagreement
        #     is legal mid-auction and a bug if it persists.
        disagree = self._cbpae_disagreement()
        if disagree:
            add((INFO, f"CBPAE: {disagree} task(s) with disagreeing "
                       f"allocation vectors across robots"))

        # 11c. switch churn: abandoning and retaking work every tick is
        #      indistinguishable from healthy re-allocation in the
        #      per-tick view, and lethal over a run -- the robot never
        #      arrives anywhere.
        for r in m.robots:
            n = self.count(r.robot_id, "abandon")
            recent = n - self._abandon_mark.get(r.robot_id, 0)
            if recent >= 5:
                add((ERROR, f"R{r.robot_id} has abandoned a task {recent} "
                            f"times in the last {self.churn_window} ticks — "
                            f"it is cycling, not re-allocating, and is "
                            f"unlikely to finish anything"))

        # 11c-2. MASS CONSERVATION. Soil only ever moves ground ->
        #        hopper -> dump; none of those steps may lose any. A leak
        #        is invisible in every other metric: the task still
        #        finishes, the makespan still reads, and the energy total
        #        is simply short by a haul that never happened. Found
        #        once by accident (0.34-0.99 units per CBPAE run); this
        #        makes it impossible to miss again.
        in_ground = sum(t.remaining for t in m.tasks.all)
        in_hoppers = sum(r.payload for r in m.robots)
        delivered = sum(r.soil_delivered for r in m.robots)
        total = in_ground + in_hoppers + delivered
        if self.soil_total < 0.0:
            self.soil_total = total
        elif abs(total - self.soil_total) > 1e-6:
            add((ERROR, f"soil is not conserved: {total:.4f} now vs "
                        f"{self.soil_total:.4f} at start "
                        f"(ground {in_ground:.2f}, hoppers {in_hoppers:.2f}, "
                        f"dumped {delivered:.2f}) — volume is being "
                        f"created or destroyed"))

        # 11d. Soil in a hopper is soil that has not been delivered. If
        #      every task is stamped while a robot is still carrying, the
        #      run is about to declare victory with material in transit.
        if m.tasks.all_done:
            carrying = [(r.robot_id, r.payload) for r in m.robots
                        if r.payload > 1e-9]
            if carrying:
                add((ERROR, "every task is stamped complete but "
                            + ", ".join(f"R{i} still carries {p:.2f}"
                                        for i, p in carrying)
                            + " — that soil never reached a dump and the "
                              "makespan is short by the final haul"))

        # 12. finished-but-unstamped AND unowned. A chunk that is empty
        #     but still owned is normal: the digger is hauling the last
        #     load and stamps completed_tick when it unloads. With no
        #     owner, nobody will ever stamp it, and makespan skips it.
        for t in m.tasks.all:
            if t.done and t.completed_tick is None and t.assigned_to is None:
                add((ERROR, f"T{t.task_id} is empty, unowned and unstamped "
                            f"— nothing will ever set completed_tick, so "
                            f"makespan silently drops it"))

        if self._errors:
            add((WARN, f"debug monitor itself raised {len(self._errors)} "
                       f"time(s); see mon._errors"))
        return out

    def _reservation_holders(self, pending) -> dict:
        """task_id -> (robot_id, "cbba"|"cbpae") for chunks that are
        unassigned in the registry but claimed in some BUSY robot's local
        vectors. A claim by an idle robot is not a reservation -- it is
        about to become an assignment."""
        held: dict = {}
        for t in pending:
            j = t.task_id
            for r in self.model.robots:
                if r.task_id is None or r.task_id == j:
                    continue
                cb = _bundle_agent(r)
                wal = getattr(cb, "winningAgentList", None) if cb else None
                if wal is not None and wal.get(j) == r.robot_id:
                    held[j] = (r.robot_id, "cbba")
                    break
                cp = getattr(r, "CBPAE", None)
                if cp is not None and cp._winner(j) == r.robot_id:
                    held[j] = (r.robot_id, "cbpae")
                    break
        return held

    def _cbpae_disagreement(self) -> int:
        robots = [r for r in self.model.robots if getattr(r, "CBPAE", None)]
        if len(robots) < 2:
            return 0
        tids: set = set()
        for r in robots:
            tids |= set(r.CBPAE.A)
        n = 0
        for tid in tids:
            views = {r.CBPAE.A.get(tid) for r in robots}
            if len(views) > 1:
                n += 1
        return n

    # ---------------------------------------------------------------- #
    # per-robot dashboard rows
    # ---------------------------------------------------------------- #
    def robot_debug_rows(self) -> list[dict]:
        rows = []
        for r in self.model.robots:
            cb = _bundle_agent(r)
            cp = getattr(r, "CBPAE", None)
            rows.append({
                "robot": r.robot_id,
                "stage": r.stage.name.lower(),
                "task": "—" if r.task_id is None else f"T{r.task_id}",
                "pos": str(r.cell.coordinate),
                "dest": "—" if r._dest is None else str(r._dest),
                "path": len(r._path),
                "stuck": r._stuck,
                "waits": r.wait_ticks,
                "p*": "—" if r.work_cell is None else str(r.work_cell),
                "q*": "—" if r.dump_cell is None else str(r.dump_cell),
                "bundle": "—" if not cb or not cb.bundle else str(cb.bundle),
                "cbba path": "—" if not cb or not cb.path else str(cb.path),
                "wins": (sum(1 for v in cb.winningAgentList.values()
                             if v == r.robot_id)
                         if getattr(cb, "winningAgentList", None) is not None
                         else sum(1 for (_j, i), (c, _s)
                                  in getattr(cb, "bids", {}).items()
                                  if i == r.robot_id and c < float("inf"))),
                "shares": (m_task.sharers
                           if (m_task := (self.model.tasks.get(r.task_id)
                                          if r.task_id is not None else None))
                           else 0),
                "bid/exec": "—" if not cp else
                            f"{'—' if cp.bidTask is None else 'T%d' % cp.bidTask}"
                            f" / "
                            f"{'—' if cp.execTask is None else 'T%d' % cp.execTask}",
                "assigns": self.count(r.robot_id, "assign"),
                "fails": self.count(r.robot_id, "assign_fail"),
                "reroutes": self.count(r.robot_id, "reroute"),
                "noroute": self.count(r.robot_id, "unreachable"),
                "drops": r.tasks_dropped,
                "done": r.tasks_completed,
            })
        return rows

    def task_debug_rows(self) -> list[dict]:
        rows = []
        for t in self.model.tasks.all:
            holders = []
            for r in self.model.robots:
                cb = _bundle_agent(r)
                wal = getattr(cb, "winningAgentList", None) if cb else None
                if wal is not None and wal.get(t.task_id) == r.robot_id:
                    holders.append(f"R{r.robot_id}")
            rows.append({
                "task": f"T{t.task_id}",
                "cell": str(t.cell),
                "left": round(t.remaining, 3),
                "seats": ",".join(f"R{i}" for i in sorted(t.assignees)) or "—",
                "claims": ",".join(holders) or "—",
                "done@": "—" if t.completed_tick is None else t.completed_tick,
            })
        return rows

    # ---------------------------------------------------------------- #
    # hooks (called from the wrappers)
    # ---------------------------------------------------------------- #
    def _on_assign(self, robot, task_id: int) -> None:
        m = self.model
        task = m.tasks.get(task_id)
        tau = e = 0.0
        try:
            tau, e, _q = leg_cost(m, robot, task)
        except Exception:                       # pragma: no cover
            self._note_error()
        # leg_cost prices the whole remaining volume as a solo job. When
        # k robots share the task each moves roughly V/k, so the bid is
        # re-priced on that share -- otherwise the ratio measures
        # solo-pricing against shared execution rather than cost model
        # against simulation, which is what it is supposed to test.
        sharers = max(1, len(task.assignees))
        if sharers > 1:
            try:
                tau, e, _q2 = leg_cost(m, robot, task,
                                       volume=task.remaining / sharers)
            except Exception:                   # pragma: no cover
                self._note_error()
        self.predictions[(robot.robot_id, task_id)] = _Prediction(
            tick=m.tick, energy=robot.energy_used, tau=tau, e=e,
            dist_at_bid=robot.distance_travelled, sharers=sharers)
        self.bump(robot.robot_id, "assign")
        share_txt = "" if sharers == 1 else f" | shared {sharers}-way"
        self.log("ASSIGN",
                 f"takes T{task_id} at {task.cell} (V={task.remaining:.2f}) "
                 f"p*={robot.work_cell} q*={robot.dump_cell} "
                 f"| bid tau={tau:.1f} E={e:.2f} "
                 f"| path={len(robot._path)} steps{share_txt}",
                 INFO, robot.robot_id, task_id)

    def _on_finish(self, robot, task_id: int | None) -> None:
        if task_id is None:
            return
        m = self.model
        pred = self.predictions.pop((robot.robot_id, task_id), None)
        if pred is None:
            self.log("DONE", f"finished T{task_id}", INFO, robot.robot_id, task_id)
            return
        d_tick = m.tick - pred.tick
        d_e = robot.energy_used - pred.energy
        tau_r = (d_tick / pred.tau) if pred.tau > 0 else None
        e_r = (d_e / pred.e) if pred.e > 0 else None
        self.accuracy.append({"robot": robot.robot_id, "task": task_id,
                              "tau_pred": pred.tau, "tau_real": d_tick,
                              "e_pred": pred.e, "e_real": d_e,
                              "tau_ratio": tau_r, "e_ratio": e_r})
        self.log("DONE",
                 f"finished T{task_id} in {d_tick} ticks / {d_e:.2f} E "
                 f"(bid said {pred.tau:.1f} / {pred.e:.2f}"
                 + (f", x{tau_r:.2f} time" if tau_r else "")
                 + (f", x{e_r:.2f} energy)" if e_r else ")"),
                 INFO, robot.robot_id, task_id)
        # A big overrun on an uncontended run means the cost model and
        # the simulation have drifted apart, which invalidates every bid.
        if tau_r and tau_r > 1.5:
            self.log("COST-DRIFT",
                     f"T{task_id} took {tau_r:.2f}x the bid time "
                     f"({robot.wait_ticks} wait ticks total, "
                     f"{self.count(robot.robot_id, 'reroute')} re-routes)",
                     WARN, robot.robot_id, task_id)

    # ---------------------------------------------------------------- #
    def _note_error(self) -> None:
        self._errors.append(traceback.format_exc())

    def _snapshot_robots(self) -> dict:
        return {r.robot_id: (r.stage, r.task_id, r.cell.coordinate,
                             round(r.payload, 6))
                for r in self.model.robots}

    def _diff_robots(self, before: dict) -> None:
        for r in self.model.robots:
            old = before.get(r.robot_id)
            if old is None:
                continue
            o_stage, o_task, o_cell, o_pay = old
            if r.stage is not o_stage:
                self.log("STAGE",
                         f"{o_stage.name} -> {r.stage.name} at "
                         f"{r.cell.coordinate}"
                         + ("" if r.task_id is None else f" (T{r.task_id})"),
                         TRACE, r.robot_id, r.task_id)
            if o_cell == r.cell.coordinate and r.stage in (Stage.TO_TASK,
                                                           Stage.TO_DUMP):
                self.bump(r.robot_id, "no_move")

    def _progress_signature(self) -> tuple:
        """Everything that counts as the simulation doing something.
        Soil alone is not enough: a robot on a 30-step haul to the dump
        moves no soil and is perfectly healthy."""
        m = self.model
        return (round(sum(t.volume - t.remaining for t in m.tasks.all), 6),
                round(sum(r.distance_travelled for r in m.robots), 3),
                tuple((r.stage, r.task_id) for r in m.robots),
                sum(1 for t in m.tasks.all if t.completed_tick is not None))

    def on_tick_end(self) -> None:
        m = self.model
        sig = self._progress_signature()
        if sig != self._progress_sig:
            self._progress_sig = sig
            self.last_progress_tick = m.tick

        if m.tick % self.churn_window == 0:
            self._abandon_mark = {r.robot_id: self.count(r.robot_id, "abandon")
                                  for r in m.robots}

        if not (any(r.task_id is None for r in m.robots) and m.tasks.pending):
            self._idle_streak = 0
        elif m.tick > self._attached_at:
            # attach() runs one evaluation before any step has happened;
            # counting it made the streak read one tick high and fire the
            # warning before the allocator had had a fair chance
            self._idle_streak += 1

        try:
            new = self.run_checks()
        except Exception:                       # pragma: no cover
            self._note_error()
            new = [(WARN, "check pass raised; see mon._errors")]
        # Log a check only when it is genuinely new. Deduping on the raw
        # text is not enough: "idle for 21 ticks" and "...22 ticks" are
        # different strings and the same problem, so numbers are
        # normalised out and each distinct message gets a cooldown.
        for level, text in new:
            if level < WARN:
                continue
            key = _NUM_RE.sub("#", text)
            if m.tick - self._check_seen.get(key, -10 ** 9) >= self.check_cooldown:
                self.log("CHECK", text, level)
            self._check_seen[key] = m.tick
        self.checks = new


# ---------------------------------------------------------------------- #
# instrumentation
# ---------------------------------------------------------------------- #
def _wrap(obj, name: str, before=None, after=None) -> bool:
    """Wrap a BOUND method on an instance. Returns False if already
    wrapped, so attach() is idempotent."""
    orig = getattr(obj, name, None)
    if orig is None or getattr(orig, "_excavsim_debug", False):
        return False

    @functools.wraps(orig)
    def wrapper(*a, **kw):
        if before is not None:
            try:
                before(*a, **kw)
            except Exception:
                pass
        out = orig(*a, **kw)
        if after is not None:
            try:
                after(out, *a, **kw)
            except Exception:
                pass
        return out

    wrapper._excavsim_debug = True
    setattr(obj, name, wrapper)
    return True


def attach(model, max_events: int = 800) -> DebugMonitor:
    """Attach (once) and return the monitor. Safe to call every frame."""
    existing = getattr(model, "_debug", None)
    if existing is not None:
        return existing

    mon = DebugMonitor(model=model, max_events=max_events)
    mon.events = deque(maxlen=max_events)
    mon._progress_sig = ()
    mon.last_progress_tick = model.tick
    mon._attached_at = model.tick
    model._debug = mon

    # --- model.step: timing, stage diffing, checks ------------------- #
    def before_step(*a, **kw):
        mon._snapshot = mon._snapshot_robots()
        mon._t0 = time.perf_counter()

    def after_step(out, *a, **kw):
        mon.tick_ms.append((time.perf_counter() - mon._t0) * 1000.0)
        mon._diff_robots(mon._snapshot)
        mon.on_tick_end()

    _wrap(model, "step", before=before_step, after=after_step)

    # --- allocator: how long, and did it converge -------------------- #
    alloc = getattr(model, "allocator", None)
    if alloc is not None:
        def before_alloc(*a, **kw):
            mon._t_alloc = time.perf_counter()

        def after_alloc(out, *a, **kw):
            mon.alloc_ms.append((time.perf_counter() - mon._t_alloc) * 1000.0)
            # read it here: model.step() clears _leg_cache at the top of
            # the next tick, so by render time it is always empty
            mon.leg_cache_peak = max(mon.leg_cache_peak,
                                     len(getattr(model, "_leg_cache", {}) or {}))
            if hasattr(alloc, "converged") and not alloc.converged:
                mon.log("AUCTION",
                        f"did NOT converge (stopped at round "
                        f"{getattr(alloc, 'last_round', '?')})", WARN)

        _wrap(alloc, "allocate", before=before_alloc, after=after_alloc)

    # --- dynamics: capture changed_cells before step() clears it ----- #
    dyn = getattr(model, "dynamics", None)
    if dyn is not None:
        prev = {"weather": dyn.weather, "hz": len(dyn.hazards),
                "ob": len(dyn.obstacles)}

        def after_dyn(out, *a, **kw):
            mon.changed_cells_last = len(model.changed_cells)
            if dyn.weather != prev["weather"]:
                mon.log("WEATHER", f"{prev['weather']} -> {dyn.weather} "
                                   f"(traction x{dyn.traction_scale():.2f}, "
                                   f"sensor x{dyn.sensor_scale():.2f})", INFO)
                prev["weather"] = dyn.weather
            # Zone counts on their own are wallpaper at any interesting
            # hazard_rate. What matters is whether the new geometry
            # landed on somebody's route or buried a target.
            if len(dyn.hazards) != prev["hz"]:
                mon.log("HAZARD", f"{prev['hz']} -> {len(dyn.hazards)} zone(s), "
                                  f"{sum(len(h.cells) for h in dyn.hazards)} cells",
                        TRACE)
                prev["hz"] = len(dyn.hazards)
            if len(dyn.obstacles) != prev["ob"]:
                mon.log("OBSTACLE", f"{prev['ob']} -> {len(dyn.obstacles)}", TRACE)
                prev["ob"] = len(dyn.obstacles)

            blocked = dyn.blocked()
            if not blocked:
                return
            for r in model.robots:
                hits = [c for c in r._path if c in blocked]
                # the same zone sitting on the same route is one event,
                # not one per tick until it expires
                seen_at, seen_cells = prev.get(f"hit{r.robot_id}", (-10 ** 9, None))
                fresh = (frozenset(hits) != seen_cells
                         or model.tick - seen_at >= 20)
                prev[f"hit{r.robot_id}"] = (model.tick, frozenset(hits))
                if hits and fresh:
                    mon.log("ROUTE-HIT",
                            f"{len(hits)} cell(s) of the remaining "
                            f"{len(r._path)}-step route to {r._dest} are now "
                            f"hazard/obstacle (first at {hits[0]}) — expect a "
                            f"wait then a re-route", WARN, r.robot_id, r.task_id)
            for t in model.tasks.unfinished:
                if t.assigned_to is not None and t.cell in blocked:
                    mon.log("BURIED",
                            f"T{t.task_id} at {t.cell} is inside a hazard while "
                            f"R{t.assigned_to} is working it", WARN,
                            t.assigned_to, t.task_id)

        _wrap(dyn, "step", after=after_dyn)

    # --- robots ------------------------------------------------------ #
    for r in model.robots:
        _instrument_robot(mon, r)

    mon.log("ATTACH", f"monitor attached to model {id(model)} "
                      f"({len(model.robots)} robots, "
                      f"{len(model.tasks.all)} chunks, "
                      f"allocator={getattr(model.allocator, 'name', '?')})", INFO)
    mon.on_tick_end()
    return mon


def _instrument_robot(mon: DebugMonitor, r) -> None:
    rid = r.robot_id

    def after_assign(ok, task_id, *a, **kw):
        if ok:
            mon._on_assign(r, task_id)
        else:
            mon.bump(rid, "assign_fail")
            mon.log("ASSIGN-FAIL",
                    f"cannot take T{task_id}: no reachable dig cell or no "
                    f"reachable dump from it", WARN, rid, task_id)

    def before_finish(*a, **kw):
        mon._pending_finish[rid] = r.task_id

    def after_finish(out, *a, **kw):
        mon._on_finish(r, mon._pending_finish.pop(rid, None))

    def before_abandon(*a, **kw):
        mon._pending_finish[rid] = r.task_id

    def after_abandon(out, *a, **kw):
        tid = mon._pending_finish.pop(rid, None)
        if tid is not None:
            mon.predictions.pop((rid, tid), None)
            mon.bump(rid, "abandon")
            mon.log("ABANDON", f"dropped T{tid} back into the pool "
                               f"(now reallocatable)", WARN, rid, tid)

    def before_reroute(occupied, *a, **kw):
        # _reroute calls _plan_leg, which resets _stuck to 0, so the
        # interesting number has to be read before the call
        mon._pending_finish[f"stuck{rid}"] = (r._stuck, r.stage.name, r._dest)

    def after_reroute(out, occupied, *a, **kw):
        stuck, stage, old_dest = mon._pending_finish.pop(
            f"stuck{rid}", (r._stuck, r.stage.name, r._dest))
        mon.bump(rid, "reroute")
        mon.log("REROUTE",
                f"blocked {stuck} ticks in {stage}, re-planned {old_dest} -> "
                f"{r._dest} around {len(occupied)} occupied cell(s); "
                f"path now {len(r._path)} steps", INFO, rid, r.task_id)

    def after_plan(out, dest, *a, **kw):
        if not r._path and dest is not None and r.cell.coordinate != dest:
            mon.bump(rid, "unreachable")
            mon.log("NO-ROUTE",
                    f"A* found no path {r.cell.coordinate} -> {dest} "
                    f"in {r.stage.name}; the robot will sit still until "
                    f"something moves", WARN, rid, r.task_id)

    def after_go_dump(out, *a, **kw):
        mon.log("DUMP", f"hopper {r.payload:.2f}/{r.spec.capacity:.1f} -> "
                        f"heading to {r.dump_cell}", TRACE, rid, r.task_id)

    _wrap(r, "assign", after=after_assign)
    _wrap(r, "_finish_task", before=before_finish, after=after_finish)
    _wrap(r, "abandon_task", before=before_abandon, after=after_abandon)
    _wrap(r, "_reroute", before=before_reroute, after=after_reroute)
    _wrap(r, "_plan_leg", after=after_plan)
    _wrap(r, "_go_dump", after=after_go_dump)


def monitor(model) -> DebugMonitor | None:
    return getattr(model, "_debug", None)


# ---------------------------------------------------------------------- #
# text rendering, shared by the CLI and the GUI panel
# ---------------------------------------------------------------------- #
def snapshot_text(model, n_events: int = 20, min_level: int = INFO) -> str:
    mon = attach(model)
    s = mon.summary()
    a = mon.allocator_state()
    lines = [
        f"tick {s['tick']}  |  idle {s['idle_robots']}/{len(model.robots)}  |  "
        f"pending {s['pending_tasks']}  reserved {s['reserved_tasks']}  "
        f"left {s['unfinished_tasks']}  |  stalled {s['stalled_for']}t  |  "
        f"{s['tick_ms']:.1f} ms/tick ({s['alloc_share']:.0f}% allocator)",
        "allocator: " + "  ".join(f"{k}={v}" for k, v in a.items()),
    ]
    acc = mon.accuracy_summary()
    if acc:
        lines.append(f"bid vs actual over {acc['n']} finished chunks: "
                     f"time x{acc['tau_ratio_min']:.2f}/"
                     f"x{acc['tau_ratio_mean']:.2f}/"
                     f"x{acc['tau_ratio_max']:.2f} (min/mean/max), "
                     f"energy x{acc['e_ratio_min']:.2f}/"
                     f"x{acc['e_ratio_mean']:.2f}/"
                     f"x{acc['e_ratio_max']:.2f}")
        if min(acc['tau_ratio_min'], acc['e_ratio_min']) < 0.95:
            lines.append("  NOTE: a bid came in UNDER its execution cost. "
                         "costs.py documents the invariant one-directionally "
                         "(realized >= bid); re-planning after a reroute can "
                         "break it the other way.")
    if mon.checks:
        lines.append("checks:")
        lines += [f"  [{LEVEL_NAME[lv]}] {txt}" for lv, txt in mon.checks]
    else:
        lines.append("checks: all clear")
    if n_events:
        lines.append("events (newest first):")
        lines += [f"  {e}" for e in mon.recent(n_events, min_level)]
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# headless driver
# ---------------------------------------------------------------------- #
def main(argv=None) -> int:            # pragma: no cover - CLI
    import argparse

    from .model import ExcavationModel

    p = argparse.ArgumentParser(description="run excavsim with the monitor on")
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--allocator", default="cbba")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--robots", type=int, default=4)
    p.add_argument("--tasks", type=int, default=8)
    p.add_argument("--hazard-rate", type=float, default=0.0)
    p.add_argument("--obstacle-rate", type=float, default=0.0)
    p.add_argument("--every", type=int, default=10,
                   help="print a summary line every N ticks")
    p.add_argument("--trace", action="store_true",
                   help="include stage transitions in the event dump")
    args = p.parse_args(argv)

    model = ExcavationModel(seed=args.seed, allocator=args.allocator,
                            n_robots=args.robots, n_tasks=args.tasks,
                            hazard_rate=args.hazard_rate,
                            obstacle_rate=args.obstacle_rate)
    mon = attach(model)
    for _ in range(args.ticks):
        if model.tasks.all_done:
            break
        model.step()
        if args.every and model.tick % args.every == 0:
            s = mon.summary()
            flags = "".join(sorted({LEVEL_NAME[lv][0].upper()
                                    for lv, _ in mon.checks if lv >= WARN}))
            print(f"t={model.tick:>4} done={sum(t.done for t in model.tasks.all):>3}"
                  f"/{len(model.tasks.all):<3} idle={s['idle_robots']} "
                  f"pending={s['pending_tasks']} stall={s['stalled_for']:>3} "
                  f"{s['tick_ms']:.1f}ms {flags}")
    print()
    print(snapshot_text(model, n_events=40,
                        min_level=TRACE if args.trace else INFO))
    return 0


if __name__ == "__main__":             # pragma: no cover
    raise SystemExit(main())