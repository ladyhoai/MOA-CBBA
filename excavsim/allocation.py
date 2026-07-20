"""Pluggable task allocators.

Every allocator implements `allocate(model)`: inspect idle robots and
pending tasks, then call `robot.assign(task_id)` for each new pairing.
This is the seam where CBBA, CBPAE and MOA-CBBA slot in — the model and
metrics code never change when you swap algorithms, which is what makes
the three-way comparison clean in batch runs.

GreedyAllocator is a working baseline (your Fig. 4 "Very Greedy",
upgraded to A* distances and the full Eq. 4/5 bid). The CBBA-family
classes are scaffolds marking where the auction/consensus phases go.
"""

from __future__ import annotations

from .costs import energy_ij, tau_ij
from .pathfinding import nearest_work_cell
from .terrain import HARDNESS, Terrain

import logging
import os
from pathlib import Path

INF = float("inf")
EPS = 1e-9
NOBID = float("-inf") 

############### Below are helper functions for debugging and formatting ###############
CBBA_VERBOSE = 1
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
    """The invariant CBBA relies on: path is a permutation of bundle, no repeats."""
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
    """Each idle robot takes the cheapest pending task, in shuffled order."""

    name = "greedy"

    def allocate(self, model) -> None:
        idle = [r for r in model.robots if r.task_id is None]
        model.random.shuffle(idle)
        for robot in idle:
            pending = model.tasks.pending
            if not pending:
                return
            bids = [(bid_value(model, robot, t, model.w1, model.w2), t)
                    for t in pending]
            bids = [(bid, dumpCoord, t) for (bid, dumpCoord), t in bids
                    if bid != float("inf")]
            for bid, dumpCoord, t in sorted(bids, key=lambda bdt: bdt[0]):
                if robot.assign(t.task_id):
                    break

def leg_cost(model, robot, task, startPos = None) -> tuple[float, ...]:
    """Cost of robot i taking task j: tau_ij and E_ij."""
    coord = task.cell
    terrain = Terrain(int(model.grid.terrain.data[coord]))
    h = HARDNESS[terrain]
    startPosLeg = robot.cell.coordinate if startPos is None else startPos
    dig = nearest_work_cell(startPosLeg, [coord],
                            model.grid.width, model.grid.height,
                            model.blocked_cells())
    if dig is None:
        return float("inf"), float("inf"), None # type: ignore
    p_star, d_task = dig
    dump = model.dump_work_cell(p_star)
    if dump is None:
        return float("inf"), float("inf"), None # type: ignore
    dumpCoord, d_dump = dump
    t = tau_ij(robot.spec, task.remaining, h, d_task, d_dump)
    e = energy_ij(robot.spec, task.remaining, h, d_task, d_dump)
    return t, e, dumpCoord

def bid_value(model, robot, task, w1: float = 1.0, w2: float = 1.0, startPos = None) -> tuple[float, ...]:
    """Marginal cost of robot i taking task j: w1*tau_ij + w2*E_ij.
    (Cost convention, used by the greedy baseline only.)"""
    t, e, dumpCoord = leg_cost(model, robot, task, startPos)
    if dumpCoord is None:
        return INF, None # type: ignore
    return w1 * t + w2 * e, dumpCoord
    
# The path score should be time-discounted, meaning later task in the path should have less weight than earlier task. 
# The path score is the sum of discounted reward minus energy cost for each task in the path.
def pathScoreCBBA(model, robot, path, lam=0.99, task_reward=10.0, energy_weight=0.1) -> float:
    t_clock, score = 0.0, 0.0
    # When the robot finish dumping, its start position for the next task will be the grid coordinate of the cell 
    # it was at to dump the load. 
    dumpCoord = None
    for taskID in path:
        task = model.tasks.get(taskID)
        tau, e, dumpCoord = leg_cost(model, robot, task, dumpCoord)
        if dumpCoord is None:
            return NOBID
        t_clock += tau
        # Below is a decaying function
        score += (lam ** t_clock) * task_reward - energy_weight * e
    return score

def _better_bid(bid_a, agent_a, bid_b, agent_b) -> bool:
    """True if (bid_a by agent_a) beats (bid_b by agent_b).
 
    Reward maximisation (Choi et al.'s original convention): the HIGHER
    bid wins. Exact ties are broken by the lower robot id — the paper
    (Sec. III-B) requires a systematic tie-break, otherwise two robots
    that compute identical bids can both keep believing they won and
    consensus never converges.
    """
    if agent_a is None:
        return False
    if agent_b is None:
        return bid_a > NOBID
    if bid_a > bid_b + EPS:
        return True
    if bid_b > bid_a + EPS:
        return False
    # The bid are equal if it gets to this line, so we choose a random winner
    return agent_a < agent_b


class CBBAAgent:
    """Consensus-Based Bundle Algorithm (Choi et al., 2009).

    TODO: bundle construction via marginal-gain bidding using
    `bid_value`, then the consensus/conflict-resolution table.
    Validate against the MIT ACL reference implementation.

    Consensus should exchange (y, z, s) via the comms layer:
        robot.send(payload)                          # broadcast
        model.comms.flush_and_deliver(model.tick)    # synchronous round
        msgs = robot.receive_all()
    """

    def __init__(self):
        self.path = []
        self.task_list = []
        self.bundle = []     # b_i
        self.winningAgentList = {}  # z_i
        self.winningBidList = {}    # y_i
        self.timeStamp: dict[int, int] = {}  # s_i

    def _bid(self, task_id) -> float:
        return self.winningBidList.get(task_id, NOBID)
    
    def _winner(self, task_id) -> int:
        return self.winningAgentList.get(task_id, None) # type: ignore
    
    def _stamp(self, agent_id) -> int:
        return self.timeStamp.get(agent_id, -1)
    
    # Sending the task to all neighbouring robots (Phase 6 will enforce the communication range)
    def broadcast(self, robot, now: int) -> None:
        self.timeStamp[robot.robot_id] = now
        robot.send({
            "sender": robot.robot_id,
            "y": dict(self.winningBidList),
            "z": dict(self.winningAgentList),
            "s": dict(self.timeStamp),
        })

    def _CBBADecisionTable(self, i, k, j, y_k, z_k, s_k) -> str:
        """Table I: receiver i's action on task j after hearing from k.
 
        Returns "update", "reset" or "leave" (the default).
        """
        zk = z_k.get(j)          # who the sender thinks won j
        zi = self._winner(j)     # who I think won j
        ykj = y_k.get(j, NOBID)  # the bid of agent k on task j
        yij = self._bid(j)       # the bid of ourselves (agent i) on task j
 
        def newer(m) -> bool:
            """s_km > s_im: the sender's information about m is fresher."""
            return s_k.get(m, -1) > self._stamp(m)
 
        def better() -> bool:
            """y_kj beats y_ij (lower cost, tie-break on robot id)."""
            return _better_bid(ykj, zk, yij, zi)
 
        # ---- sender (k) thinks z_kj = k (it won the task itself) ---------- #
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
 
        # ---- sender thinks nobody won it ----------------------------- #
        if zk is None:
            if zi == i:
                return "leave"
            if zi == k:
                return "update"
            if zi is None:
                return "leave"
            return "update" if newer(zi) else "leave"                 # zi = m
 
        # ---- sender thinks z_kj = m, some third agent ---------------- #
        m = zk
        if zi == i:
            return "update" if (newer(m) and better()) else "leave"
        if zi == k:
            return "update" if newer(m) else "reset"
        if zi is None:
            return "update" if newer(m) else "leave"
        if zi == m:
            return "update" if newer(m) else "leave"
 
        # ---- zi = n, a fourth agent ---------------------------------- #
        n = zi
        if newer(m) and newer(n):
            return "update"
        if newer(m) and better():
            return "update"
        if newer(n) and self._stamp(m) > s_k.get(m, -1):
            return "reset"
        return "leave"

    # Equation 6 in the paper: if a task is outbid, we have to remove all the tasks that were added after it because
    # the score for that bundle will turn incorrect
    def _releaseOutbid(self, i) -> bool:

        n_bar = None
        for n, task_id in enumerate(self.bundle):
            if self._winner(task_id) != i:
                n_bar = n
                break
        if n_bar is None:
            return False
 
        for task_id in self.bundle[n_bar + 1:]:

            # Resetting the bid and winning agent for that task
            if self._winner(task_id) == i:
                self.winningBidList[task_id] = NOBID
                self.winningAgentList[task_id] = None
        
        # Removing from the path dictionary
        for task_id in self.bundle[n_bar:]:
            if task_id in self.path:
                self.path.remove(task_id)

        # Removing from the bundle
        del self.bundle[n_bar:]
        return True

    # winningBidList is a local dictionary that each robot has. This will be updated in the conflict resolution phase
    def createBundle(self, model, robot, winningAgentList, winningBidList, bundle):
        # NOTE: these are copies now, not aliases. See notes at the bottom of the file.
        # Also, I don't understand why do we have to do this instead of directly operating on the class members??
        currentWinningBidList = dict(winningBidList)
        currentWinningAgentList = dict(winningAgentList)
        self.bundle = list(bundle)
        currentPath = list(self.path)
    
        tag = f"[R{robot.robot_id}]"
        known_ids = [t.task_id for t in self.task_list]
    
        _dbg(1, "=" * 78)
        _dbg(1, f"{tag} createBundle ENTRY")
        _dbg(1, f"{tag}   capacity        : {robot.capacity}")
        _dbg(1, f"{tag}   bundle in       : {self.bundle}")
        _dbg(1, f"{tag}   path   in       : {currentPath}")
        _dbg(1, f"{tag}   y (winning bids): {_fmt_map(currentWinningBidList)}")
        _dbg(1, f"{tag}   z (winners)     : {_fmt_map(currentWinningAgentList)}")
        _dbg(1, f"{tag}   task_list ids   : {known_ids}")
    
        _check_consistency(tag, self.bundle, currentPath, "on entry")
    
        # Are the bid/winner tables actually initialised for every task?
        missing_y = [tid for tid in known_ids if tid not in currentWinningBidList]
        if missing_y:
            _dbg(1, f"{tag}   note: no y entry for {missing_y} -> treated as -inf (nobid)")
    
        iteration = 0
        while len(self.bundle) < robot.capacity:
            allTaskIDs = set(self.bundle)

            # Create a list of tasks that are not added to the bundle of the current robot yet
            taskNotInBundles = [t for t in self.task_list if t.task_id not in allTaskIDs]
    
            # The score of the currently constructed path.
            currentPathScore = pathScoreCBBA(model, robot, currentPath)
    
            _dbg(1, "-" * 78)
            _dbg(1, f"{tag} iter {iteration} | bundle={self.bundle} path={currentPath} "
                    f"| S(path)={_fmt_num(currentPathScore)}")
            _dbg(1, f"{tag} candidates: {[t.task_id for t in taskNotInBundles]}")
    
            if not taskNotInBundles:
                _dbg(1, f"{tag} STOP: no tasks left outside the bundle")
                break
    
            # Preparing the variables to deal with the maximising problem
            currentImprovement = NOBID
            Ji = None
            JiPathLocation = None
    
            _dbg(1, f"{tag}   {'task':>6} {'pos':>4} {'S(new)':>8} {'c_ij':>8} {'y_ij':>8}  verdict")
    
            for taskNotInBundle in taskNotInBundles:
                bestBidTask = NOBID
                bestTaskLocation = None

                # Try inserting the task at every possible positions in the current path
                for trialPosition in range(len(currentPath) + 1):
                    trialPath = (currentPath[:trialPosition]
                                + [taskNotInBundle.task_id]
                                + currentPath[trialPosition:])
                    # Compute the path score for the new path with the inserted task
                    trialBid = pathScoreCBBA(model, robot, trialPath)
                    _dbg(2, f"{tag}      try {taskNotInBundle.task_id} @ pos {trialPosition}: "
                            f"path={trialPath} S={_fmt_num(trialBid)}")
                    
                    # Find the best location to insert that task
                    if trialBid > bestBidTask:
                        bestBidTask = trialBid
                        bestTaskLocation = trialPosition
    
                # Check if the task's bid can win the bid for that same task coming from the sender
                improvement = bestBidTask - currentPathScore
                y_ij = currentWinningBidList.get(taskNotInBundle.task_id, NOBID)
    
                # --- diagnostics on the bid itself ---
                notes = []
                if bestTaskLocation is None:
                    notes.append("NO VALID POSITION (all insertions infeasible)")
                if improvement < 0:
                    notes.append("negative gain (energy penalty > discounted reward)")
                if improvement != improvement:  # NaN
                    notes.append("NaN marginal gain")
    
                beats_y = improvement > y_ij    # line 8: must beat the incumbent
                if not beats_y:
                    verdict = "loses to incumbent bid"
                elif improvement > currentImprovement:
                    verdict = "NEW BEST"
                else:
                    verdict = "beats y, not best"

                _dbg(1, f"{tag}   {taskNotInBundle.task_id:>6} {str(bestTaskLocation):>4} "
                        f"{_fmt_num(bestBidTask)} {_fmt_num(improvement)} {_fmt_num(y_ij)}  {verdict}"
                        + ("  <<< " + "; ".join(notes) if notes else ""))
    
                # If the task is winnable and offers the best improvement in all tasks tried, we update Ji to hold that task, which will be added 
                # to the bundle
                if beats_y and improvement > currentImprovement:
                    currentImprovement = improvement
                    Ji = taskNotInBundle.task_id
                    JiPathLocation = bestTaskLocation
    
            if Ji is None:
                _dbg(1, f"{tag} STOP: no task beat its incumbent bid "
                        f"(bundle size {len(self.bundle)}/{robot.capacity})")
                break
    
            # Adding the task to the bundle, path, bid list and winning agent list
            self.bundle.append(Ji)
            currentPath.insert(JiPathLocation, Ji) # type: ignore
            currentWinningBidList[Ji] = currentImprovement
            prev_winner = currentWinningAgentList.get(Ji)
            currentWinningAgentList[Ji] = robot.robot_id
    
            _dbg(1, f"{tag} ADD task {Ji} at path pos {JiPathLocation} "
                    f"| c={_fmt_num(currentImprovement)} | took it from z={prev_winner}")
            _dbg(1, f"{tag}     bundle -> {self.bundle}")
            _dbg(1, f"{tag}     path   -> {currentPath}")
            _check_consistency(tag, self.bundle, currentPath, f"after adding {Ji}")
    
            iteration += 1
            if iteration > len(self.task_list) + robot.capacity + 1:
                _dbg(1, f"{tag} !! ABORT: while-loop is not terminating")
                break
    
        if len(self.bundle) >= robot.capacity:
            _dbg(1, f"{tag} STOP: capacity reached ({len(self.bundle)}/{robot.capacity})")

        # Update the internal class variable to reflect the new changes
        self.winningAgentList = currentWinningAgentList
        self.winningBidList = currentWinningBidList
        self.path = currentPath
    
        _dbg(1, f"{tag} createBundle EXIT")
        _dbg(1, f"{tag}   bundle out      : {self.bundle}")
        _dbg(1, f"{tag}   path   out      : {self.path}")
        _dbg(1, f"{tag}   y (winning bids): {_fmt_map(currentWinningBidList)}")
        _dbg(1, f"{tag}   z (winners)     : {_fmt_map(currentWinningAgentList)}")
        # bundle order == order added; path order == execution order. They should differ
        # in general, but must contain the same ids.
        _check_consistency(tag, self.bundle, self.path, "on exit")
        _dbg(1, "=" * 78)
    
        return currentWinningAgentList, currentWinningBidList, self.bundle

    # Phase 2: Conflict Resolution
    def resolveConflicts(self, robot, receivedMessages) -> bool:
        # Deciding the outcome for received tasks using table 1
        # Return true if the fleet hasn't converged and another iteration is needed
        i = robot.robot_id
        changed = False

        for msg in receivedMessages:
            payload = msg.payload
            # Getting the ID of the sender who send out that message
            k = payload.get("sender")
            if k is None or k == i:
                continue

            # This is the list from the other senders
            otherWinningBidList = payload.get("y", {})
            otherWinningAgentList = payload.get("z", {})
            otherTimestamps = payload.get("s", {})

            # Find conflicting tasks between both agents
            taskIDInBothParties = set(self.winningBidList) | set(otherWinningBidList) | set(self.winningAgentList) | set(otherWinningAgentList)
            # print("Tasl IDs in both parties: {}".format(taskIDInBothParties))
            for taskID in taskIDInBothParties:
                # Resolving the allocation conflict using the Table 1 in the paper
                action = self._CBBADecisionTable(i, k, taskID, otherWinningBidList, otherWinningAgentList, otherTimestamps)

                if action == "update":
                    new_y = otherWinningBidList.get(taskID, NOBID)
                    new_z = otherWinningAgentList.get(taskID)

                    # if there are new information
                    if new_z != self._winner(taskID) or abs(new_y - self._bid(taskID)) > EPS:
                        # Updating the list based on the conflict resolution rules
                        self.winningBidList[taskID] = new_y
                        self.winningAgentList[taskID] = new_z
                        changed = True
                
                # Resetting the winning bid and agent list to the initial value of -inf and None
                elif action == "reset":
                    if self._winner(taskID) is not None or self._bid(taskID) != NOBID:
                        self.winningBidList[taskID] = NOBID
                        self.winningAgentList[taskID] = None
                        changed = True
                # The other case is "leave", in which we will do nothing
        
        # Removing and resetting all the task after the outbidded tasks after the for loop above.
        if self._releaseOutbid(i):
            changed = True
        
        print(f"Robot {robot.robot_id} has finished its conflict resolution. Current Bundle: {self.bundle}, Current Path: {self.path}, Current Winning Bid List: {self.winningBidList}, Current Winning Agent List: {self.winningAgentList}")

        return changed
    

# This is the allocator that serves the simulation purpose, therefore it will be centralised.
# All of the decision-making on CBBA is decentralised
class CBBAAllocator(Allocator):
    name = "cbba"
    MAX_ROUNDS = 20
    
    def __init__(self) -> None:
        self.last_round = 0
    
    def allocate(self, model) -> None:
        idle = [r for r in model.robots if r.task_id is None]
        pending = model.tasks.pending
        if not idle or not pending:
            return
        
        _dbg(1, f"\n[AUCTION @ tick {model.tick}] "
                f"{len(idle)} idle robot(s) {[r.robot_id for r in idle]} | "
                f"{len(pending)} open task(s) {[t.task_id for t in pending]}")
        
        # Flush the robots' inbox
        for r in model.robots:
            r.receive_all()
        
        # Clean the robot's CBBA variables before a new bidding round begin
        for r in idle:
            c = r.CBBA
            c.task_list = pending
            c.bundle, c.path = [], []
            c.winningAgentList, c.winningBidList = {}, {}
            c.timeStamp = {}

        self.last_round = 0
        converged = False
        for rnd in range(1, self.MAX_ROUNDS + 1):
            for robot in idle:
                robot.CBBA.createBundle(model, robot, robot.CBBA.winningAgentList, robot.CBBA.winningBidList, robot.CBBA.bundle)
                robot.CBBA.broadcast(robot, rnd)
            model.comms.flush_and_deliver(model.tick)
            changed = [r.CBBA.resolveConflicts(r, r.receive_all()) for r in idle]
            self.last_round = rnd

            _dbg(1, f"[AUCTION]   round {rnd}: changed="
                    f"{dict(zip([r.robot_id for r in idle], changed))}")
            
            if not any(changed):
                converged = True
                break

        _dbg(1, f"[AUCTION] {'converged' if converged else 'HIT MAX_ROUNDS'} "
                f"after {self.last_round} round(s)")

        # Assign the tasks to the robot. It will skip already completed tasks
        for robot in idle:
            started = None
            for task_id in robot.CBBA.path:
                if robot.assign(task_id):
                    started = task_id
                    break
                _dbg(1, f"[AUCTION]   robot {robot.robot_id}: task "
                        f"{task_id} unreachable/taken, trying next in path")
            plan = [t for t in robot.CBBA.path if t != started]
            if started is None:
                _dbg(1, f"[AUCTION]   robot {robot.robot_id}: NOTHING "
                        f"assigned (path={robot.CBBA.path})")
            else:
                _dbg(1, f"[AUCTION]   robot {robot.robot_id}: starts task "
                        f"{started}, plan for later: {plan or 'none'}")

class CBPAEAllocator():
    """Das et al. (2015): bidding continues during task execution."""

    name = "cbpae"

    def allocate(self, model) -> None:
        raise NotImplementedError



class MOACBBAAllocator(Allocator):
    """Your contribution: multi-objective bids (w1, w2) + re-allocation
    triggered by DetectCellChanges (Algorithm 1, lines 3-7).

    Hook: `model.changed_cells` is populated each tick by terrain edits;
    a non-empty delta should trigger a re-bid round.
    """

    name = "moa-cbba"

    def allocate(self, model) -> None:
        raise NotImplementedError


ALLOCATORS = {a.name: a for a in (GreedyAllocator, CBBAAllocator,
                                  CBPAEAllocator, MOACBBAAllocator)}
