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


class Allocator:
    name = "base"

    def allocate(self, model) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def bid_value(model, robot, task, w1: float = 1.0, w2: float = 1.0, startPos = None) -> tuple[float, ...]:
    """Marginal cost of robot i taking task j: w1*tau_ij + w2*E_ij."""
    coord = task.cell
    terrain = Terrain(int(model.grid.terrain.data[coord]))
    h = HARDNESS[terrain]
    startPosBid = robot.cell.coordinate if startPos is None else startPos
    dig = nearest_work_cell(startPosBid, [coord],
                            model.grid.width, model.grid.height,
                            model.blocked_cells())
    if dig is None:
        return float("inf"), None
    p_star, d_task = dig
    dump = model.dump_work_cell(p_star)
    if dump is None:
        return float("inf"), None
    dumpCoord, d_dump = dump
    t = tau_ij(robot.spec, task.remaining, h, d_task, d_dump)
    e = energy_ij(robot.spec, task.remaining, h, d_task, d_dump)
    return w1 * t + w2 * e, dumpCoord
    
def pathScoreCBBA(model, robot, path, w1: float = 1.0, w2: float = 1.0) -> float:
    score = 0
    # When the robot finish dumping, its start position for the next task will be the grid coordinate of the cell 
    # it was at to dump the load. 
    dumpCoord = None
    for taskID in path:
        task = model.tasks.get(taskID)
        if task is None:
            continue
        bid, dumpCoordTemp = bid_value(model, robot, task, w1, w2, dumpCoord)
        dumpCoord = dumpCoordTemp
        score += bid
    return score

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

class CBBAAllocator(Allocator):
    """Consensus-Based Bundle Algorithm (Choi et al., 2009).

    TODO: bundle construction via marginal-gain bidding using
    `bid_value`, then the consensus/conflict-resolution table.
    Validate against the MIT ACL reference implementation.
    """

    name = "cbba"
    def __init__(self):
        self.path = []
        self.task_list = []

    # winningBidList is a local dictionary that each robot has. This will be updated in the conflict resolution phase
    def createBundle(self, model, robot, winningAgentList, winningBidList, bundle) -> None:
        # Phase 1: Bundle Construction
        currentWinningBidList = winningBidList
        currentWinningAgentList = winningAgentList
        currentBundle = bundle
        currentPath = self.path # Why does this not exist in the parameters of this function ???
        while len(currentBundle) < robot.capacity:
            allTaskIDs = [task for task in currentBundle]
            taskNotInBundles = [task for task in self.task_list if task.task_id not in allTaskIDs]
            # Helper variables for line 8
            currentImprovement = float("inf")
            Ji = -1
            JiPathLocation = -1
            # Iterate the list of available tasks
            for taskNotInBundle in taskNotInBundles:
                # Try inserting the task at every possible position in the current path
                bestBidTask = float("inf")
                bestTaskLocation = None
                for trialPosition in range(len(currentPath) + 1):
                    # REMEMBER THAT THE CURRENTPATH LIST IS A LIST OF ID
                    trialPath = currentPath[:trialPosition] + [taskNotInBundle.task_id] + currentPath[trialPosition:]
                    # Computing the path score for the new path with the inserted task
                    trialBid = pathScoreCBBA(model, robot, trialPath, model.w1, model.w2)
                    # Find the best bid and the corresponding task location (the lower the path score, the better the bid)
                    if trialBid < bestBidTask:
                        bestBidTask = trialBid
                        bestTaskLocation = trialPosition
                # Find the bid for the current task (line 7 of Algorithm 3, Consensus-Based Decentralized Auctions for Robust Task Allocation)
                improvement = bestBidTask - pathScoreCBBA(model, robot, currentPath, model.w1, model.w2)
                # Line 8 of the Algorithm to find the winning task 
                if improvement < currentWinningBidList.get(taskNotInBundle.task_id, float("inf")):
                    # Line 9 of the Algorithm to find the task to add to the bundle
                    if improvement < currentImprovement:
                        currentImprovement = improvement
                        Ji = taskNotInBundle.task_id
                        JiPathLocation = bestTaskLocation
            if Ji == -1:
                break
            # Line 11: Add the task to the bundle
            currentBundle.append(Ji)
            # Line 12: Add the task to path
            currentPath.insert(JiPathLocation, Ji)
            # Line 13: Record my own winning bid in y_{i, J_i}
            currentWinningBidList[Ji] = currentImprovement
            # Line 14: Record myself as the winner for task j
            currentWinningAgentList[Ji] = robot.robot_id
        
        # These returned lists will be sent to the other robots to resolve conflicts in Phase 2
        print(f"Robot {robot.robot_id} has finished its bundle construction. Current Bundle: {currentBundle}, Current Path: {currentPath}, Current Winning Bid List: {currentWinningBidList}, Current Winning Agent List: {currentWinningAgentList}")
        return currentWinningAgentList, currentWinningBidList, currentBundle

    # Phase 2: Conflict Resolution
    def resolveConflicts():
        return

class CBPAEAllocator(Allocator):
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
