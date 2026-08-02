from .bidding import leg_cost, bid_value, _better_bid, INF, EPS, NOBID, Stage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ExcavationModel
    from .robot import ExcavatorRobot
    from .comms import Message
# run 1 greedy: 135
# run 2 CBBA: 147  ; 2/8: 143
# run 3 CBPAE: 160 ; 1/8: 154
class CBPAEAgent():
    def __init__(self) -> None:
        self.winningAgentList = {} # Map: task ID -> Robot ID 
        self.winningBidList = {}  # Map: task ID -> best bid seen among robots
        self.timeStamp = {}
        self.bidTask = None # The task that this robot is bidding on
        self.execTask = None # The task that the robot will execute after the current task is done

    def _bid(self, task_id):
        return self.winningBidList.get(task_id, NOBID)
    
    def _winner(self, task_id):
        return self.winningAgentList.get(task_id, None)
    
    def _stamp(self, agent_id):
        return self.timeStamp.get(agent_id, -1)
    
    def biddable_tasks(self, model, robot):
        out = []
        for t in model.tasks.all:
            if t.future_remaining > 0 and not t.done:
                _, _, end = leg_cost(model, robot, t)
                if end is not None: # if the task is legit and executable
                    out.append(t)
        return out
    
    def compute_bid(self, model, robot, lam=0.99, task_reward=10.0):
        best_task, best_bid = None, NOBID
        for t in self.biddable_tasks(model, robot):
            w = self._winner(t.task_id)

            # bid only on task no one has claimed
            if w is not None and w != robot.robot_id:
                continue
            if robot.stage == Stage.IDLE:
                tau, e, _ = leg_cost(model, robot, t)
            else:
                tau, e, _ = leg_cost(model, robot, t, robot.dump_cell)
            bid = -(model.w1 * tau + model.w2 * e)
            if bid > best_bid:
                best_bid, best_task = bid, t.task_id
        
        self.bidTask = best_task
        if best_task is not None:
            if _better_bid(best_bid, robot.robot_id, self._bid(best_task), self._winner(best_task)):
                self.winningBidList[best_task] = best_bid
                self.winningAgentList[best_task] = robot.robot_id
        return best_task, best_bid
    
    def broadcast(self, robot, now):
        self.timeStamp[robot.robot_id] = now
        robot.send({
            "sender": robot.robot_id,
            "y": dict(self.winningBidList),
            "z": dict(self.winningAgentList),
            "s": dict(self.timeStamp),
            "exec": self.execTask
        })

    def resolveConflicts(self, robot: "ExcavatorRobot", messages: "Message"):
        i = robot.robot_id
        changed = False
        for msg in messages:
            p = msg.payload
            k = p.get("sender")
            if k is None or k == i:
                continue
            otherRobot_y = p.get("y", {})
            otherRobot_z = p.get("z", {})
            exec_task = p.get("exec")

            # Ignoring the task which has already been executed
            if exec_task is not None and self._winner(exec_task) != k:
                self.winningAgentList[exec_task] = k
                self.winningBidList[exec_task] = INF
                changed = True

            for j in set(otherRobot_y) | set(otherRobot_z):
                yk = otherRobot_y.get(j, NOBID)
                zk = otherRobot_z.get(j)
                if _better_bid(yk, zk, self._bid(j), self._winner(j)):
                    self.winningBidList[j] = yk
                    self.winningAgentList[j] = zk
                    changed = True

        return changed

# The simulator will be running the allocate method in this class
class CBPAEAllocator():
    """Das et al. (2015): bidding continues during task execution."""

    name = "cbpae"

    def __init__(self) -> None:
        self.round = 0

    def allocate(self, model: "ExcavationModel") -> None:
        self.round += 1
        now = self.round

        # Do the auctioning and resolving conflicts to start assigning task to agents
        for r in model.robots:
            r.CBPAE.execTask = r.task_id
            r.CBPAE.compute_bid(model, r)
            r.CBPAE.broadcast(r, now)
        model.comms.flush_and_deliver(model.tick)
        for r in model.robots:
            r.CBPAE.resolveConflicts(r, r.receive_all())

        # Assigning the task to the agents and starting the execution
        for r in model.robots:
            if r.task_id is not None:
                continue
            j = r.CBPAE.bidTask
            if j is None:
                continue
            task = model.tasks.get(j)
            if not task.done and r.CBPAE._winner(j) == r.robot_id:
                if r.assign(j):
                    r.CBPAE.execTask = j
                    task.future_remaining -= r.spec.capacity
