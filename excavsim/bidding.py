from .costs import energy_ij, tau_ij
from .pathfinding import nearest_work_path, path_climb
from .terrain import HARDNESS, Terrain
from enum import Enum, auto

INF = float("inf")
EPS = 1e-9
NOBID = float("-inf")

class Stage(Enum):
    IDLE = auto()
    TO_TASK = auto()
    DIG = auto()
    TO_DUMP = auto()
    UNLOAD = auto()

def leg_cost(model, robot, task, startPos = None) -> tuple[float, ...]:
    """Cost of robot i taking task j: tau_ij and E_ij."""
    coord = task.cell
    terrain = Terrain(int(model.grid.terrain.data[coord]))
    h = HARDNESS[terrain]
    startPosLeg = robot.cell.coordinate if startPos is None else startPos

    dig = nearest_work_path(startPosLeg, [coord],
                            model.grid.width, model.grid.height,
                            model.blocked_cells())
    if dig is None:
        return float("inf"), float("inf"), None # type: ignore
    p_star, d_task, path_task = dig
    dump = model.dump_work_path(p_star)
    if dump is None:
        return float("inf"), float("inf"), None # type: ignore
    dumpCoord, d_dump, path_dump = dump

    climbEnergyUsageToTask = path_climb(path_task, model.grid.elevation.data)
    climbEnergyUsageTaskToDump = path_climb(path_dump, model.grid.elevation.data)

    t = tau_ij(robot.spec, task.remaining, h, d_task, d_dump)
    e = energy_ij(robot.spec, task.remaining, h, d_task, d_dump, climbEnergyUsageToTask, climbEnergyUsageTaskToDump)
    return t, e, dumpCoord

def bid_value(model, robot, task, w1: float = 1.0, w2: float = 1.0, startPos = None) -> tuple[float, ...]:
    """Marginal cost of robot i taking task j: w1*tau_ij + w2*E_ij.
    (Cost convention, used by the greedy baseline only.)"""
    t, e, dumpCoord = leg_cost(model, robot, task, startPos)
    if dumpCoord is None:
        return INF, None # type: ignore
    return w1 * t + w2 * e, dumpCoord

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
