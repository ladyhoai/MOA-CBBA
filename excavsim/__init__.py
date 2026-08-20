"""excavsim: the multi-robot excavation simulation package.

Re-exports the pieces most callers need (ExcavationModel, ExcavatorRobot,
the cost/bidding functions, the three allocators) so scripts can write
`from excavsim import ExcavationModel` instead of reaching into
individual submodules. See the package README.md and DOCUMENTATION.md
at the repo root for an overview of how the pieces fit together.
"""

from .bidding import Stage, bid_value, leg_cost, residual_cost
from .comms import CommNetwork, Message
from .costs import RobotSpec, energy_ij, n_trips, objective, tau_ij
from .model import ExcavationModel
from .robot import ExcavatorRobot
from .tasks import Task, TaskRegistry
from .fleet import ROBOT_CLASSES, build_fleet, fleet_summary
from .allocation import ALLOCATORS, CBBAAgent, CBBAAllocator, GreedyAllocator
from .CBPAE import CBPAEAgent, CBPAEAllocator
from .MOACBBA import MOACBBAAgent, MOACBBAAllocator

__all__ = ["CommNetwork", "Message", "ExcavationModel", "ExcavatorRobot",
           "Task", "TaskRegistry", "Stage",
           "RobotSpec", "tau_ij", "energy_ij", "n_trips", "objective",
           "leg_cost", "bid_value", "residual_cost",
           "ROBOT_CLASSES", "build_fleet", "fleet_summary",
           "ALLOCATORS", "GreedyAllocator", "CBBAAllocator", "CBBAAgent",
           "CBPAEAllocator", "CBPAEAgent",
           "MOACBBAAllocator", "MOACBBAAgent"]