"""Terrain types and Phase 1 physical constants.

Hardness values H(v) feed directly into Eq. (4) and (5) of the proposal:
    T_dig = V_j * H(l_j) / rho_i        E_dig = beta * V_j * H(l_j) / rho_i

New-reader primer: this is the smallest, most foundational file in the
project -- almost everything else (costs.py, robot.py, pathfinding.py)
imports constants from here. There is no logic to trace, just the
"physical laws" of the simulated world: what a grid cell can be made of,
how hard each material is to dig, and the fixed energy/time prices that
convert distance and dig effort into simulation ticks and battery drain.
If a number in a bid or an energy readout looks wrong, this is usually
where to check first.
"""

from enum import IntEnum


class Terrain(IntEnum):
    """What a single grid cell is made of. IntEnum (rather than plain
    Enum) so a terrain value can be stored directly as a plain integer
    in the numpy PropertyLayer grid (see model.py) instead of a Python
    object -- Mesa's property layers are numeric arrays under the hood."""

    SOIL = 0
    GRAVEL = 1
    ROCK = 2
    BEDROCK = 3  # not diggable
    DUMP_SITE = 4


# H(v): dimensionless hardness multiplier per terrain type. Bigger H
# means the same volume takes longer and more energy to dig (see
# tau_ij / energy_ij in costs.py). BEDROCK and DUMP_SITE are "infinitely
# hard", i.e. nobody can ever dig them -- that's what marks them as
# permanently impassable/undiggable obstacles rather than slow terrain.
HARDNESS = {
    Terrain.SOIL: 1.0,
    Terrain.GRAVEL: 1.8,
    Terrain.ROCK: 3.5,
    Terrain.BEDROCK: float("inf"),
    Terrain.DUMP_SITE: float("inf"),
}

# The terrain types a robot is physically able to excavate. Anything not
# in this set (bedrock, dump sites) can only ever be driven around, never
# through.
DIGGABLE = {Terrain.SOIL, Terrain.GRAVEL, Terrain.ROCK}

# Battery drain coefficients (Eq. 5).
ALPHA = 0.05  # energy per grid step travelled
BETA = 0.20   # energy per dig tick (see note in costs.py)

T_UNLOAD = 3  # fixed unloading time per dump trip, in ticks (Eq. 4)

GAMMA = 0.10 # energy per meter climbed
FULL_PAYLOAD_GAMMA = 1.6 # multiplier when the robot is carrying a full payload