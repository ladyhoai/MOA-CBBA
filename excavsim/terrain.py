"""Terrain types and Phase 1 physical constants.

Hardness values H(v) feed directly into Eq. (4) and (5) of the proposal:
    T_dig = V_j * H(l_j) / rho_i        E_dig = beta * V_j * H(l_j) / rho_i
"""

from enum import IntEnum


class Terrain(IntEnum):
    SOIL = 0
    GRAVEL = 1
    ROCK = 2
    BEDROCK = 3  # not diggable
    DUMP_SITE = 4


# H(v): dimensionless hardness multiplier per terrain type.
HARDNESS = {
    Terrain.SOIL: 1.0,
    Terrain.GRAVEL: 1.8,
    Terrain.ROCK: 3.5,
    Terrain.BEDROCK: float("inf"),
    Terrain.DUMP_SITE: float("inf"),
}

DIGGABLE = {Terrain.SOIL, Terrain.GRAVEL, Terrain.ROCK}

# Battery drain coefficients (Eq. 5).
ALPHA = 0.05  # energy per grid step travelled
BETA = 0.20   # energy per dig tick (see note in costs.py)

T_UNLOAD = 3  # fixed unloading time per dump trip, in ticks (Eq. 4)

GAMMA = 0.10 # energy per meter climbed
FULL_PAYLOAD_GAMMA = 1.6 # multiplier when the robot is carrying a full payload