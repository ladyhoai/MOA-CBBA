"""Robot "vehicle classes" and how a mixed fleet is assembled.

New-reader primer: RobotSpec (costs.py) describes ONE robot's physical
properties; this file is where those properties get grouped into named
classes ("small"/"medium"/"large", like light/medium/heavy excavators)
and where a whole fleet of n robots gets built for a simulation run. If
you just want to know "what robots exist in this simulation and how
different are they", start here.
"""

from __future__ import annotations

from .costs import RobotSpec

# name -> full Phase 2 spec
ROBOT_CLASSES: dict[str, RobotSpec] = {
    "small": RobotSpec(capacity=1.0, v_max=1.2, dig_rate=0.35,
                       battery=80.0, sensor_range=6.0,
                       drain_scale=0.8, name="small"),
    "medium": RobotSpec(capacity=2.0, v_max=1.0, dig_rate=0.50,
                        battery=100.0, sensor_range=8.0,
                        drain_scale=1.0, name="medium"),
    "large": RobotSpec(capacity=3.0, v_max=0.8, dig_rate=0.75,
                       battery=140.0, sensor_range=10.0,
                       drain_scale=1.4, name="large"),
}
 
BASELINE = "medium"          # the class "none" mode falls back to
DEFAULT_MIX = ("small", "medium", "large")
 
HETEROGENEITY_MODES = ("none", "capacity", "full")
 
 
def _capacity_only(base: RobotSpec, other: RobotSpec) -> RobotSpec:
    """`other`'s payload capacity, everything else from `base`.

    Builds the fleet_mode="capacity" ablation: robots differ ONLY in
    hopper size C_i, so the only cost term that varies between them is
    the trip count n_ij = ceil(V_j/C_i) (see costs.n_trips). Speed, dig
    rate, battery and drain are held at the baseline class, which
    isolates "does payload capacity alone change the allocation?"

    CALLED BY: build_fleet, when mode == "capacity".
    """
    return RobotSpec(capacity=other.capacity, v_max=base.v_max,
                     dig_rate=base.dig_rate, battery=base.battery,
                     sensor_range=base.sensor_range,
                     drain_scale=base.drain_scale, name=other.name)
 
 
def build_fleet(n_robots: int, mode: str = "capacity",
                mix: tuple[str, ...] | None = None,
                rng=None) -> list[RobotSpec]:
    """Return one RobotSpec per robot -- the fleet for a whole run.

    The three `mode` values are the Phase 2 heterogeneity ablation
    ladder, from least to most varied:
      "none"     -- every robot is the BASELINE class ("medium"). The
                    homogeneous control condition.
      "capacity" -- robots differ only in hopper size (_capacity_only).
      "full"     -- robots use their class spec verbatim, so speed, dig
                    rate, battery, sensor range and drain all differ.

    Assignment is round-robin: robot i gets class mix[i % len(mix)]. For
    n_robots=4 and the default ("small","medium","large") mix that is
    small, medium, large, small. No randomness is involved, so the same
    n_robots always yields the same class multiset -- important when
    comparing allocators at matched seeds, since a randomly sampled
    fleet would confound allocator performance with fleet luck.
    Pass `rng` to shuffle the assignment order instead (the multiset of
    classes is unchanged; only which robot gets which class moves).

    CALLED BY: model.ExcavationModel.__init__, once per run; its result
    is stored as model.fleet_specs and one ExcavatorRobot is built per
    entry.
    """
    if mode not in HETEROGENEITY_MODES:
        raise ValueError(f"mode must be one of {HETEROGENEITY_MODES}, got {mode!r}")
 
    base = ROBOT_CLASSES[BASELINE]
    if mode == "none":
        return [base] * n_robots
 
    mix = mix or DEFAULT_MIX
    unknown = set(mix) - set(ROBOT_CLASSES)
    if unknown:
        raise ValueError(f"unknown robot class(es): {sorted(unknown)}")
 
    specs = []
    for i in range(n_robots):
        cls = ROBOT_CLASSES[mix[i % len(mix)]]
        specs.append(cls if mode == "full" else _capacity_only(base, cls))
 
    if rng is not None:
        rng.shuffle(specs)
    return specs
 
 
def fleet_summary(specs: list[RobotSpec]) -> dict[str, int]:
    """Class -> count, for logging and for the report's method section.

    e.g. {"small": 2, "medium": 1, "large": 1} for a 4-robot default mix.

    CALLED BY: model.ExcavationModel.__init__, stored as
    model.fleet_summary and displayed by the GUI's config panel.
    """
    out: dict[str, int] = {}
    for s in specs:
        out[s.name] = out.get(s.name, 0) + 1
    return out