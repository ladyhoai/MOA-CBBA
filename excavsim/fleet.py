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
    """`other`'s payload capacity, everything else from `base`."""
    return RobotSpec(capacity=other.capacity, v_max=base.v_max,
                     dig_rate=base.dig_rate, battery=base.battery,
                     sensor_range=base.sensor_range,
                     drain_scale=base.drain_scale, name=other.name)
 
 
def build_fleet(n_robots: int, mode: str = "capacity",
                mix: tuple[str, ...] | None = None,
                rng=None) -> list[RobotSpec]:
    """Return one RobotSpec per robot.
 
    Classes are dealt round-robin from `mix`, so the composition is
    deterministic and balanced for a given n_robots — important when
    comparing allocators at matched seeds, since a randomly sampled
    fleet would confound allocator performance with fleet luck.
    Pass `rng` to shuffle the assignment order instead (the multiset of
    classes is unchanged; only which robot gets which class moves).
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
    """Class -> count, for logging and for the report's method section."""
    out: dict[str, int] = {}
    for s in specs:
        out[s.name] = out.get(s.name, 0) + 1
    return out
