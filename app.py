"""Interactive browser dashboard for excavsim (Mesa SolaraViz).

Run with:
    solara run app.py          # then open http://localhost:8765

Three tabs:
  Page 0 — operations view (proposal Fig. 1): grid map, system metrics,
           robot list, task list. All update live every tick.
  Page 1 — time series: tasks done, total energy, mean idle ratio.
  Page 2 — DEBUG: invariant checks, allocator internals, bid-vs-actual
           cost accounting, and the event log.

Sliders/seed/allocator take effect on Reset. Cards can be dragged and
resized by grabbing their edges (layout resets on page reload).

DEBUGGING. The map answers "where is everyone"; it does not answer "why
is that robot not moving". The overlays (Debug section on Page 0) and
Page 2 answer the second question:
  - planned routes show where a robot THINKS it is going, so a robot
    with no line is a robot with no plan;
  - p*/q* markers show the dig and unload cells the bid was priced on;
  - a red ring marks robots that are blocked or have an empty path;
  - Page 2 lists the invariants that are currently violated, with the
    reason spelled out, plus what each finished chunk cost against what
    its bid promised.
All of it is read-only: excavsim.debug consumes no RNG, so a run with
the dashboard open is bit-identical to a headless batch run.
"""

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd
import solara
from matplotlib.colors import ListedColormap

from mesa.visualization import SolaraViz, Slider, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.utils import update_counter

from excavsim.model import ExcavationModel, TaskMarker
from excavsim.bidding import Stage
from excavsim.terrain import HARDNESS, Terrain
from excavsim.dynamics import WEATHER
from excavsim.debug import (ERROR, INFO, LEVEL_NAME, TRACE, WARN, attach,
                            snapshot_text)

COMPACT_CSS = """
.v-application h1, .v-application h2, .v-application h3,
.v-application p { margin: 2px 0 !important; line-height: 1.25 !important; }
.v-data-table td, .v-data-table th {
    height: 22px !important; padding: 0 6px !important; font-size: 12px !important;
}
.v-data-footer { min-height: 26px !important; font-size: 11px !important; }
.v-btn { min-width: 64px !important; height: 26px !important; }
.v-card, .v-card__text { padding: 4px !important; }
"""

show_layer = solara.reactive("terrain")
sel_x = solara.reactive(0)
sel_y = solara.reactive(0)

# --- debug overlay switches ---------------------------------------- #
dbg_on = solara.reactive(True)        # master switch for every overlay
dbg_paths = solara.reactive(True)     # planned A* routes
dbg_targets = solara.reactive(True)   # p* (dig cell) and q* (unload cell)
dbg_ids = solara.reactive(True)       # R0.. / T0.. labels
dbg_blocked = solara.reactive(True)   # everything blocked_cells() returns
dbg_comms = solara.reactive(True)     # who can hear whom
dbg_sensor = solara.reactive(True)    # lidar disc + what the robot cannot see
dbg_focus = solara.reactive("all")    # "all" or "R3": draw one robot only
log_level = solara.reactive("info")   # event-log threshold


def refresh_map():
    """Force the map to redraw AND re-apply post_process (which is what
    paints the selection box; Mesa's redraw clears all patches)."""
    renderer._post_process_applied = False
    update_counter.value += 1


_SEL_TAG = "_cell_selection"

TERRAIN_CMAP = ListedColormap(
    ["#90d743", "#b5651d", "#8a8a8a", "#1a1a1a", "#2e86de"],
    name="excav_terrain",
)
if "excav_terrain" not in plt.colormaps():
    plt.colormaps.register(TERRAIN_CMAP)

STAGE_COLORS = {
    Stage.IDLE: "#bdbdbd",
    Stage.TO_TASK: "#f39c12",
    Stage.DIG: "#e74c3c",
    Stage.TO_DUMP: "#3498db",
    Stage.UNLOAD: "#9b59b6",
}

# One stable colour per robot, so a route on the map and a row in the
# debug table are obviously the same machine.
_TAB10 = plt.get_cmap("tab10")

# white outline: labels have to stay readable over dark rock and bright
# soil alike, and the map has both in every frame
_HALO = [patheffects.withStroke(linewidth=2.0, foreground="white")]


def robot_color(robot_id: int) -> str:
    from matplotlib.colors import to_hex
    return to_hex(_TAB10(robot_id % 10))


def _next_task(r):
    cbba = getattr(r, "CBBA", None)
    if cbba is not None and getattr(cbba, "path", None):
        if r.task_id is None and cbba.path:
            return f"T{cbba.path[0]}"
        if len(cbba.path) > 1:
            return f"T{cbba.path[1]}"
    cbpae = getattr(r, "CBPAE", None)
    if cbpae is not None and cbpae.bidTask is not None:
        if cbpae._winner(cbpae.bidTask) == r.robot_id \
                and cbpae.bidTask != r.task_id:
            return f"T{cbpae.bidTask}"
    return "-"


def agent_portrayal(agent):
    if isinstance(agent, TaskMarker):
        return AgentPortrayalStyle(color="#f1c40f", marker="s", size=90,
                                   zorder=2, edgecolors="black",
                                   linewidths=0.8)
    return AgentPortrayalStyle(color=STAGE_COLORS[agent.stage], marker="o",
                               size=120, zorder=3, edgecolors="black",
                               linewidths=1.0)


def layer_portrayal(layer):
    if layer.name != show_layer.value:
        return None
    if layer.name == "terrain":
        return PropertyLayerStyle(colormap="excav_terrain", vmin=0, vmax=4, alpha=0.9, colorbar=False)
    if layer.name == "elevation":
        return PropertyLayerStyle(colormap="terrain", alpha=0.9, colorbar=True)
    return None


# ------------------------------------------------------------------ #
# Fig. 1 right panel: metrics, robot list, task list (live tables)
# ------------------------------------------------------------------ #
def robot_frame(model) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(model.robots):
        rows.append({
            "robot": i,
            "class": r.spec.name,          # Phase 2: which machine is this?
            "stage": r.stage.name.lower(),
            "pos": str(r.cell.coordinate),
            "task": "—" if r.task_id is None else f"T{r.task_id}",
            "next": _next_task(r),
            "battery %": round(100 * r.battery / r.spec.battery, 1),
            "payload": f"{r.payload:.2f}/{r.spec.capacity:.1f}",
            "energy": round(r.energy_used, 2),
            "idle": r.idle_ticks,
            "done": r.tasks_completed,
        })
    return pd.DataFrame(rows)


def task_frame(model) -> pd.DataFrame:
    rows = []
    for t in model.tasks.all:
        if t.done and t.completed_tick is not None:
            status = f"done @ {t.completed_tick}"
        elif t.done:
            status = f"finishing (robot {t.assigned_to})"
        elif t.assignees:
            status = f"{t.sharers} robot(s)"
        else:
            status = "pending"
        rows.append({
            "task": f"T{t.task_id}",
            "cell": str(t.cell),
            "remaining": f"{t.remaining:.2f}/{t.volume:.2f}",
            "seats": ",".join(f"R{i}" for i in sorted(t.assignees)) or "—",
            "status": status,
        })
    return pd.DataFrame(rows)


@solara.component
def CellInspector(model):
    update_counter.get()
    renderer._post_process_applied = False
    w, h = model.grid.width, model.grid.height
    x = max(0, min(int(sel_x.value), w - 1))
    y = max(0, min(int(sel_y.value), h - 1))
    c = (x, y)

    solara.Markdown("**Inspector**")

    def set_x(v):
        sel_x.value = v
        refresh_map()

    def set_y(v):
        sel_y.value = v
        refresh_map()

    with solara.Row(gap="4px"):
        solara.InputInt("x", value=sel_x.value, on_value=set_x,
                        continuous_update=True)
        solara.InputInt("y", value=sel_y.value, on_value=set_y,
                        continuous_update=True)

    terrain = Terrain(int(model.grid.terrain.data[c]))
    hardness = HARDNESS[terrain]
    elev = float(model.grid.elevation.data[c])
    soil = float(model.grid.soil_volume.data[c])
    blocked = c in model.blocked_cells()

    rows = [
        ("cell", f"({x}, {y})"),
        ("terrain", terrain.name.lower()),
        ("hardness H(v)", "∞" if hardness == float("inf") else f"{hardness:.2f}"),
        ("elevation", f"{elev:.2f} m"),
        ("soil volume", f"{soil:.2f}"),
        ("traversable", "no" if blocked else "yes"),
    ]

    task = next((t for t in model.tasks.all if t.cell == c), None)
    if task is not None:
        if task.done and task.completed_tick is not None:
            status = f"done @ {task.completed_tick}"
        elif task.assigned_to is not None:
            status = f"robot {task.assigned_to}"
        else:
            status = "pending"
        rows += [
            ("task", f"T{task.task_id}"),
            ("remaining", f"{task.remaining:.2f} / {task.volume:.2f}"),
            ("status", status),
        ]

    robot = next((r for r in model.robots if r.cell.coordinate == c), None)
    if robot is not None:
        rows += [
            ("robot", f"R{robot.robot_id}"),
            ("stage", robot.stage.name.lower()),
            ("target task", "—" if robot.task_id is None else f"T{robot.task_id}"),
            ("class", robot.spec.name),
            ("payload", f"{robot.payload:.2f} / {robot.spec.capacity:.1f}"),
            ("v_max / dig rate",
             f"{robot.spec.v_max:.2f} / {robot.spec.dig_rate:.2f}"),
            ("drain scale", f"x{robot.spec.drain_scale:.2f}"),
            ("battery", f"{100 * robot.battery / robot.spec.battery:.1f} %"),
            ("energy (tr/cl/dig)",
             f"{robot.energy_travel:.2f} / {robot.energy_climb:.2f} / {robot.energy_dig:.2f}"),
            ("metres climbed", f"{robot.metres_climbed:.2f}"),
            ("wait ticks", robot.wait_ticks),
            ("distance", f"{robot.distance_travelled:.0f} steps"),
            # --- debug: what the mover is actually doing right now --- #
            ("dest / steps left",
             f"{robot._dest} / {len(robot._path)}"),
            ("stuck counter", robot._stuck),
            ("p* / q*", f"{robot.work_cell} / {robot.dump_cell}"),
            ("bundle b_i", robot.CBBA.bundle or "—"),
            ("path p_i", robot.CBBA.path or "—"),
            ("cbpae bid", "—" if robot.CBPAE.bidTask is None
                          else f"T{robot.CBPAE.bidTask}"),
            ("cbpae exec", "—" if robot.CBPAE.execTask is None
                           else f"T{robot.CBPAE.execTask}"),
        ]

    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    solara.Markdown(f"| | |\n|---|---|\n{body}")


@solara.component
def SidePanel(model):
    solara.Style(COMPACT_CSS)
    update_counter.get()
    globals()["_current_model"] = model
    renderer._post_process_applied = False
    mon = attach(model)

    def pick(value):
        show_layer.value = value
        refresh_map()

    with solara.Column(gap="2px"):
        solara.Markdown("**Map layer**")
        solara.ToggleButtonsSingle(value=show_layer.value,
                                   values=["terrain", "elevation"],
                                   on_value=pick, dense=True)

        done = sum(t.done for t in model.tasks.all)
        soil = sum(t.volume - t.remaining for t in model.tasks.all)
        energy = sum(r.energy_used for r in model.robots)
        idle = (sum(r.idle_ticks for r in model.robots)
                / max(1, model.tick * len(model.robots)))
        solara.Markdown(
            f"**tick** {model.tick} "
            f"&nbsp;|&nbsp; **tasks** {done}/{len(model.tasks.all)} "
            f"&nbsp;|&nbsp; **soil** {soil:.2f} &nbsp;|&nbsp; **energy** {energy:.2f} "
            f"&nbsp;|&nbsp; **idle** {idle:.3f} &nbsp;|&nbsp; **J(x)** {model.current_objective():.1f}")

        # --- Phase 4 dynamics ---
        d = model.dynamics
        w = WEATHER[d.weather] if d.weather_enabled else None
        weather_txt = (f"{d.weather} (sensor ×{w.sensor_scale:.2f}, "
                       f"traction ×{w.traction_scale:.2f})" if w else "off")
        hazard_cells = sum(len(h.cells) for h in d.hazards)
        solara.Markdown(
            f"**weather** {weather_txt} &nbsp;|&nbsp; "
            f"**hazards** {len(d.hazards)} ({hazard_cells} cells) &nbsp;|&nbsp; "
            f"**obstacles** {len(d.obstacles)} &nbsp;|&nbsp; "
            f"**changed** {mon.changed_cells_last}")

        # --- health line: the one-glance "is anything wrong" ---------- #
        HealthLine(mon)
        ConfigPanel(model)

        # --- overlay switches ---------------------------------------- #
        DebugSwitches(model)

        mix = " ".join(f"{k}x{v}" for k, v in sorted(model.fleet_summary.items()))
        solara.Markdown(f"**Robots** &nbsp; _fleet: {model.fleet_mode} ({mix})_")
        solara.DataFrame(robot_frame(model), items_per_page=12)
        solara.Markdown("**Tasks**")
        solara.DataFrame(task_frame(model), items_per_page=10)


@solara.component
def ConfigPanel(model):
    """What this run is ACTUALLY configured as.

    Every value here is read off the live model object, never off
    model_params. The two can disagree -- model_params is what the
    widgets will send on the NEXT Reset, the model is what is running
    now -- and when they did, there was no way to tell from the screen.
    """
    update_counter.get()
    a = model.allocator
    rng = model.comms.comm_range
    dyn = model.dynamics

    def row(k, v):
        return f"| {k} | {v} |"

    rows = [
        row("allocator", f"`{getattr(a, 'name', type(a).__name__)}`"),
        row("fleet", f"`{model.fleet_mode}` — "
                     + ", ".join(f"{k}×{v}" for k, v in
                                 sorted(model.fleet_summary.items()))),
        row("robots / tasks", f"{len(model.robots)} / {len(model.tasks.all)}"),
        row("objective", f"w1={model.w1:g}, w2={model.w2:g}"),
        row("comm range", "unlimited" if rng is None else f"{rng:g} cells"),
        row("lidar sensing", "ON" if model.sensing_enabled else "OFF (omniscient)"),
    ]
    if model.sensing_enabled:
        radii = ", ".join(f"R{r.robot_id} {r.sensor_radius:.1f}"
                          for r in model.robots)
        rows.append(row("sensor radius now", radii))
    rows += [
        row("hazards", f"rate {dyn.hazard_rate:g}, size {dyn.hazard_size}, "
                       f"{dyn.hazard_duration}t"),
        row("obstacles", f"rate {dyn.obstacle_rate:g}, "
                         f"max {dyn.max_obstacles}"),
        row("weather", dyn.weather if dyn.weather_enabled else "off"),
        row("grid", f"{model.grid.width}×{model.grid.height}, "
                    f"{len(model.dump_blocks)} dump sites"),
        row("seed", getattr(model, "_seed", "—")),
    ]
    # allocator-specific knobs, only the ones this allocator actually has
    for key, label in (("max_sharers", "max robots/task"),
                       ("capacity_affinity", "capacity affinity κ"),
                       ("enable_switching", "en-route switching"),
                       ("max_adds_per_round", "bundle adds/round"),
                       ("min_share", "min share/seat")):
        if hasattr(a, key):
            rows.append(row(label, f"`{getattr(a, key)}`"))

    with solara.Details("Current configuration", expand=False):
        solara.Markdown("| | |\n|---|---|\n" + "\n".join(rows))


@solara.component
def HealthLine(mon):
    """Red/amber/green, plus the single most severe open complaint."""
    errs = [t for lv, t in mon.checks if lv >= ERROR]
    warns = [t for lv, t in mon.checks if lv == WARN]
    if errs:
        solara.Markdown(f"🔴 **{len(errs)} error(s), {len(warns)} warning(s)** "
                        f"— {errs[0]}")
    elif warns:
        solara.Markdown(f"🟠 **{len(warns)} warning(s)** — {warns[0]}")
    else:
        solara.Markdown("🟢 **checks clear**")


@solara.component
def DebugSwitches(model):
    def toggle(reactive):
        def _set(v):
            reactive.value = v
            refresh_map()
        return _set

    def set_focus(v):
        dbg_focus.value = v
        refresh_map()

    with solara.Details("Debug overlays", expand=True):
        solara.Checkbox(label="overlays on", value=dbg_on.value,
                        on_value=toggle(dbg_on))
        with solara.Row(gap="6px"):
            solara.Checkbox(label="routes", value=dbg_paths.value,
                            on_value=toggle(dbg_paths))
            solara.Checkbox(label="p*/q*", value=dbg_targets.value,
                            on_value=toggle(dbg_targets))
            solara.Checkbox(label="ids", value=dbg_ids.value,
                            on_value=toggle(dbg_ids))
        with solara.Row(gap="6px"):
            solara.Checkbox(label="blocked", value=dbg_blocked.value,
                            on_value=toggle(dbg_blocked))
            solara.Checkbox(label="comm links", value=dbg_comms.value,
                            on_value=toggle(dbg_comms))
            solara.Checkbox(label="lidar", value=dbg_sensor.value,
                            on_value=toggle(dbg_sensor))
        solara.Markdown("_pick one robot below to see its **blind spot** "
                        "(red ×) — cells that are blocked but absent from "
                        "its map._")
        solara.ToggleButtonsSingle(
            value=dbg_focus.value,
            values=["all"] + [f"R{r.robot_id}" for r in model.robots],
            on_value=set_focus, dense=True)


# ------------------------------------------------------------------ #
# Page 2 — debug
# ------------------------------------------------------------------ #
def _events_frame(mon, min_level: int, robot: int | None) -> pd.DataFrame:
    rows = [{"tick": e.tick,
             "lvl": LEVEL_NAME[e.level],
             "robot": "—" if e.robot is None else f"R{e.robot}",
             "kind": e.kind,
             "what": e.text}
            for e in mon.recent(200, min_level, robot)]
    return pd.DataFrame(rows or [{"tick": "—", "lvl": "—", "robot": "—",
                                  "kind": "—", "what": "nothing logged yet"}])


@solara.component
def DebugPanel(model):
    solara.Style(COMPACT_CSS)
    update_counter.get()
    globals()["_current_model"] = model
    mon = attach(model)

    s = mon.summary()
    a = mon.allocator_state()
    acc = mon.accuracy_summary()

    with solara.Column(gap="2px"):
        solara.Markdown("### Debug")
        HealthLine(mon)

        solara.Markdown(
            f"**tick** {s['tick']} &nbsp;|&nbsp; **idle** {s['idle_robots']}/{len(model.robots)}"
            f" &nbsp;|&nbsp; **pending** {s['pending_tasks']}"
            f" &nbsp;|&nbsp; **reserved** {s['reserved_tasks']}"
            f" &nbsp;|&nbsp; **left** {s['unfinished_tasks']}"
            f" &nbsp;|&nbsp; **frozen for** {s['stalled_for']} ticks")
        solara.Markdown(
            f"**{s['tick_ms']:.1f} ms/tick** ({s['alloc_share']:.0f}% of it in "
            f"the allocator, {s['alloc_ms']:.1f} ms) &nbsp;|&nbsp; "
            f"**changed cells** {s['changed_cells']}")
        solara.Markdown("**allocator** &nbsp; "
                        + " &nbsp;|&nbsp; ".join(f"{k} `{v}`"
                                                 for k, v in a.items()))
        if acc:
            solara.Markdown(
                f"**bid vs actual** over {acc['n']} finished chunk(s): "
                f"time ×{acc['tau_ratio_min']:.2f} / "
                f"×{acc['tau_ratio_mean']:.2f} / "
                f"×{acc['tau_ratio_max']:.2f} &nbsp;|&nbsp; "
                f"energy ×{acc['e_ratio_min']:.2f} / "
                f"×{acc['e_ratio_mean']:.2f} / "
                f"×{acc['e_ratio_max']:.2f} &nbsp; _(min/mean/max)_  \n"
                f"_×1.00 means the simulation cost exactly what the bid "
                f"promised. Above 1 is contention, re-routes or drift "
                f"between costs.py and robot.py. **Below 1 breaks the "
                f"invariant in the direction costs.py does not document** "
                f"— usually a reroute that found a cheaper p\\* or q\\* "
                f"than the one the bid was priced on._")

        # --- open complaints, in full --------------------------------- #
        if mon.checks:
            body = "\n".join(
                f"- {'🔴' if lv >= ERROR else '🟠' if lv == WARN else '·'} {txt}"
                for lv, txt in mon.checks)
            solara.Markdown(f"**Checks**\n\n{body}")
        else:
            solara.Markdown("**Checks** — all clear")

        solara.Markdown("**Robots** _(internal state the ops table hides)_")
        solara.DataFrame(pd.DataFrame(mon.robot_debug_rows()), items_per_page=12)

        solara.Markdown("**Chunks** _(`claims` = who holds it in y/z, which "
                        "is not the same as who is executing it)_")
        solara.DataFrame(pd.DataFrame(mon.task_debug_rows()), items_per_page=10)

        # --- event log ------------------------------------------------ #
        with solara.Row(gap="6px"):
            solara.Markdown("**Event log**")
            solara.ToggleButtonsSingle(
                value=log_level.value,
                values=["trace", "info", "warn"],
                on_value=lambda v: log_level.set(v), dense=True)
            solara.Button("print full state to terminal", dense=True,
                          on_click=lambda: print(snapshot_text(model, 40)))
        level = {"trace": TRACE, "info": INFO, "warn": WARN}[log_level.value]
        focus = None if dbg_focus.value == "all" else int(dbg_focus.value[1:])
        solara.DataFrame(_events_frame(mon, level, focus), items_per_page=20)


# ------------------------------------------------------------------ #
# Assembly
# ------------------------------------------------------------------ #
# Must match model_params["allocator"]["value"], or the first frame runs
# one allocator while the widget claims another -- the same disagreement
# between the widgets and the live model that the config panel exists to
# expose.
model_instance = ExcavationModel(
    seed=42, allocator="moa-cbba",
    weather_enabled=True, weather_change_rate=0.05,
    # Phase 4 is off in the model defaults; the dashboard turns it on so
    # there is something to look at.
    hazard_rate=0.05, hazard_size=2, hazard_duration=8,
    obstacle_rate=0.3, max_obstacles=10)

renderer = SpaceRenderer(model_instance, backend="matplotlib")
renderer.setup_propertylayer(layer_portrayal)
renderer.setup_agents(agent_portrayal)
renderer.render()  # REQUIRED: sets the meshes SolaraViz redraws each frame


def _live_model():
    """The model the UI is currently showing. SidePanel stashes it each
    render, so this follows Reset even when renderer.space.model is None."""
    m = globals().get("_current_model")
    if m is not None:
        return m
    space = getattr(renderer, "space", None)
    return getattr(space, "model", None) or model_instance


def _clear(ax, tag: str) -> None:
    """Remove every artist this module drew under `tag`. Matplotlib keeps
    lines, patches, collections and texts in separate lists and the
    renderer only clears some of them, so overlays stack up over frames
    unless each family is swept."""
    for seq in (ax.lines, ax.patches, ax.collections, ax.texts):
        for art in list(seq):
            if getattr(art, "_gid", None) == tag:
                art.remove()


def _focused(robot) -> bool:
    return (dbg_focus.value == "all"
            or dbg_focus.value == f"R{robot.robot_id}")


def _draw_selection(ax):
    """Red box on the selected cell. Removes any previous box first so
    repeated calls don't stack patches."""
    _clear(ax, _SEL_TAG)
    rect = plt.Rectangle((sel_x.value - 0.5, sel_y.value - 0.5), 1, 1,
                         fill=False, ec="red", lw=2.0, zorder=10)
    rect._gid = _SEL_TAG
    ax.add_patch(rect)


def _draw_hazards(ax):
    """Red wash on active hazard-zone cells, re-added each frame
    (the renderer clears patches between frames)."""
    _clear(ax, "_hazard")
    dyn = getattr(_live_model(), "dynamics", None)
    if dyn is None:
        return
    for hz in dyn.hazards:
        for (x, y) in hz.cells:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                 facecolor="red", alpha=0.30,
                                 edgecolor="none", zorder=9)
            rect._gid = "_hazard"
            ax.add_patch(rect)


def _draw_obstacles(ax):
    """drawing red triangles as dynamic obstacles"""
    _clear(ax, "_obstacles")
    obs = getattr(_live_model(), "dynamics", None)
    if obs is None or not obs.obstacles:
        return
    xs = [o.cell[0] for o in obs.obstacles]
    ys = [o.cell[1] for o in obs.obstacles]
    sc = ax.scatter(xs, ys, marker="^", s=140, c="#e74c3c",
                    edgecolors="black", linewidths=1.0, zorder=11)
    sc._gid = "_obstacles"


# ------------------------------------------------------------------ #
# debug overlays
# ------------------------------------------------------------------ #
def _draw_blocked(ax, model):
    """Everything A* refuses to route through THIS tick. A route that
    looks absurd usually makes sense once you can see what it is going
    around."""
    _clear(ax, "_dbg_blocked")
    if not (dbg_on.value and dbg_blocked.value):
        return
    # DYNAMIC blockage only. Hatching everything blocked_cells() returns
    # meant re-drawing every bedrock ridge and dump block on top of the
    # terrain layer that already draws them in black and blue -- pure
    # noise over most of the map. What is worth seeing is the part that
    # was not there a moment ago and will not be there shortly.
    for (x, y) in model.dynamics.blocked():
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                             hatch="///", edgecolor="#222222", lw=0.0,
                             alpha=0.45, zorder=7)
        rect._gid = "_dbg_blocked"
        ax.add_patch(rect)


def _draw_sensor(ax, model):
    """The lidar disc, and — in focus mode — the blind spot.

    With sensing on, a robot plans against `known_blocked()`, not the
    truth. The gap between those two is the single most useful thing to
    see on this map and the only one that cannot be inferred from any
    other overlay: a route that looks reckless is usually a route into a
    hazard the robot has no way of knowing about yet.

    Solid ring   = current sensing radius (sigma_i x weather scale)
    Red hatching = truly blocked, NOT in this robot's map (focus mode)
    """
    _clear(ax, "_dbg_sensor")
    if not (dbg_on.value and dbg_sensor.value
            and getattr(model, "sensing_enabled", False)):
        return
    for r in model.robots:
        if not _focused(r):
            continue
        x, y = r.cell.coordinate
        disc = plt.Circle((x, y), r.sensor_radius, fill=False,
                          ec=robot_color(r.robot_id), lw=1.0, ls=":",
                          alpha=0.75, zorder=5)
        disc._gid = "_dbg_sensor"
        ax.add_patch(disc)

    # Blind spot only makes sense for ONE robot: with "all" selected the
    # unknown sets differ per robot and overlaying them means nothing.
    if dbg_focus.value == "all":
        return
    rid = int(dbg_focus.value[1:])
    robot = next((r for r in model.robots if r.robot_id == rid), None)
    if robot is None:
        return
    unknown = model.dynamics.blocked() - robot.known_blocked()
    for (x, y) in unknown:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                             hatch="xxx", edgecolor="#c0392b", lw=0.0,
                             alpha=0.85, zorder=8)
        rect._gid = "_dbg_sensor"
        ax.add_patch(rect)


def _draw_paths(ax, model):
    """The remaining A* route per robot, from where it stands now."""
    _clear(ax, "_dbg_path")
    if not (dbg_on.value and dbg_paths.value):
        return
    for r in model.robots:
        if not r._path or not _focused(r):
            continue
        xs = [r.cell.coordinate[0]] + [c[0] for c in r._path]
        ys = [r.cell.coordinate[1]] + [c[1] for c in r._path]
        line, = ax.plot(xs, ys, ls="--", lw=1.8, color=robot_color(r.robot_id),
                        alpha=0.95, zorder=8, solid_capstyle="round")
        line._gid = "_dbg_path"


def _draw_targets(ax, model):
    """p* (the dig cell the bid was priced from) and q* (the unload
    cell). If a robot is walking somewhere that is neither, its leg was
    re-planned and the bid no longer describes what it is doing."""
    _clear(ax, "_dbg_target")
    if not (dbg_on.value and dbg_targets.value):
        return
    for r in model.robots:
        if not _focused(r):
            continue
        col = robot_color(r.robot_id)
        if r.work_cell is not None:
            sc = ax.scatter([r.work_cell[0]], [r.work_cell[1]], marker="x",
                            s=110, c=col, linewidths=2.0, zorder=12)
            sc._gid = "_dbg_target"
        if r.dump_cell is not None:
            sc = ax.scatter([r.dump_cell[0]], [r.dump_cell[1]], marker="*",
                            s=150, c=col, edgecolors="black", linewidths=0.5,
                            zorder=12)
            sc._gid = "_dbg_target"


def _draw_alerts(ax, model):
    """Ring the robots that are in trouble: blocked by a neighbour, or
    committed to a leg with no route to walk. These are the ones that
    look identical to a healthy robot on the plain map."""
    _clear(ax, "_dbg_alert")
    if not dbg_on.value:
        return
    for r in model.robots:
        stalled = (r.stage in (Stage.TO_TASK, Stage.TO_DUMP) and not r._path)
        if not (r._stuck or stalled):
            continue
        x, y = r.cell.coordinate
        ring = plt.Circle((x, y), 0.52, fill=False,
                          ec="#ff2d55" if stalled else "#ffb300",
                          lw=2.2, zorder=13)
        ring._gid = "_dbg_alert"
        ax.add_patch(ring)


def _draw_ids(ax, model):
    """R# next to each robot in its own colour, T# on each chunk cell."""
    _clear(ax, "_dbg_id")
    if not (dbg_on.value and dbg_ids.value):
        return
    for r in model.robots:
        x, y = r.cell.coordinate
        txt = ax.text(x + 0.45, y + 0.45, f"R{r.robot_id}", fontsize=9,
                      color=robot_color(r.robot_id), zorder=14,
                      ha="left", va="bottom", weight="bold")
        txt.set_path_effects(_HALO)
        txt._gid = "_dbg_id"
    by_cell: dict = {}
    for t in model.tasks.unfinished:
        by_cell.setdefault(t.cell, []).append(t.task_id)
    for (x, y), ids in by_cell.items():
        label = ",".join(f"T{i}" for i in ids[:2]) + ("…" if len(ids) > 2 else "")
        txt = ax.text(x - 0.48, y - 0.5, label, fontsize=7, color="#1a1a1a",
                      zorder=14, ha="left", va="top")
        txt.set_path_effects(_HALO)
        txt._gid = "_dbg_id"


def _draw_comms(ax, model):
    """Who can hear whom. Only meaningful with a finite comm_range —
    with the default (None) every robot hears every other and the mesh
    would just be noise."""
    _clear(ax, "_dbg_comm")
    if not (dbg_on.value and dbg_comms.value):
        return
    comms = getattr(model, "comms", None)
    if comms is None or comms.comm_range is None:
        return
    seen = set()
    for r in model.robots:
        for other in comms.neighbors(r):
            key = tuple(sorted((r.robot_id, other.robot_id)))
            if key in seen:
                continue
            seen.add(key)
            ax_, ay = r.cell.coordinate
            bx, by = other.cell.coordinate
            line, = ax.plot([ax_, bx], [ay, by], ls=":", lw=0.9,
                            color="#00b894", alpha=0.7, zorder=6)
            line._gid = "_dbg_comm"


def fit_canvas(ax):
    """Applied once per renderer via post_process — and copy_renderer
    carries post_process across Reset, so the size survives resets."""
    ax.set_aspect("equal")
    ax.get_figure().set_size_inches(10.0, 10.0)
    _draw_selection(ax)
    _draw_hazards(ax)
    _draw_obstacles(ax)

    model = _live_model()
    if model is None:
        return
    # Overlays must not be able to take the dashboard down: a debug view
    # that crashes the run it is debugging is worse than no debug view.
    try:
        _draw_blocked(ax, model)
        _draw_sensor(ax, model)
        _draw_comms(ax, model)
        _draw_paths(ax, model)
        _draw_targets(ax, model)
        _draw_alerts(ax, model)
        _draw_ids(ax, model)
    except Exception as exc:                      # pragma: no cover
        print(f"[debug overlay] {type(exc).__name__}: {exc}")


renderer.post_process = fit_canvas
fit_canvas(renderer.canvas)      # apply to the initial frame too

model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "random seed"},
    "n_robots": Slider("robots", 4, 1, 12, 1),
    "n_tasks": Slider("sites", 8, 1, 30, 1),
    "rock_fraction": Slider("rock fraction", 0.15, 0.0, 0.5, 0.05),
    "gravel_fraction": Slider("gravel fraction", 0.20, 0.0, 0.5, 0.05),
    "allocator": {
        "type": "Select",
        "value": "moa-cbba",
        "values": ["greedy", "cbba", "cbpae", "moa-cbba"],
        "label": "allocator",
    },
    "fleet_mode": {
        # Must match ExcavationModel's own default. SolaraViz passes every
        # entry in model_params to the constructor on Reset, so a stale
        # value here silently OVERRIDES the model default -- which is how
        # this read "capacity" while the code had moved to "full".
        "type": "Select",
        "value": "full",
        "values": ["none", "capacity", "full"],
        "label": "fleet heterogeneity",
    },
    "sensing_enabled": {
        "type": "Checkbox",
        "value": True,
        "label": "lidar sensing (off = omniscient)",
    },
    "comm_range": Slider("comm range (0 = unlimited)", 10.0, 0.0, 45.0, 1.0),
    "w1": Slider("w1 (makespan weight)", 1.0, 0.0, 5.0, 0.25),
    "w2": Slider("w2 (energy weight)", 1.0, 0.0, 5.0, 0.25),
    "hazard_rate": Slider("hazard rate", 0.0, 0.0, 0.5, 0.05),
    "obstacle_rate": Slider("obstacle rate", 0.0, 0.0, 0.6, 0.05),
    "weather_enabled": {
        "type": "Checkbox", "value": True, "label": "weather",
    },
    # Enabling weather with rate 0 leaves it on "clear" forever, and
    # clear has sensor_scale = traction_scale = 1.0 -- i.e. the switch
    # appears on and does precisely nothing.
    "weather_change_rate": Slider("weather change rate", 0.05, 0.0, 0.3, 0.01),
    "bedrock_ridges": Slider("bedrock ridges", 6, 0, 15, 1),
    "width": 32,
    "height": 32,
}

page = SolaraViz(
    model_instance,
    renderer,
    components=[
        # Page 0 — operations view (space map is auto-inserted first)
        (CellInspector, 0),
        (SidePanel, 0),
        # Page 1 — time series
        make_plot_component("tasks_done", page=1),
        make_plot_component("total_energy", page=1),
        make_plot_component("mean_idle_ratio", page=1),
        # Page 2 — debug
        (DebugPanel, 2),
    ],
    model_params=model_params,
    name="excavsim — Phase 1",
    play_interval=80,
)
page  # noqa: B018