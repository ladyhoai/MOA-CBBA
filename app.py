"""Interactive browser dashboard for excavsim (Mesa SolaraViz).

Run with:
    solara run app.py          # then open http://localhost:8765

Two tabs:
  Page 0 — operations view (proposal Fig. 1): grid map, system metrics,
           robot list, task list. All update live every tick.
  Page 1 — time series: tasks done, total energy, mean idle ratio.

Sliders/seed/allocator take effect on Reset. Cards can be dragged and
resized by grabbing their edges (layout resets on page reload).
"""

import matplotlib.pyplot as plt
import pandas as pd
import solara
from matplotlib.colors import ListedColormap

from mesa.visualization import SolaraViz, Slider, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.utils import update_counter

from excavsim.model import ExcavationModel, TaskMarker
from excavsim.robot import Stage
from excavsim.terrain import HARDNESS, Terrain
from excavsim.dynamics import WEATHER

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

def _next_task(r):
    cbba = getattr(r, "CBBA", None)
    if cbba is not None and getattr(cbba, "path", None):
        if r.task_id is None and cbba.path:
            return f"T{cbba.path[0]}"
        if len(cbba.path) > 1:
            return f"T{cbba.path[1]}"
    cbpae = getattr(r, "CBPAE", None)
    if cbpae is not None and cbpae.bidTask is not None:
        won = cbpae._winner(cbpae.bidTask) == r.robot_id
        if won and cbpae.bidTask != r.task_id:
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
    uid_to_idx = {r.unique_id: i for i, r in enumerate(model.robots)}
    rows = []
    for t in model.tasks.all:
        if t.done and t.completed_tick is not None:
            status = f"done @ {t.completed_tick}"
        elif t.done:
            status = f"finishing (robot {uid_to_idx.get(t.assigned_to, '?')})"
        elif t.assigned_to is not None:
            status = f"robot {uid_to_idx.get(t.assigned_to, '?')}"
        else:
            status = "pending"
        rows.append({
            "task": f"T{t.task_id}",
            "site": f"S{t.site_id}",
            "cell": str(t.cell),
            "remaining": f"{t.remaining:.2f}/{t.volume:.2f}",
            "sharers": model.tasks.sharers(t.site_id),
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

    here = model.tasks.at_cell(c)
    if here:
        site = here[0].site_id
        uid_to_idx0 = {r.unique_id: i for i, r in enumerate(model.robots)}
        who = [f"R{uid_to_idx0[t.assigned_to]}" for t in here
               if t.assigned_to is not None and t.assigned_to in uid_to_idx0]
        rows += [
            ("site", f"S{site}  ({len(here)} chunk"
                     f"{'s' if len(here) > 1 else ''})"),
            ("site remaining",
             f"{model.tasks.site_remaining(site):.2f} / {here[0].site_volume:.2f}"),
            ("sharing robots", ", ".join(who) if who else "—"),
        ]

    task = next((t for t in model.tasks.all if t.cell == c), None)
    if task is not None:
        uid_to_idx = {r.unique_id: i for i, r in enumerate(model.robots)}
        if task.done and task.completed_tick is not None:
            status = f"done @ {task.completed_tick}"
        elif task.assigned_to is not None:
            status = f"robot {uid_to_idx.get(task.assigned_to, '?')}"
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
            ("bundle", robot.CBBA.bundle or "—"),
            ("path", robot.CBBA.path or "—"),
        ]

    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    solara.Markdown(f"| | |\n|---|---|\n{body}")

@solara.component
def SidePanel(model):
    solara.Style(COMPACT_CSS)
    update_counter.get()
    globals()["_current_model"] = model
    renderer._post_process_applied = False
    # print(f"[stash] model id={id(model)} obstacles={len(model.dynamics.obstacles)}")
    def pick(value):
        show_layer.value = value
        refresh_map()

    with solara.Column(gap="2px"):
        solara.Markdown("**Map layer**")
        solara.ToggleButtonsSingle(value=show_layer.value,
                                   values=["terrain", "elevation"],
                                   on_value=pick, dense=True)

        done = sum(t.done for t in model.tasks.all)
        sites_done = sum(model.tasks.site_done(s) for s in model.tasks.sites)
        soil = sum(t.volume - t.remaining for t in model.tasks.all)
        energy = sum(r.energy_used for r in model.robots)
        idle = (sum(r.idle_ticks for r in model.robots)
                / max(1, model.tick * len(model.robots)))
        solara.Markdown(
            f"**tick** {model.tick} &nbsp;|&nbsp; **sites** {sites_done}/{model.tasks.n_sites} "
            f"&nbsp;|&nbsp; **chunks** {done}/{len(model.tasks.all)} "
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
            f"**changed** {len(model.changed_cells)}")

        mix = " ".join(f"{k}x{v}" for k, v in sorted(model.fleet_summary.items()))
        solara.Markdown(f"**Robots** &nbsp; _fleet: {model.fleet_mode} ({mix})_")
        solara.DataFrame(robot_frame(model), items_per_page=12)
        solara.Markdown("**Tasks**")
        solara.DataFrame(task_frame(model), items_per_page=10)# Assembly
# ------------------------------------------------------------------ #
model_instance = ExcavationModel(seed=42, hazard_rate=0.05, hazard_size=2, obstacle_rate=0.3, hazard_duration=5, max_obstacles=10, max_sharers=4)

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

def _draw_selection(ax):
    """Red box on the selected cell. Removes any previous box first so
    repeated calls don't stack patches."""
    for p in list(ax.patches):
        if getattr(p, "_gid", None) == _SEL_TAG:
            p.remove()
    rect = plt.Rectangle((sel_x.value - 0.5, sel_y.value - 0.5), 1, 1,
                         fill=False, ec="red", lw=2.0, zorder=10)
    rect._gid = _SEL_TAG
    ax.add_patch(rect)

def _draw_hazards(ax):
    """Red wash on active hazard-zone cells, re-added each frame
    (the renderer clears patches between frames)."""
    import matplotlib.pyplot as plt
    for p in list(ax.patches):
        if getattr(p, "_gid", None) == "_hazard":
            p.remove()
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
    for coll in list(ax.collections):
        if getattr(coll, "_gid", None) == "_obstacles":
            coll.remove()
    obs = getattr(_live_model(), "dynamics", None)
    # print(f"[draw] live model id={id(_live_model())} obstacles={0 if obs is None else len(obs.obstacles)}")

    if obs is None or not obs.obstacles:
        return
    xs = [o.cell[0] for o in obs.obstacles]
    ys = [o.cell[1] for o in obs.obstacles]
    sc = ax.scatter(xs, ys, marker="^", s=140, c="#e74c3c", edgecolors="black", linewidths=1.0, zorder=11)
    sc._gid = "_obstacles"

def fit_canvas(ax):
    """Applied once per renderer via post_process — and copy_renderer
    carries post_process across Reset, so the size survives resets."""
    ax.set_aspect("equal")
    ax.get_figure().set_size_inches(10.0, 10.0)
    _draw_selection(ax)
    _draw_hazards(ax)
    _draw_obstacles(ax)

renderer.post_process = fit_canvas
fit_canvas(renderer.canvas)      # apply to the initial frame too

model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "random seed"},
    "n_robots": Slider("robots", 4, 1, 12, 1),
    "n_tasks": Slider("sites", 8, 1, 30, 1),
    "max_sharers": Slider("max robots per site (1 = off)", 1, 1, 4, 1),
    "rock_fraction": Slider("rock fraction", 0.15, 0.0, 0.5, 0.05),
    "gravel_fraction": Slider("gravel fraction", 0.20, 0.0, 0.5, 0.05),
    "allocator": {
        "type": "Select",
        "value": "greedy",
        "values": ["greedy", "cbba", "cbpae", "moa-cbba"],
        "label": "allocator (moa-cbba not implemented yet)",
    },
    "fleet_mode": {
        "type": "Select",
        "value": "capacity",
        "values": ["none", "capacity", "full"],
        "label": "fleet heterogeneity (Phase 2)",
    },
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
    ],
    model_params=model_params,
    name="excavsim — Phase 1",
    play_interval=80,
)
page  # noqa: B018