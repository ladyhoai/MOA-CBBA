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
        return PropertyLayerStyle(colormap="excav_terrain", vmin=0, vmax=4,
                                  alpha=0.9, colorbar=False)
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
            "stage": r.stage.name.lower(),
            "pos": str(r.cell.coordinate),
            "task": "—" if r.task_id is None else f"T{r.task_id}",
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
            "cell": str(t.cell),
            "remaining": f"{t.remaining:.2f}/{t.volume:.2f}",
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
            ("payload", f"{robot.payload:.2f} / {robot.spec.capacity:.1f}"),
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
            f"**tick** {model.tick} &nbsp;|&nbsp; **tasks** {done}/{len(model.tasks.all)} "
            f"&nbsp;|&nbsp; **soil** {soil:.2f} &nbsp;|&nbsp; **energy** {energy:.2f} "
            f"&nbsp;|&nbsp; **idle** {idle:.3f} &nbsp;|&nbsp; **J(x)** {model.current_objective():.1f}")

        solara.Markdown("**Robots**")
        solara.DataFrame(robot_frame(model), items_per_page=12)
        solara.Markdown("**Tasks**")
        solara.DataFrame(task_frame(model), items_per_page=10)# ------------------------------------------------------------------ #
# Assembly
# ------------------------------------------------------------------ #
model_instance = ExcavationModel(seed=42)

renderer = SpaceRenderer(model_instance, backend="matplotlib")
renderer.setup_propertylayer(layer_portrayal)
renderer.setup_agents(agent_portrayal)
renderer.render()  # REQUIRED: sets the meshes SolaraViz redraws each frame

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

def fit_canvas(ax):
    """Applied once per renderer via post_process — and copy_renderer
    carries post_process across Reset, so the size survives resets."""
    ax.set_aspect("equal")
    ax.get_figure().set_size_inches(10.0, 10.0)
    _draw_selection(ax)


renderer.post_process = fit_canvas
fit_canvas(renderer.canvas)      # apply to the initial frame too

model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "random seed"},
    "n_robots": Slider("robots", 4, 1, 12, 1),
    "n_tasks": Slider("tasks", 8, 1, 30, 1),
    "rock_fraction": Slider("rock fraction", 0.15, 0.0, 0.5, 0.05),
    "gravel_fraction": Slider("gravel fraction", 0.20, 0.0, 0.5, 0.05),
    "allocator": {
        "type": "Select",
        "value": "greedy",
        "values": ["greedy", "cbba", "cbpae", "moa-cbba"],
        "label": "allocator (only greedy implemented so far)",
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