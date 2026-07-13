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
    if layer.name == "terrain":
        return PropertyLayerStyle(colormap="excav_terrain", vmin=0, vmax=4,
                                  alpha=0.9, colorbar=False)
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
def SystemMetrics(model):
    update_counter.get()  # re-render every model step
    done = sum(t.done for t in model.tasks.all)
    soil = sum(t.volume - t.remaining for t in model.tasks.all)
    energy = sum(r.energy_used for r in model.robots)
    idle = (sum(r.idle_ticks for r in model.robots)
            / max(1, model.tick * len(model.robots)))
    solara.Markdown(f"""### System metrics
| | |
|---|---|
| tick | **{model.tick}** |
| tasks done | **{done} / {len(model.tasks.all)}** |
| soil moved | {soil:.2f} |
| total energy | {energy:.2f} |
| mean idle ratio | {idle:.3f} |
| J(x) so far (w₁={model.w1}, w₂={model.w2}) | **{model.current_objective():.1f}** |
""")


@solara.component
def RobotList(model):
    update_counter.get()
    solara.Markdown("### Robots")
    solara.DataFrame(robot_frame(model), items_per_page=12)


@solara.component
def TaskList(model):
    update_counter.get()
    solara.Markdown("### Tasks")
    solara.DataFrame(task_frame(model), items_per_page=10)


# ------------------------------------------------------------------ #
# Assembly
# ------------------------------------------------------------------ #
model_instance = ExcavationModel(seed=42)

renderer = SpaceRenderer(model_instance, backend="matplotlib")
renderer.setup_propertylayer(layer_portrayal)
renderer.setup_agents(agent_portrayal)
renderer.render()  # REQUIRED: sets the meshes SolaraViz redraws each frame


def fit_canvas(ax):
    """Applied once per renderer via post_process — and copy_renderer
    carries post_process across Reset, so the size survives resets."""
    ax.set_aspect("equal")
    ax.get_figure().set_size_inches(10.0, 10.0)


renderer.post_process = fit_canvas
fit_canvas(renderer.canvas)      # apply to the initial frame too

model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "random seed"},
    "n_robots": Slider("robots", 4, 1, 12, 1),
    "n_tasks": Slider("tasks", 10, 1, 30, 1),
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
        (SystemMetrics, 0),
        (RobotList, 0),
        (TaskList, 0),
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