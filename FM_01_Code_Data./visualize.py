"""Visualize held-out demonstrations vs conditional Flow Matching predictions.
Run from project root: python visualize.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib

# Use a noninteractive backend to save PNGs without opening a GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# This import is currently unused.
from plotly.subplots import make_subplots

# Reuse the trained architecture, condition builder, sampler, and default model output path.
from waypoint_fm import Flow, condition, sample_waypoints, DEFAULT_OUTPUT

# Resolve data paths relative to this file, independent of the terminal working directory.
# Read model artifacts from folder 2 and write visualization artifacts to folder 3.
ROOT = Path(__file__).resolve().parent
MODEL_OUT = DEFAULT_OUTPUT
# Create the visualization folder and any missing parent directories.
OUT = ROOT.parent / "FM_03_Visualizations." / "output" / "visualizations"
OUT.mkdir(exist_ok=True, parents=True)
# Load weights, task names, waypoint count, and normalization metadata on CPU.
# Only load trusted checkpoints when using weights_only=False.
ckpt = torch.load(MODEL_OUT / "model.pt", map_location="cpu", weights_only=False)
# Restore a network with 3*k output features and 6 + task_count condition features.
model = Flow(3 * ckpt["k"], 6 + len(ckpt["tasks"]), ckpt["width"])
# Apply saved weights and enable evaluation mode; this script does not train the model.
model.load_state_dict(ckpt["model"])
model.eval()
# Load the held-out segments and their interpolated reference waypoints.
examples = json.loads((MODEL_OUT / "test_examples.json").read_text())
# Assume all evaluation examples belong to the CSV named by the first example.
raw = pd.read_csv(ROOT / "data" / examples[0]["csv"])
# Separate contiguous task runs whenever the task name changes.
groups = raw.current_task.ne(raw.current_task.shift()).cumsum()
# Index recorded segments by task name; repeated names overwrite earlier segments.
by_task = {str(g.current_task.iloc[0]): g for _, g in raw.groupby(groups, sort=False)}
# Collect JSON evaluation records and coordinate arrays for the summary plots.
results = []
traces = []
# Process each evaluation segment in its original order.
for i, e in enumerate(examples):
    # start/goal have shape (3,); ref has shape (k, 3). All are in meters.
    # The reference interpolates marked rows and endpoints, rather than the full recorded trajectory.
    start = np.array(e["start_m"])
    goal = np.array(e["goal_m"])
    ref = np.array(e["target_m"])
    # Generate interior waypoints conditioned on task and endpoints.
    # Saved mean/std normalize the condition; yscale restores residual magnitudes.
    # Use 80 Euler steps, random seed 42, and CPU inference with gradients disabled.
    pred = sample_waypoints(
        model,
        condition(e, ckpt["tasks"]),
        start,
        goal,
        np.asarray(ckpt["cond_mean"]),
        np.asarray(ckpt["cond_std"]),
        np.asarray(ckpt["yscale"]),
        ckpt["k"],
        80,
        42,
        "cpu",
    )
    # Sample the straight-line baseline at the same interior fractions.
    # For k=4 these are 20%, 40%, 60%, and 80%; None adds an axis for XYZ broadcasting.
    straight = start + (goal - start) * np.linspace(0, 1, ckpt["k"] + 2)[1:-1, None]
    # Read every recorded TCP position in the segment and convert mm to m.
    rawpath = by_task[e["task"]][["EE_x", "EE_y", "EE_z"]].to_numpy(float) / 1000
    # Prepend the start and append the goal, then convert plot coordinates to mm.
    # [None] turns a (3,) point into a (1, 3) row; np.r_ joins rows.
    # mp: model, rr: reference, sg: straight baseline, truepath: recorded trajectory.
    mp = np.r_[start[None], pred, goal[None]] * 1000
    rr = np.r_[start[None], ref, goal[None]] * 1000
    sg = np.r_[start[None], straight, goal[None]] * 1000
    truepath = rawpath * 1000
    # Compute mean Euclidean distance to corresponding reference interior points in mm.
    # This metric does not measure full-trajectory error, collisions, or task success.
    err = np.linalg.norm(pred - ref, axis=1).mean() * 1000
    baseerr = np.linalg.norm(straight - ref, axis=1).mean() * 1000
    # Convert arrays to JSON-compatible lists; round errors to two decimals and coordinates to three.
    result = {
        "task": e["task"],
        "model_error_mm": round(float(err), 2),
        "straight_baseline_error_mm": round(float(baseerr), 2),
        "raw_waypoint_marks": e["raw_mark_count"],
        "start_mm": (start * 1000).tolist(),
        "goal_mm": (goal * 1000).tolist(),
        "generated_waypoints_mm": (pred * 1000).round(3).tolist(),
    }
    results.append(result)
    # Retain the task and four paths for the summary and interactive figures.
    traces.append((e["task"], truepath, mp, rr, sg))
    # Plot one task in 3D: gray recorded path, orange prediction, teal dashed reference.
    # Transpose and unpack coordinate arrays to pass X, Y, and Z separately.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        *truepath.T,
        color="#263238",
        lw=1.5,
        alpha=0.65,
        label="Recorded TCP trajectory",
    )
    ax.plot(
        *mp.T,
        color="#e4572e",
        lw=2.2,
        marker="o",
        markersize=5,
        label="Flow Matching + endpoints",
    )
    ax.plot(
        *rr.T,
        color="#2a9d8f",
        lw=1.7,
        ls="--",
        marker="x",
        label="Interpolated supervision",
    )
    # Mark the fixed start in green and goal in blue.
    ax.scatter(*start * 1000, color="green", s=70, label="Start")
    ax.scatter(*goal * 1000, color="blue", s=70, label="Goal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f'{e["task"]}\nFM {err:.1f} mm | straight baseline {baseerr:.1f} mm',
        fontsize=11,
    )
    ax.legend(fontsize=7, loc="upper left")
    # Use all displayed points to find bounds, with a minimum span of 100 mm.
    # Apply the same numeric range around each axis midpoint for consistent comparisons.
    a = np.vstack([truepath, mp, rr])
    span = np.maximum(a.max(axis=0) - a.min(axis=0), 100)
    mid = (a.max(axis=0) + a.min(axis=0)) / 2
    ax.set_xlim(mid[0] - span.max() * 0.55, mid[0] + span.max() * 0.55)
    ax.set_ylim(mid[1] - span.max() * 0.55, mid[1] + span.max() * 0.55)
    ax.set_zlim(mid[2] - span.max() * 0.55, mid[2] + span.max() * 0.55)
    # Adjust margins, save the numbered task PNG, and release the figure.
    fig.tight_layout()
    fig.savefig(OUT / f'{i+1:02d}_{e["task"]}.png', dpi=155)
    plt.close(fig)

# Project all tasks onto X-Z panels; Y information is omitted.
# The 4-by-4 layout supports at most 16 segments; axs.flat indexes panels sequentially.
fig, axs = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)
for i, (task, rawpath, mp, rr, sg) in enumerate(traces):
    ax = axs.flat[i]
    ax.plot(rawpath[:, 0], rawpath[:, 2], color="#555", label="Recorded TCP", lw=1.3)
    ax.plot(mp[:, 0], mp[:, 2], "-o", color="#e4572e", label="FM", ms=3)
    ax.plot(rr[:, 0], rr[:, 2], "--x", color="#2a9d8f", label="Target", ms=3)
    ax.plot(sg[:, 0], sg[:, 2], ":", color="#78909c", label="Straight", lw=1)
    ax.set_title(task.replace("_", " "), fontsize=9)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
# Hide unused panels and reuse the first panel legend as the shared legend.
for j in range(len(traces), 16):
    axs.flat[j].axis("off")
handles, labels = axs.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.09), ncol=2)
fig.suptitle(
    "Held-out demonstration (CSV #3) — XYZ waypoint predictions, XZ view", fontsize=16
)
fig.savefig(OUT / "all_segments_xz.png", dpi=170)
plt.close(fig)

# Compare model and straight-line errors with paired bars for each task.
# Offsets of +/-0.2 separate the bars; each bar has width 0.4.
fig, ax = plt.subplots(figsize=(12, 6))
names = [r["task"].replace("_", " ") for r in results]
xx = np.arange(len(names))
ax.bar(xx - 0.2, [r["model_error_mm"] for r in results], 0.4, label="Flow Matching")
ax.bar(
    xx + 0.2,
    [r["straight_baseline_error_mm"] for r in results],
    0.4,
    label="Straight-line baseline",
)
ax.set_xticks(xx, names, rotation=50, ha="right")
ax.set_ylabel("Mean target waypoint distance (mm)")
ax.set_title("Held-out CSV: Flow Matching vs straight line")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "error_comparison.png", dpi=170)
plt.close(fig)

# Interactive 3D all traces with menu showing one task at a time
# Build a rotatable, zoomable Plotly figure.
# Each task contributes five traces: four paths and one endpoint marker trace.
fig = go.Figure()
ntr = 5
for i, (task, rawpath, mp, rr, sg) in enumerate(traces):
    for vals, name, color, dash, markers in [
        (rawpath, "Recorded TCP", "#555555", "solid", False),
        (mp, "Flow Matching", "#e4572e", "solid", True),
        (rr, "Interpolated target", "#2a9d8f", "dash", True),
        (sg, "Straight baseline", "#7d8b99", "dot", False),
    ]:
        # Columns 0/1/2 provide X/Y/Z. Show markers only where requested.
        # Initially display only the fourth task (i == 3).
        fig.add_trace(
            go.Scatter3d(
                x=vals[:, 0],
                y=vals[:, 1],
                z=vals[:, 2],
                mode="lines+markers" if markers else "lines",
                name=name,
                visible=i == 3,
                line=dict(
                    color=color, width=7 if name == "Flow Matching" else 4, dash=dash
                ),
                marker=dict(size=4),
            )
        )
    # Add labeled Start/Goal markers as the fifth trace for this task.
    fig.add_trace(
        go.Scatter3d(
            x=[mp[0, 0], mp[-1, 0]],
            y=[mp[0, 1], mp[-1, 1]],
            z=[mp[0, 2], mp[-1, 2]],
            mode="markers+text",
            text=["Start", "Goal"],
            textposition="top center",
            marker=dict(color=["green", "blue"], size=6),
            name="Start/Goal",
            visible=i == 3,
        )
    )
# Create a dropdown that shows only the five traces belonging to the chosen task.
buttons = []
for i, (task, *_) in enumerate(traces):
    # Integer division by ntr maps each trace index to its task index.
    visible = [j // ntr == i for j in range(len(traces) * ntr)]
    # Update both trace visibility and the figure title when a task is selected.
    buttons.append(
        dict(
            label=task,
            method="update",
            args=[
                {"visible": visible},
                {"title": f"Flow Matching vs held-out CSV: {task}"},
            ],
        )
    )
# Use millimeter axes with proportions based on the data ranges.
# The initial title assumes at least four evaluation segments.
fig.update_layout(
    title=f"Flow Matching vs held-out CSV: {traces[3][0]}",
    scene=dict(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
        aspectmode="data",
    ),
    updatemenus=[dict(buttons=buttons, x=0, y=1.12, xanchor="left", yanchor="top")],
    margin=dict(t=110, l=0, r=0, b=0),
    legend=dict(orientation="h", y=-0.05),
)
# Embed Plotly JavaScript in the HTML so viewing does not require a CDN connection.
fig.write_html(OUT / "interactive_3d.html", include_plotlyjs=True)
# Save per-task metrics and overall means as JSON.
# Means use rounded per-task errors, so final digits may differ from report.json.
# No inverse kinematics or collision validation is performed.
(OUT / "visual_evaluation.json").write_text(
    json.dumps(
        {
            "heldout_csv": examples[0]["csv"],
            "results": results,
            "mean_fm_mm": float(np.mean([r["model_error_mm"] for r in results])),
            "mean_straight_baseline_mm": float(
                np.mean([r["straight_baseline_error_mm"] for r in results])
            ),
        },
        indent=2,
    )
)
# Print the output directory and both mean waypoint errors.
print(
    "Saved",
    OUT,
    "FM",
    np.mean([r["model_error_mm"] for r in results]),
    "straight",
    np.mean([r["straight_baseline_error_mm"] for r in results]),
)
