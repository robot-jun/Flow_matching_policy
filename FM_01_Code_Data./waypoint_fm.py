"""Machine-tending sparse XYZ waypoint conditional flow matching.
Uses only uploaded demonstration CSVs and URDF joint-limit metadata.
CSV EE position is in mm, URDF length in meters. Predictions are NOT robot-safe by default.
"""

import argparse, glob, json, random, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

XYZ = ["EE_x", "EE_y", "EE_z"]
JOINTS = [f"joint{i}_angle" for i in range(1, 7)]

# Resolve the default training output relative to this source file, regardless of the working directory.
# An explicit --out argument overrides this location.
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "FM_02_Model_Results." / "output"
)


# Seed Python, NumPy, and PyTorch to improve reproducibility in the same environment.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Read joint1 through joint6 limits as URDF metadata.
# These limits are not enforced during learning or inference; no IK or collision checking is performed.
def read_urdf(path):
    root = ET.parse(path).getroot()
    limits = {}
    for j in root.findall("joint"):
        if j.attrib.get("name") in [f"joint{i}" for i in range(1, 7)]:
            x = j.find("limit")
            limits[j.attrib["name"]] = {
                "lower_rad": float(x.attrib["lower"]),
                "upper_rad": float(x.attrib["upper"]),
            }
    if len(limits) != 6:
        raise ValueError("Expected all six robot revolute joints in URDF")
    return {
        "robot": root.attrib.get("name"),
        "joint_limits": limits,
        "tcp_link": "tool_center_point",
        "collision_validation": "NOT IMPLEMENTED",
    }


# Build fixed-size training examples from exactly three demonstration CSVs.
# Return examples, task names, robot metadata, and per-file diagnostics.
def load_examples(folder, urdf, k=4):
    # Sorted file order defines file_idx 0, 1, and 2, used for the train/test split.
    files = sorted(glob.glob(str(Path(folder) / "*.csv")))
    if len(files) != 3:
        raise ValueError(f"Expected exactly 3 CSVs in {folder}, found {len(files)}")
    info = read_urdf(urdf)
    items = []
    tasks = set()
    diagnostics = []
    for fi, f in enumerate(files):
        d = pd.read_csv(f)
        # Require position, task, waypoint, and joint columns; reject non-finite numeric values.
        # Joint angles are validated but are not included in the model condition.
        for x in XYZ + ["current_task", "waypoint"] + JOINTS:
            if x not in d:
                raise ValueError(f"Missing {x} in {f}")
        if not np.isfinite(d[XYZ + JOINTS].to_numpy(float)).all():
            raise ValueError(f"NaN / inf in positions/joints: {f}")
        # Assign a new group whenever the task name changes between consecutive rows.
        groups = d.current_task.ne(d.current_task.shift()).cumsum()
        for _, g in d.groupby(groups, sort=False):
            task = str(g.current_task.iloc[0])
            tasks.add(task)
            # Skip segments with fewer than three recorded samples.
            if len(g) < 3:
                continue
            # Convert CSV positions from millimeters to meters.
            # The movement filter below uses endpoint distance, not total path length; loops may also be excluded.
            start = g[XYZ].iloc[0].to_numpy(float) / 1000.0
            end = g[XYZ].iloc[-1].to_numpy(float) / 1000.0
            if np.linalg.norm(end - start) < 0.015:
                continue  # stationary operation, not transit
            # Marked points plus end anchors, interpolate at equally-spaced normalized indices.
            # Locate waypoint markers after trimming whitespace and normalizing capitalization.
            hits = np.flatnonzero(
                g.waypoint.astype(str).str.strip().str.lower().eq("waypoint").to_numpy()
            )
            if len(hits) == 0:
                continue  # unmarked segments cannot supervise sparse landmarks
            # Combine marked rows with both endpoint anchors and remove duplicate indices.
            # If markers occur only at the endpoints, the resulting reference may be a straight line.
            x_idx = np.unique(np.r_[0, hits, len(g) - 1])
            xy = g[XYZ].iloc[x_idx].to_numpy(float) / 1000.0
            # Normalize row indices to [0, 1], rather than using timestamps or arc length.
            # For k=4, sample the interior at 0.2, 0.4, 0.6, and 0.8.
            t = x_idx / (len(g) - 1)
            sample = np.linspace(0, 1, k + 2)[1:-1]
            # Interpolate each XYZ axis independently into a (k, 3) reference array.
            # These labels come from sparse markers, not every point of the recorded trajectory.
            target = np.column_stack([np.interp(sample, t, xy[:, i]) for i in range(3)])
            # Task identity, current start joint angles (URDF degrees vs radians unclear):
            # use XYZ start / goal, and task only, to avoid implicit unit assumptions.
            # Construct the straight-line baseline at the same sample locations.
            # Learn deviations from this baseline instead of absolute waypoint coordinates.
            base = start[None, :] + sample[:, None] * (end - start)[None, :]
            # Flatten (k, 3) residuals to 3*k features: 12 features for four interior points.
            residual = (target - base).reshape(-1)
            # Store identifiers, meter-based geometry, residual labels, and marker counts.
            items.append(
                {
                    "file_idx": fi,
                    "csv": Path(f).name,
                    "task": task,
                    "start_m": start.tolist(),
                    "goal_m": end.tolist(),
                    "target_m": target.tolist(),
                    "residual_m": residual.tolist(),
                    "raw_mark_count": int(len(hits)),
                    "rows": int(len(g)),
                }
            )
        diagnostics.append({"csv": Path(f).name, "rows": len(d)})
    # Build the task vocabulary from usable segments across all CSVs.
    # Only training CSVs contribute to weight updates and normalization statistics.
    task_names = sorted({i["task"] for i in items})
    if not items:
        raise ValueError("No nonstationary, waypoint-marked segments to learn")
    return items, task_names, info, diagnostics


# Create a condition vector with 6 + number_of_tasks features.
# It contains absolute start XYZ, goal-minus-start XYZ, and a one-hot task identity.
def condition(e, task_names):
    # normalize all geometry to meters and centered on start; absolute start retained.
    a = np.asarray(e["start_m"], dtype=np.float32)
    b = np.asarray(e["goal_m"], dtype=np.float32)
    # Encode the selected task as 1 and all other task categories as 0.
    hot = np.zeros(len(task_names), np.float32)
    hot[task_names.index(e["task"])] = 1
    return np.r_[a, b - a, hot].astype(np.float32)


# Predict a conditional velocity field v(x, t, c) using a multilayer perceptron.
# x is the residual-space state, t is generation time, and c is the task/geometry condition.
# This velocity describes the generative process, not physical robot motion.
class Flow(nn.Module):
    def __init__(self, dim, cond_dim, width=128):
        # Concatenate residual features, condition features, and one time feature.
        # Hidden layers use SiLU; the output has the same dimension as the residual state.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, dim),
        )

    def forward(self, x, t, c):
        # Concatenate features along the last axis while preserving the batch dimension.
        return self.net(torch.cat([x, t, c], dim=-1))


# Select examples by CSV index and assemble condition C and target residual Y.
# For N examples, their shapes are (N, 6 + task_count) and (N, 3*k).
def make_arrays(items, task_names, ids):
    selected = [e for e in items if e["file_idx"] in ids]
    return (
        selected,
        np.stack([condition(e, task_names) for e in selected]),
        np.asarray([e["residual_m"] for e in selected], np.float32),
    )


# Train on two demonstrations, evaluate on the held-out one, and save model artifacts.
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    items, tasks, urdf, diag = load_examples(args.data_dir, args.urdf, args.k)
    # Exclude the held-out CSV entirely; do not randomly split correlated rows within a demonstration.
    train_examples, C, Y = make_arrays(
        items, tasks, [i for i in range(3) if i != args.test_csv]
    )
    # Prepare held-out examples. Ct and Yt are not directly used below.
    test_examples, Ct, Yt = make_arrays(items, tasks, [args.test_csv])
    # Task generalization is not evaluated: if a task occurs only in held-out demonstration, skip eval that task.
    # Evaluate only tasks present in training; this does not test unseen task categories.
    known = {e["task"] for e in train_examples}
    eval_examples = [e for e in test_examples if e["task"] in known]
    if len(eval_examples) != len(test_examples):
        print("Warning: skipped unseen task(s) in evaluation")
    # Compute statistics from training conditions only.
    # Floor the standard deviation at 0.05 to avoid amplifying nearly constant features.
    cond_mean = C.mean(0)
    cond_std = C.std(0)
    cond_std = np.maximum(cond_std, 0.05)
    # Do not center one-hot features: preserve explicit categorical features.
    # Preserve categorical one-hot features as their original 0/1 values.
    cond_mean[6:] = 0.0
    cond_std[6:] = 1.0
    # Scale each residual feature by its training standard deviation, with a 0.10 m floor.
    # Do not subtract its mean; inference restores the scale by multiplication.
    yscale = np.maximum(Y.std(0), 0.10)  # per-position residual scaling, in meters
    yn = Y / yscale[None, :]
    cn = (C - cond_mean) / cond_std
    net = Flow(Y.shape[1], C.shape[1], args.width).to(device)
    # Use AdamW with the requested learning rate and weight decay.
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    x = torch.tensor(yn, device=device)
    c = torch.tensor(cn, device=device)
    # Each step is one optimizer update, not a full pass through the dataset (epoch).
    for step in range(1, args.steps + 1):
        # Sample a minibatch with replacement; an example may appear multiple times.
        ix = torch.randint(len(x), (args.batch,), device=device)
        # x1 is the normalized target residual; z is standard Gaussian noise of matching shape.
        # Sample a separate t in [0, 1) for each example; t is not robot execution time.
        x1 = x[ix]
        z = torch.randn_like(x1)
        t = torch.rand(len(ix), 1, device=device)
        # Flow Matching: interpolate xt = (1-t)*z + t*x1.
        # Its derivative with respect to t is x1-z, which provides the target velocity.
        xt = (1 - t) * z + t * x1
        velocity = x1 - z
        # Minimize mean squared error between predicted and target velocities.
        # Clear old gradients, backpropagate, then update network weights.
        loss = ((net(xt, t, c[ix]) - velocity) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, args.steps // 5) == 0:
            print(f"step {step:5d}/{args.steps} train FM loss={loss.item():.5f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.pt"
    # Save weights and all configuration needed to reconstruct the model and coordinate transforms.
    torch.save(
        {
            "model": net.state_dict(),
            "tasks": tasks,
            "k": args.k,
            "width": args.width,
            "cond_mean": cond_mean.tolist(),
            "cond_std": cond_std.tolist(),
            "yscale": yscale.tolist(),
            "urdf": urdf,
            "train_csv_ids": [i for i in range(3) if i != args.test_csv],
            "test_csv_id": args.test_csv,
            "seed": args.seed,
        },
        path,
    )
    # Record dataset counts, task names, evaluation metrics, and interpretation limits.
    report = {
        "data_diagnostics": diag,
        "train_segments": len(train_examples),
        "test_segments": len(test_examples),
        "evaluated_segments": len(eval_examples),
        "task_names": tasks,
        "train_raw_mark_counts": [e["raw_mark_count"] for e in train_examples],
        "notes": [
            "K internal points are interpolated from sparse CSV waypoint flags, not direct ground-truth collision-free targets.",
            "Evaluation is held-out demonstration prediction vs interpolated labels, not collision/success rate.",
            "Object/grasp pose is assumed encoded indirectly by start/goal and task; no object/grasp pose columns exist in CSV.",
            "URDF supplies robot joint limits, but does not establish CSV-to-URDF EE frame alignment. No IK or collision check implemented.",
        ],
    }
    # Generate waypoints for held-out examples without updating model weights.
    if eval_examples:
        errors = []
        for e in eval_examples:
            pred = sample_waypoints(
                net,
                condition(e, tasks),
                e["start_m"],
                e["goal_m"],
                cond_mean,
                cond_std,
                yscale,
                args.k,
                args.ode_steps,
                args.seed,
                device,
            )
            # Average XYZ Euclidean error across interior points, then convert meters to millimeters.
            # This measures interpolated-label agreement, not collision rate or task success.
            errors.append(
                float(
                    np.linalg.norm(pred - np.array(e["target_m"]), axis=1).mean() * 1000
                )
            )
        report["heldout_mean_waypoint_error_mm"] = float(np.mean(errors))
        report["heldout_per_segment_error_mm"] = [
            {"task": e["task"], "error_mm": round(er, 2)}
            for e, er in zip(eval_examples, errors)
        ]
    # Write the evaluation report and held-out examples alongside model.pt.
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "test_examples.json").write_text(
        json.dumps(test_examples, indent=2, ensure_ascii=False)
    )
    print("Model saved:", path, "\nReport:", out / "report.json")
    print(
        "Heldout mean waypoint error [mm]:",
        report.get("heldout_mean_waypoint_error_mm"),
    )


# Generate residuals from noise, then add them to the straight-line baseline.
# Inputs and returned (k, 3) positions are in meters; endpoints are not included.
def sample_waypoints(
    net, raw_cond, start, goal, mean, std, yscale, k, steps, seed, device
):
    # Explicit Euler integration of learned conditional velocity field t=0 .. t=1.
    # Use a local random generator so the seed controls the initial inference noise.
    generator = torch.Generator(device=device).manual_seed(int(seed))
    # Start with a single Gaussian sample of shape (1, 3*k).
    z = torch.randn((1, 3 * k), device=device, generator=generator)
    # Apply saved training normalization and add a batch dimension to the condition.
    cc = torch.tensor(
        ((raw_cond - mean) / std)[None, :], dtype=torch.float32, device=device
    )
    net.eval()
    # Disable gradient tracking during inference.
    with torch.no_grad():
        # Integrate generation time from 0 toward 1 using a fixed step size dt=1/steps.
        for i in range(steps):
            tt = torch.full((1, 1), i / steps, device=device)
            # Explicit Euler update: state += predicted_velocity * dt.
            # The condition stays fixed while the residual state evolves.
            z = z + net(z, tt, cc) / steps
    # Reshape the generated vector to (k, 3) and restore residual units in meters.
    delta = z.cpu().numpy().reshape(k, 3) * yscale.reshape(k, 3)
    # Reconstruct the same straight-line interior locations used during training.
    base = (
        np.asarray(start)[None, :]
        + np.linspace(0, 1, k + 2)[1:-1, None]
        * (np.asarray(goal) - np.asarray(start))[None, :]
    )
    # Return baseline plus generated offsets; no feasibility or collision validation follows.
    return base + delta


# Load a checkpoint and generate waypoints from a task and user-provided endpoints in mm.
def predict(args):
    device = torch.device(args.device)
    # Load full checkpoint metadata; weights_only=False requires a trusted model file.
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    # Restore saved configuration and convert input coordinates from mm to m.
    k = ckpt["k"]
    tasks = ckpt["tasks"]
    start = np.asarray(args.start_mm, float) / 1000
    goal = np.asarray(args.goal_mm, float) / 1000
    # Reject task names absent from the saved vocabulary.
    if args.task not in tasks:
        raise ValueError(f"Task not in trained labels: {tasks}")
    # Recreate the network dimensions before loading its trained weights.
    model = Flow(3 * k, 6 + len(tasks), ckpt["width"]).to(device)
    model.load_state_dict(ckpt["model"])
    e = {"start_m": start, "goal_m": goal, "task": args.task}
    pred = (
        sample_waypoints(
            model,
            condition(e, tasks),
            start,
            goal,
            np.asarray(ckpt["cond_mean"]),
            np.asarray(ckpt["cond_std"]),
            np.asarray(ckpt["yscale"]),
            k,
            args.ode_steps,
            args.seed,
            device,
        )
        * 1000
    )
    # Serialize output coordinates in mm with a reminder that execution has not been validated.
    result = {
        "task": args.task,
        "start_xyz_mm": list(np.round(start * 1000, 3)),
        "goal_xyz_mm": list(np.round(goal * 1000, 3)),
        "intermediate_waypoints_xyz_mm": pred.round(3).tolist(),
        "warning": "Unvalidated XYZ proposals. Check TCP frame, IK, and full-path collisions before robot execution.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))


# Reuse predict with the start and goal of a matching held-out example.
def demo(args):
    # Select the first example matching the requested task.
    samples = json.loads(Path(args.samples).read_text())
    e = next((x for x in samples if x["task"] == args.task), None)
    if e is None:
        raise ValueError(f"No held-out example for task {args.task}")
    import types

    # Copy command-line settings and attach the selected endpoints converted to mm.
    a = types.SimpleNamespace(**vars(args))
    a.start_mm = (np.array(e["start_m"]) * 1000).tolist()
    a.goal_mm = (np.array(e["goal_m"]) * 1000).tolist()
    predict(a)


# Dispatch the train, predict, or demo command.
# Explicit relative paths are resolved from the terminal working directory.
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    # Training inputs: demonstration folder, required URDF file, and output location.
    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", default="data")
    tr.add_argument("--urdf", required=True)
    tr.add_argument("--out", default=str(DEFAULT_OUTPUT))
    # k counts interior waypoints; test-csv selects the held-out file by sorted index (default: third file).
    tr.add_argument("--k", type=int, default=4)
    tr.add_argument("--test-csv", type=int, choices=[0, 1, 2], default=2)
    # steps counts optimizer updates, batch sets minibatch size, and lr sets the learning rate.
    tr.add_argument("--steps", type=int, default=1500)
    tr.add_argument("--batch", type=int, default=32)
    tr.add_argument("--lr", type=float, default=0.001)
    # width sets hidden-layer size; ode-steps sets the integration resolution for generation.
    tr.add_argument("--width", type=int, default=128)
    tr.add_argument("--ode-steps", type=int, default=80)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--device", default="cpu")
    # Direct prediction requires a saved model, a known task, and start/goal XYZ.
    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True)
    pr.add_argument("--task", required=True)
    # Each endpoint accepts three numbers in X Y Z order, measured in millimeters.
    pr.add_argument("--start-mm", type=float, nargs=3, required=True)
    pr.add_argument("--goal-mm", type=float, nargs=3, required=True)
    pr.add_argument("--ode-steps", type=int, default=80)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--output")
    # Demo mode reads endpoint coordinates from the saved evaluation examples.
    dm = sub.add_parser("demo")
    dm.add_argument("--model", required=True)
    dm.add_argument("--samples", required=True)
    dm.add_argument("--task", default="retreat_feeder_to_approach_machine")
    dm.add_argument("--ode-steps", type=int, default=80)
    dm.add_argument("--seed", type=int, default=42)
    dm.add_argument("--device", default="cpu")
    dm.add_argument("--output")
    # Parse arguments and invoke the function associated with the requested mode.
    args = p.parse_args()
    {"train": train, "predict": predict, "demo": demo}[args.mode](args)


# Run the command-line entry point only on direct execution, not when imported by visualize.py.
if __name__ == "__main__":
    main()
