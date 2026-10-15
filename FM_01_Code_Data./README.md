# Machine tending: sparse XYZ waypoint Conditional Flow Matching

This is a research prototype based on the 1 provided URDF and 3 provided demonstration CSVs. It learns **four interior XYZ locations** conditioned on the task, start XYZ, and goal XYZ. **Not a motion planner, not a robot-safe controller.**

## Setup
```bash
pip install -r requirements.txt
python waypoint_fm.py train --data-dir data --urdf 'data/M_3D_R_AA_0025(6).urdf' --out output --steps 1500 --test-csv 2
python waypoint_fm.py demo --model output/model.pt --samples output/test_examples.json --task retreat_feeder_to_approach_machine --output output/generated.json
```

To infer for a novel start/goal in CSV **millimeters**:
```bash
python waypoint_fm.py predict --model output/model.pt --task retreat_feeder_to_approach_machine --start-mm -640 -680 1300 --goal-mm 1140 -28 1115 --output output/custom.json
```

The example coordinates above are illustrative, not guaranteed feasible or scene-safe. Task name must be seen in training.

## Ground truth and transforms

Each `current_task` run is segmented. Only moving task runs with at least one explicit `waypoint` marker are supervised. Start/end are first/last row XYZ of the contiguous task run. Marked points are combined with start/end and linearly resampled at `K` normalized **time indices**, not arclength, yielding fixed-size targets even when waypoint counts differ. Learning labels are **interpolated waypoints**, not necessarily true hand-selected original waypoints. An endpoint-anchored straight line is used as baseline; model learns relative XYZ residuals. XYZ CSV values are divided by 1000 from millimeters to meters for network computation; outputs convert back to millimeters.

`URDF` is parsed for six robot revolute-joint limits. It is **not** used for IK or collision detection; outside meshes/frames may be unresolved. CSV XYZ EE frame alignment to URDF world frame is unverified. Grasp and object positions are NOT provided as separate columns, so here they are only *implicitly* represented by task, start and goal. If your machine layout, grasp point, or obstacle position changes, this model lacks explicit conditioning on those changes.

The held-out CSV (default sorted index 2, meaning 003) is **never used for training**. The test error is XYZ deviation from targets interpolated from the held-out marked points; it does not measure safe execution or task success. Each sample within the same demonstration remains correlated; no claims of generalization from only 3 runs.

## Next upgrades

1. Add explicit object/grasp poses and EE orientation with consistent reference frames, e.g. `object_xyz_m`, `grasp_xyz_m` and an approach direction; append to conditioning.
2. Add a robotics geometry stack (Pinocchio / PyBullet / MoveIt2) and mesh asset resolution to check IK for each TCP 6D pose; collision-check densely interpolated segments and joint trajectory including scene obstacles.
3. Gather more independent demonstrations and perturb starts/goals/obstacle configurations; test held-out scenes, not just held-out repeat trajectories.
4. For genuinely **sparse landmark** output rather than resampled points, represent cardinality/landmark mask separately and train across multiple labels per task.
