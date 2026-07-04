"""Clamp out-of-bounds agent positions in a joint trajectory CSV.

Usage:
    python fix_bounds.py <path/to/team_N_joint_traj.csv>

The script looks for config.yaml four directories above the CSV file
(CSV -> test/ -> gen_N/ -> trial_N/ -> <variant>/config.yaml).
Per-agent bounds are read from the config; agents with no bounds entry
are constrained to the map extents. POI columns and obs/dx/dy columns
are passed through unchanged.

Output is written as <stem>_corrected.csv beside the input file.
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(csv_path: Path) -> dict:
    config_path = csv_path.parent.parent.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at expected path: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def map_bounds(config: dict) -> tuple[float, float, float, float]:
    w, h = config["env"]["map_size"]
    return 0.0, 0.0, float(w), float(h)


def agent_bounds(agent_cfg: dict, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if "bounds" in agent_cfg:
        b = agent_cfg["bounds"]
        return float(b["low_x"]), float(b["low_y"]), float(b["high_x"]), float(b["high_y"])
    return fallback


def fix_bounds(csv_path: Path) -> Path:
    config = load_config(csv_path)
    fallback = map_bounds(config)

    rovers_cfg = config["env"]["agents"].get("rovers", [])
    uavs_cfg = config["env"]["agents"].get("uavs", [])

    df = pd.read_csv(csv_path)

    corrections = 0
    for agent_type, cfgs in [("rover", rovers_cfg), ("uav", uavs_cfg)]:
        for idx, cfg in enumerate(cfgs):
            lx, ly, hx, hy = agent_bounds(cfg, fallback)
            x_col = f"{agent_type}_{idx}_x"
            y_col = f"{agent_type}_{idx}_y"
            if x_col in df.columns:
                before = df[x_col].copy()
                df[x_col] = df[x_col].clip(lx, hx)
                corrections += (df[x_col] != before).sum()
            if y_col in df.columns:
                before = df[y_col].copy()
                df[y_col] = df[y_col].clip(ly, hy)
                corrections += (df[y_col] != before).sum()

    out_path = csv_path.with_stem(csv_path.stem + "_corrected")
    df.to_csv(out_path, index=False)
    print(f"Corrected {corrections} out-of-bounds values. Written to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fix_bounds.py",
        description="Clamp out-of-bounds agent positions in a joint trajectory CSV",
    )
    parser.add_argument(
        "csv",
        help="Path to joint trajectory CSV (e.g. team_0_joint_traj.csv)",
        type=str,
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    fix_bounds(csv_path)
