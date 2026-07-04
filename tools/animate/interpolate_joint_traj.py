'''Smooth a joint trajectory CSV by adding frames via a smoothing spline.

Rover and UAV x/y positions are fitted with a cubic smoothing spline
(UnivariateSpline, k=3). The smoothing factor s controls how closely the
spline follows the original points: s=0 passes through every point exactly;
larger s allows more deviation for a smoother curve. The default (s=None)
lets scipy choose based on data length, which is already quite smooth.

All other columns (POI coords, observations, dx/dy) are linearly interpolated.
The output CSV has (N-1)*steps_per_frame + 1 rows for an N-row input.
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import UnivariateSpline, interp1d


def interpolate_trajectory(input_path: Path, steps_per_frame: int, output_path: Path,
                           smoothing: float | None):
    df = pd.read_csv(input_path)
    n_rows = len(df)
    if n_rows < 4:
        raise ValueError(f"Need at least 4 rows for cubic smoothing spline, got {n_rows}")

    t_orig = np.arange(n_rows, dtype=float)
    t_new = np.linspace(0.0, n_rows - 1.0, (n_rows - 1) * steps_per_frame + 1)

    pos_cols = [
        c for c in df.columns
        if (c.startswith('rover_') or c.startswith('uav_'))
        and (c.endswith('_x') or c.endswith('_y'))
    ]
    other_cols = [c for c in df.columns if c not in pos_cols]

    result = {}
    for col in pos_cols:
        spline = UnivariateSpline(t_orig, df[col].values, k=3, s=smoothing)
        result[col] = spline(t_new)
    for col in other_cols:
        result[col] = interp1d(t_orig, df[col].values, kind='linear')(t_new)

    out_df = pd.DataFrame({col: result[col] for col in df.columns})
    out_df.to_csv(output_path, index=False)
    print(f"{n_rows} frames -> {len(out_df)} frames written to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='interpolate_joint_traj.py',
        description='Smooth a joint trajectory CSV by adding interpolated frames',
    )
    parser.add_argument(
        'input',
        help='Path to input joint trajectory CSV',
        type=str
    )
    parser.add_argument(
        '--steps-per-frame',
        help='Interpolated sub-steps between each original frame (default: 4)',
        type=int,
        default=4
    )
    parser.add_argument(
        '--smoothing',
        help=(
            'Smoothing factor s for UnivariateSpline. s=0 interpolates exactly through '
            'every point; larger values allow more deviation for a smoother result. '
            'Omit to let scipy choose automatically (recommended starting point).'
        ),
        type=float,
        default=None
    )
    parser.add_argument(
        '--output',
        help='Output CSV path (default: <input stem>_interpolated.csv beside the input)',
        type=str,
        default=None
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.parent / (input_path.stem + '_interpolated.csv')
    else:
        output_path = Path(args.output)

    interpolate_trajectory(input_path, args.steps_per_frame, output_path, args.smoothing)
