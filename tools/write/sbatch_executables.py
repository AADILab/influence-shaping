"""Write sbatch executable files for configs nested in the specified top directory
Example Usage:
python tools/write_sbatch_executables.py ~/influence-shaping/results/10_29_2024/alpha ~/influence-shaping/sbatch/10_29_2024/alpha
"""

import argparse
from influence.sbatch import write_sbatch_executables_cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="write_sbatch_executables.py",
        description="Write sbatch executable files for configs nested in the specified top directory",
        epilog=""
    )
    parser.add_argument("config_directory", help="top directory of configs that need to be batched")
    parser.add_argument(
        'sbatch_directory',
        help='directory to write sbatch files to',
        type=str,
        nargs='?',
        default=None
    )
    parser.add_argument("--seperate_trials", help="flag to treat each trial as a seperate job", action='store_true')
    args = parser.parse_args()

    write_sbatch_executables_cli(args.config_directory, args.sbatch_directory, args.seperate_trials)
