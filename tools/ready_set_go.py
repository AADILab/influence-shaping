# This is meant to combine generating config files, writing sbatch bash files, and sending jobs to slurm

import argparse
import subprocess
from influence.config import write_config_tree_cli
from influence.sbatch import write_sbatch_executables_cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ready_set_go.py",
        description="Write configs, write sbatch files, and send jobs to slurm",
        epilog=""
    )
    parser.add_argument("config_directory", help="Directory of yaml file with sweep parameters")
    parser.add_argument('--time', help='time limit for job formatted as D-HH:MM:SS', default='2-00:00:00')
    parser.add_argument("--seperate-trials", help="flag to treat each trial as a seperate job", action='store_true')
    parser.add_argument('--cnv', help='flag to only request nodes cn-v-[1-9]', action='store_true')
    args = parser.parse_args()

    top_write_dir = write_config_tree_cli(args.config_directory, None)
    sbatch_exec = write_sbatch_executables_cli(top_write_dir, None, args.time, args.seperate_trials, args.cnv)

    print(f"Running {sbatch_exec}...")
    try:
        result = subprocess.run([str(sbatch_exec)], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running sbatch file: {e}")
        print(e.stderr)
