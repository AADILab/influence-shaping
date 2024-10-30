"""Give this python script a config file and it will run the CCEA using the specified config"""

import argparse
from influence.ccea_lib import runCCEA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="config.py",
        description="This runs a CCEA according to the given configuration file",
        epilog=""
    )
    parser.add_argument("config_dir")
    parser.add_argument(
        '-t', '--num_trial',         # The flags for the optional argument
        type=int,                 # Specify the type of the argument (optional)
        help='Number of trial to run. Defaults to running all trials if none is specified.',  # Description of the argument
        default=None   # Provide a default value (optional)
    )
    parser.add_argument(
        '--load_checkpoint',
        help='Look for a checkpoint and load it in if one is found',
        action='store_true'
    )
    args = parser.parse_args()

    runCCEA(args.config_dir, args.num_trial, args.load_checkpoint)
