"""Write configs according to a parameter sweep into a corresponding directory tree
Example Usage:
python tools/write_config_tree.py tree_gen/10_29_2024/alpha.yaml results/10_29_2024/alpha
"""

import argparse
from influence.config import write_config_tree_cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="write_config_tree.py",
        description="Write configs according to a parameter sweep into a corresponding directory tree",
        epilog=""
    )
    parser.add_argument("config_directory", help="Directory of yaml file with sweep parameters")
    parser.add_argument(
        'top_write_directory',
        help='Top directory to write configs to',
        type=str,
        nargs='?',
        default=None
    )
    args = parser.parse_args()

    write_config_tree_cli(args.config_directory, args.top_write_directory)
