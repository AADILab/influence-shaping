"""Overwrite the parameters of configs inside a given directory using a specified overwriting config
Example Usage:
python tools/overwrite_configs.py results/10_29_2024/echo example/overwrite/echo.yaml
"""

import argparse
from influence.config import overwrite_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="overwrite_configs.py",
        description="Overwrite parameters in specified configs according to new parameters in overwriting config",
        epilog=""
    )
    parser.add_argument("configs_directory", help="top directory of configs that need to be overwritten")
    parser.add_argument("overwrite_directory", help="config with overwriting parameters specified")
    args = parser.parse_args()

    overwrite_configs(args.configs_directory, args.overwrite_directory)
