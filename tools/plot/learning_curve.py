'''Give this python script a fitness.csv file and it will plot the learning curve'''

import argparse
from pathlib import Path
from influence.plotting import plot_learning_curve

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='learning_curve.py',
        description='plot the learning curve from the specified fitness.csv file',
        epilog=''
    )
    parser.add_argument(
        'fitness_dir', 
        help='directory of csv file containing fitnesses',
        type=str
    )
    parser.add_argument(
        '-s', '--silent',
        help='run silently, without showing the plot',
        action='store_true',
    )
    parser.add_argument(
        '-o', '--output', 
        help='directory to output image of plot to',
        type=str
    )
    parser.add_argument(
        '--title',
        help='title of generated plot',
        type=str
    )
    parser.add_argument(
        '--individual_agents',
        help="include individual agents' shaped rewards",
        action='store_true'
    )
    args = parser.parse_args()

    if args.output is not None:
        args.output = Path(args.output)

    plot_learning_curve(Path(args.fitness_dir), args.output, args.silent, args.title, args.individual_agents)
