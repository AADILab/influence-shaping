'''Give this python script a fitness.csv file and it will plot the learning curve'''

from pathlib import Path
from influence.plotting import plot_learning_curve
from influence.parsing import LinePlotParser

if __name__ == '__main__':
    parser = LinePlotParser(
        prog='learning_curve.py',
        description='plot the learning curve from the specified fitness.csv file',
        epilog=''
    )
    parser.add_plot_args()
    parser.add_argument(
        'fitness_dir',
        help='directory of csv file containing fitnesses',
        type=str
    )
    parser.add_argument(
        '--individual-agents',
        help="include individual agents' shaped rewards",
        action='store_true'
    )
    args = parser.parse_args()

    plot_learning_curve(Path(args.fitness_dir), args.individual_agents, parser.dump_line_plot_args(args), parser.dump_plot_args(args))
