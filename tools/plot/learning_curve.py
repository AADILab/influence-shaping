'''Give this python script a fitness.csv file and it will plot the learning curve'''

from pathlib import Path
from influence.plotting import plot_learning_curve
from influence.parsing import PlotParser

if __name__ == '__main__':
    parser = PlotParser(
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
        '--individual_agents',
        help="include individual agents' shaped rewards",
        action='store_true'
    )
    parser.add_argument(
        '--window_size',
        help='window size for moving average filter on final plot',
        type=int
    )
    parser.add_argument(
        '--downsample',
        help='downsample and only plot one point every _ points',
        type=int,
        default=1
    )
    args = parser.parse_args()

    plot_learning_curve(Path(args.fitness_dir), args.individual_agents, args.window_size, args.downsample, parser.dump_plot_args(args))
