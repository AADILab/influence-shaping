"""This plots the learning curves for paper figures"""

# Result 1 is in yabby 27 square_10x10
# This is the easy version of the environment

from pathlib import Path
import os
from influence.plotting import plot_comparison, generate_comparison_plot
from influence.parsing import LinePlotArgs, PlotArgs
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from matplotlib import font_manager

comparison_dir = Path(os.path.expanduser('~/influence-shaping/results/gecco2025/experiment_1'))

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

fig = generate_comparison_plot(
    experiment_dir=comparison_dir,
    use_fitness_colors=False,
    csv_name='elite_fitness.csv',
    line_plot_args=LinePlotArgs(
        window_size=60,
        downsample=1
    ),
    plot_args=PlotArgs(
        title=None,
        output=None,
        xlim=[0,5000],
        ylim=[0,4],
        xlabel=None,
        ylabel=None,
        silent=False
    )
)

ax = fig.axes[0]

# Set the background color
ax.set_facecolor((0.9, 0.9, 0.9, 1))  # Set background to grey

# Customize the grid lines
ax.set_axisbelow(True)
ax.grid(color='white', linestyle='-', linewidth=1, zorder=0)  # Set grid lines to white

# Remove methods that aren't being showcased
remove_lines = [
    'Global',
    'D-Indirect-Traj',
    'Difference',
]
for line in ax.lines:
    if line.get_label() in remove_lines:
        line.remove()
remove_shading = [
    '_child1',
    '_child3',
    '_child5'
# _child7
# _child9
# _child11
# _child13
# _child15
]
for collection in ax.collections:
    if collection.get_label() in remove_shading:
        collection.remove()

label_map = {
    'D-Indirect-Timestep': 'Elite Preservation',
    'D-Indirect-Timestep-no-preservation': 'Dynamic Influence',
    'D-Indirect-Traj-no-preservation': 'Static Influence',
    'Difference-no-preservation': 'Difference',
    'Global-no-preservation': 'Global'
}

tab10_colors = plt.cm.tab10.colors
colors = reversed(tab10_colors[:5])
markers = ['*', 's', '^', 'o', 'D']
markersize_map = {
    'o': 8,
    's': 8,
    '^': 10,
    '*': 12,
    'D': 8
}
for line, collection, color, marker in zip(ax.lines, ax.collections, colors, markers):
    line.set_label(label_map[line.get_label()])
    line.set_color(color)
    collection.set_color(color)
    ax.plot(
        line.get_xdata()[::600], 
        line.get_ydata()[::600], 
        marker=marker, 
        color=color, 
        linestyle='', 
        markersize=markersize_map[marker]
    )

# ax.lines[0].set_marker()

# custom_lines = [Line2D([0], [0], color='blue', lw=4, marker='o', markersize=8, label='Custom Circle'),
#                 Line2D([0], [0], color='red', lw=4, marker='^', markersize=8, label='Custom Triangle')]
legend_lines = []
for line, marker in zip(ax.lines, markers):
    legend_line = copy(line)
    legend_line.set_marker(marker)
    legend_line.set_markersize(markersize_map[marker])
    # legend_line.set_linewidth(1)
    legend_lines.append(legend_line)

for line in ax.lines:
    line.set_linewidth(1)

font_properties = font_manager.FontProperties(family='Helvetica', size=12)
ax.legend(handles=legend_lines, prop=font_properties)
# ax.legend()

# 'font',**{'family':'sans-serif','sans-serif':['Helvetica']}

ax.tick_params(axis='both', labelsize=12)  # Use desired font size (e.g., 14)

# font_properties = font_manager.FontProperties(size=12)
font_properties = font_manager.FontProperties(family='Helvetica', size=14)
ax.set_xlabel(ax.get_xlabel(), fontproperties=font_properties)
ax.set_ylabel(ax.get_ylabel(), fontproperties=font_properties)

fig.tight_layout()

fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/experiment_2.png'))
fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/experiment_2.svg'))

plt.show()

