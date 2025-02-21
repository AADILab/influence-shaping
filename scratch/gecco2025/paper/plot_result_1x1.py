"""This plots the learning curves for paper figures"""

from pathlib import Path
import os
from influence.plotting import plot_comparison, generate_comparison_plot
from influence.parsing import LinePlotArgs, PlotArgs
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from matplotlib import font_manager
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

comparison_dir = Path(os.path.expanduser('~/influence-shaping/results/01_05_2025/yabby/31_more_circles/1x1'))

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

fig = generate_comparison_plot(
    experiment_dir=comparison_dir,
    use_fitness_colors=False,
    csv_name='fitness.csv',
    line_plot_args=LinePlotArgs(
        window_size=60,
        downsample=1
    ),
    plot_args=PlotArgs(
        title=None,
        output=None,
        xlim=[0,1000],
        ylim=[0,1],
        xlabel=None,
        ylabel=None,
        silent=False
    )
)

ax = fig.axes[0]

ax.set_title("1 Rover, Drone and POI", fontsize=18, fontweight='bold', fontname='Helvetica')

# Set the background color
ax.set_facecolor((0.9, 0.9, 0.9, 1))  # Set background to grey

# Customize the grid lines
ax.set_axisbelow(True)
ax.grid(color='white', linestyle='-', linewidth=1, zorder=0)  # Set grid lines to white

# Remove methods that aren't being showcased
remove_lines = [
    'Global-no-preservation',
    'D-Indirect-Traj-no-preservation',
    'Difference-no-preservation',
    'D-Indirect-Timestep-no-preservation'
]
for line in ax.lines:
    if line.get_label() in remove_lines:
        line.remove()
remove_shading = [
    # '_child1',
    # '_child3',
    # '_child5',
    # '_child7'
    '_child9',
    '_child11',
    '_child13',
    '_child15',
]
for collection in ax.collections:
    if collection.get_label() in remove_shading:
        collection.remove()

label_map = {
    'D-Indirect-Timestep': 'Dynamic Influence',
    'D-Indirect-Traj': 'Static Influence',
    'Difference': 'Difference',
    'Global': 'Global'
}

tab10_colors = plt.cm.tab10.colors
# # colors = reversed(tab10_colors[:4])
colors = tab10_colors[:4]
markers = ['s', '^', 'o', 'D']
# markers = list(reversed(markers[:4]))
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
        line.get_xdata()[::200], 
        line.get_ydata()[::200], 
        marker=marker, 
        color=color, 
        linestyle='', 
        markersize=markersize_map[marker]
    )

legend_lines = []
for line, marker in zip(ax.lines, markers):
    legend_line = copy(line)
    legend_line.set_marker(marker)
    legend_line.set_markersize(markersize_map[marker])
    # legend_line.set_linewidth(1)
    legend_lines.append(legend_line)
legend_lines.reverse()
for line in ax.lines:
    line.set_linewidth(1)

# ax.set_xticks(ax.get_xticks(), labels=ax.get_xticklabels())
# ax.set_xt

ticksize = 16
xticks = ax.get_xticks()
xticks = [str(int(t)) for t in xticks]
ax.set_xticklabels(xticks, fontsize=ticksize)
yticks = ax.get_yticks()
yticks = [str(float(t))[:3] for t in yticks]
ax.set_yticklabels(yticks, fontsize=ticksize)

font_properties = font_manager.FontProperties(family='Helvetica', size=17)
ax.legend(handles=legend_lines, prop=font_properties)

# # 'font',**{'family':'sans-serif','sans-serif':['Helvetica']}

# ax.tick_params(axis='both', labelsize=12)  # Use desired font size (e.g., 14)

# font_properties = font_manager.FontProperties(size=12)
font_properties = font_manager.FontProperties(family='Helvetica', size=17)
ax.set_xlabel(ax.get_xlabel(), fontproperties=font_properties)
ax.set_ylabel(ax.get_ylabel(), fontproperties=font_properties)

fig.tight_layout()

fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/experiment_1x1.png'))
fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/experiment_1x1.svg'))

plt.show()

