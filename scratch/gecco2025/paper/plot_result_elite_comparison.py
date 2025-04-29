"""This plots the learning curves for paper figures"""

from pathlib import Path
import os
from influence.plotting import plot_comparison, generate_comparison_plot
from influence.parsing import LinePlotArgs, PlotArgs
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from matplotlib import font_manager

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
        ylim=[0,1.0],
        xlabel=None,
        ylabel=None,
        silent=False
    )
)

ax = fig.axes[0]

ax.set_title("CCEA Comparison", fontsize=18, fontweight='bold', fontname='Helvetica')

# Set the background color
ax.set_facecolor((0.9, 0.9, 0.9, 1))  # Set background to grey

# Customize the grid lines
ax.set_axisbelow(True)
ax.grid(color='white', linestyle='-', linewidth=1, zorder=0)  # Set grid lines to white

for line, c in zip(ax.lines, ax.collections):
    print(line.get_label(), c.get_label())

# Remove methods that aren't being showcased
remove_lines = [
    'Global-no-preservation',
    'Difference-no-preservation',
    'Global',
    'Difference'
]
for line in ax.lines:
    if line.get_label() in remove_lines:
        line.remove()
remove_shading = [
    '_child1',
    '_child3',
    # '_child5',
    # '_child7'
    # '_child9',
    # '_child11',
    '_child13',
    '_child15',
]

for collection in ax.collections:
    if collection.get_label() in remove_shading:
        collection.remove()

tab10_colors = plt.cm.tab10.colors
colors_dict = {
    'D-Indirect-Timestep': tab10_colors[3],
    'D-Indirect-Traj': tab10_colors[2],
    'D-Indirect-Timestep-no-preservation': tab10_colors[6],
    'D-Indirect-Traj-no-preservation': tab10_colors[7]
}
markers_dict = {
    'D-Indirect-Timestep': 'D',
    'D-Indirect-Traj': 'o',
    'D-Indirect-Timestep-no-preservation': 'p',
    'D-Indirect-Traj-no-preservation': 'P'
}
markersize_map = {
    'o': 8,
    's': 8,
    '^': 10,
    '*': 12,
    'D': 8,
    'P': 8,
    'p': 8
}
label_map = {
    'D-Indirect-Timestep': 'Dynamic Influence + Mixed Elites',
    'D-Indirect-Traj': 'Static Influence + Mixed Elites',
    'D-Indirect-Timestep-no-preservation': 'Dynamic Influence + Standard Elites',
    'D-Indirect-Traj-no-preservation': 'Static Influence + Standard Elites'
}
reverse_label_map = {label_map[k]: k for k in label_map}

legend_dict = {}
# legend_lines = []
for line in ax.lines:
    color = colors_dict[line.get_label()]
    marker = markers_dict[line.get_label()]
    legend_line = copy(line)
    legend_line.set_marker(marker)
    legend_line.set_color(color)
    legend_line.set_markersize(markersize_map[marker])
    legend_line.set_label(label_map[line.get_label()])
    # legend_line.set_linewidth(1)
    legend_dict[legend_line.get_label()] = legend_line

    # legend_lines.append(legend_line)

ordered_list = [
    'Dynamic Influence + Mixed Elites',
    'Static Influence + Mixed Elites',
    'Dynamic Influence + Standard Elites',
    'Static Influence + Standard Elites'
]

legend_lines = []
for label in ordered_list:
    legend_lines.append(legend_dict[label])

# # Sort the legend line
# new_legend_lines = []
# for line in legend_line

for line, collection in zip(ax.lines, ax.collections):
    color = colors_dict[line.get_label()]
    ax.plot(
        line.get_xdata()[::200],
        line.get_ydata()[::200],
        marker=markers_dict[line.get_label()],
        color=color,
        linestyle='',
        markersize=markersize_map[markers_dict[line.get_label()]]
    )
    line.set_label(label_map[line.get_label()])
    line.set_color(color)
    collection.set_color(color)



# for line in ax.lines:
#     line.set_linewidth(1)

# for line, collection in zip(ax.lines, ax.collections):
#     color = colors_dict[line.get_label()]
#     ax.plot(
#         line.get_xdata()[::200],
#         line.get_ydata()[::200],
#         marker=markers_dict[line.get_label()],
#         color=color,
#         linestyle='',
#         markersize=markersize_map[markers_dict[line.get_label()]]
#     )
#     line.set_label(label_map[line.get_label()])
#     line.set_color(color)
#     collection.set_color(color)

ticksize = 16
xticks = ax.get_xticks()
# xticks = [str(int(t)) for t in xticks[::2]]
xticks = [str(int(t)) for t in xticks]

ax.set_xticklabels(xticks, fontsize=ticksize)
ax.set_xticks([int(x) for x in xticks])

yticks = ax.get_yticks()
yticks = [str(float(t))[:3] for t in yticks]
ax.set_yticklabels(yticks, fontsize=ticksize)

font_properties = font_manager.FontProperties(family='Helvetica', size=16)
ax.legend(handles=legend_lines, prop=font_properties)
# # ax.get_legend().remove()


font_properties = font_manager.FontProperties(family='Helvetica', size=17)
ax.set_xlabel(ax.get_xlabel(), fontproperties=font_properties)
ax.set_ylabel(ax.get_ylabel(), fontproperties=font_properties)

fig.tight_layout()

fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/experiment_elite_comparison.png'), dpi=300)
fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/experiment_elite_comparison.svg'))

plt.show()

