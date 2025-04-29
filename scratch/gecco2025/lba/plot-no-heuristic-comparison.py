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

comparison_dir = Path(os.path.expanduser('~/influence-shaping/results/organize/1x1_random/preservation/no-heuristic'))

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

# Changing color on D-Indirect-Traj-Random
ax.get_lines()[2].set_color('purple')
ax.collections[2].set_facecolor('purple')

# Let's add markers
markers_map = {
    'Global': 's',
    'Difference': '^',
    'D-Indirect-Traj-Random': 'o'
}
markersize_map = {
    's': 8,
    '^': 10,
    'o': 8
}
for line, collection in zip(ax.lines, ax.collections):
    ax.plot(
        line.get_xdata()[::200],
        line.get_ydata()[::200],
        marker=markers_map[line.get_label()],
        color=line.get_color(),
        linestyle='',
        markersize=markersize_map[markers_map[line.get_label()]]
    )

# Set the background color
ax.set_facecolor((0.9, 0.9, 0.9, 1))  # Set background to grey

# Customize the grid lines
ax.set_axisbelow(True)
ax.grid(color='white', linestyle='-', linewidth=1, zorder=0)  # Set grid lines to white

# Let's order the legend handles
ordered_list = [
    'D-Indirect-Traj-Random',
    'Global',
    'Difference'
]

ordered_legend_handles = []
for label in ordered_list:
    for line in ax.lines:
        if line.get_label() == label:
            legend_line = copy(line)
            ordered_legend_handles.append(legend_line)
            legend_line.set_marker(markers_map[line.get_label()])

# Now fix the names
label_map = {
    'Global': 'Global',
    'Difference': 'Difference',
    'D-Indirect-Traj-Random': 'Random Credit'
}
legend_lines = []
for line in ordered_legend_handles:
    legend_line = copy(line)
    legend_line.set_label(label_map[line.get_label()])
    legend_lines.append(legend_line)

font_properties = font_manager.FontProperties(family='Helvetica', size=16)
ax.legend(handles=legend_lines, prop=font_properties)

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

fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/random_credit.png'), dpi=300)
fig.savefig(os.path.expanduser('~/influence-shaping/outfigs/gecco2025/random_credit.svg'))

plt.show()

