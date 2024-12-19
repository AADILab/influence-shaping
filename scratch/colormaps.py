import matplotlib.pyplot as plt
import numpy as np

# Create a figure and an Axes instance
fig, ax = plt.subplots()

# Set a color cycle using a colormap (e.g., 'tab10')
colors = plt.cm.tab10.colors  # Extract the colors from a qualitative colormap
ax.set_prop_cycle(color=colors)

# Generate example data
x = np.linspace(0, 10, 100)
y = [np.sin(x + phase) for phase in np.linspace(0, 2 * np.pi, 5)]

# Plot multiple lines with the custom color cycle
for i, y_line in enumerate(y):
    ax.plot(x, y_line, label=f'Line {i+1}')

# Set a color cycle using a colormap (e.g., 'tab10')
colors = plt.cm.Pastel1.colors  # Extract the colors from a qualitative colormap
ax.set_prop_cycle(color=colors)

# Generate example data
x = np.linspace(0, 10, 100)
y = [np.sin(x + phase) for phase in np.linspace(0, 2 * np.pi, 5)]

# Plot multiple lines with the custom color cycle
for i, y_line in enumerate(y):
    ax.plot(x, y_line, label=f'Line {i+100}')

ax.legend()
plt.show()
