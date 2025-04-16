import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# Define paths and labels
mesh_dirs = {
    "report/input6": "Mesh 6",
    "report/input12": "Mesh 12",
    "report/input24": "Mesh 24",
    "report/inputR": "Radial Mesh"
}

# Define marker styles for each mesh
markers = ['o', 's', '^', 'd']

# Updated, diverse color scheme
sigma_xx_colors = ['#1f77b4', '#4c8b9f', '#5d3b6d', '#3c6e91']  # Blues with teal and violet for sigma_xx
sigma_yy_colors = ['#9c0f48', '#d32f2f', '#c2185b', '#800020']  # Reds with crimson, scarlet, and burgundy for sigma_yy

# Container for all datasets
data = {}

# Parse each file
for mesh_dir in mesh_dirs:
    filepath = os.path.join(mesh_dir, "x_y_axis_stress_data.txt")
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Extract data lines
    x_data = []
    y_data = []
    mode = None
    for line in lines:
        line = line.strip()
        if line.startswith("X-Axis Elements"):
            mode = "x"
            continue
        elif line.startswith("Y-Axis Elements"):
            mode = "y"
            continue
        elif not line or mode is None:
            continue

        parts = line.split(",")
        if len(parts) != 3:
            continue
        avg, s_xx, s_yy = map(float, parts)
        if mode == "x":
            x_data.append((avg, s_xx, s_yy))
        elif mode == "y":
            y_data.append((avg, s_xx, s_yy))

    data[mesh_dir] = {"x": x_data, "y": y_data}

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# LEFT: X-axis data
for i, (mesh_dir, label) in enumerate(mesh_dirs.items()):
    x_data = data[mesh_dir]["x"]
    avg_x = [row[0] for row in x_data]
    sigma_xx = [row[1] for row in x_data]
    sigma_yy = [row[2] for row in x_data]
    
    # Scatter plot for sigma_xx and sigma_yy
    axes[0].scatter(avg_x, sigma_xx, marker=markers[i], color=sigma_xx_colors[i], label=rf"{label}: $\sigma_{{xx}}$")
    axes[0].scatter(avg_x, sigma_yy, marker=markers[i], color=sigma_yy_colors[i], label=rf"{label}: $\sigma_{{yy}}$")

axes[0].set_title(r"X-Axis Elements")
axes[0].set_xlabel(r"$\frac{x}{R}$", fontsize=16)
axes[0].set_ylabel(r"$\frac{\sigma}{\tilde{\sigma}}$", fontsize=16)

# Refine the grid
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid for both major and minor ticks
axes[0].minorticks_on()  # Enable minor ticks
axes[0].grid(True, which='minor', linestyle=':', linewidth=0.5)  # Minor gridlines are dotted and thinner

# Make axis lines bold and set them to pass through (0,0)
axes[0].spines['left'].set_position(('data', 0))  # Move left spine to 0
axes[0].spines['bottom'].set_position(('data', 0))  # Move bottom spine to 0
axes[0].spines['left'].set_linewidth(2)  # Bold left axis
axes[0].spines['bottom'].set_linewidth(2)  # Bold bottom axis
axes[0].spines['right'].set_linewidth(2)  # Bold right axis
axes[0].spines['top'].set_linewidth(2)  # Bold top axis

# Ensure the axes form a complete frame, making sure the top and right sides are visible.
axes[0].spines['top'].set_color('black')
axes[0].spines['right'].set_color('black')

axes[0].legend(fontsize=10)
axes[0].set_xlim(left=0)

# RIGHT: Y-axis data
for i, (mesh_dir, label) in enumerate(mesh_dirs.items()):
    y_data = data[mesh_dir]["y"]
    avg_y = [row[0] for row in y_data]
    sigma_xx = [row[1] for row in y_data]
    sigma_yy = [row[2] for row in y_data]
    
    # Scatter plot for sigma_xx and sigma_yy
    axes[1].scatter(avg_y, sigma_xx, marker=markers[i], color=sigma_xx_colors[i], label=rf"{label}: $\sigma_{{xx}}$")
    axes[1].scatter(avg_y, sigma_yy, marker=markers[i], color=sigma_yy_colors[i], label=rf"{label}: $\sigma_{{yy}}$")

axes[1].set_title(r"Y-Axis Elements")
axes[1].set_xlabel(r"$\frac{y}{R}$", fontsize=16)
axes[1].set_ylabel(r"$\frac{\sigma}{\tilde{\sigma}}$", fontsize=16)

# Refine the grid for the right plot
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid for both major and minor ticks
axes[1].minorticks_on()  # Enable minor ticks
axes[1].grid(True, which='minor', linestyle=':', linewidth=0.5)  # Minor gridlines are dotted and thinner

# Make axis lines bold and set them to pass through (0,0)
axes[1].spines['left'].set_position(('data', 0))  # Move left spine to 0
axes[1].spines['bottom'].set_position(('data', 0))  # Move bottom spine to 0
axes[1].spines['left'].set_linewidth(2)  # Bold left axis
axes[1].spines['bottom'].set_linewidth(2)  # Bold bottom axis
axes[1].spines['right'].set_linewidth(2)  # Bold right axis
axes[1].spines['top'].set_linewidth(2)  # Bold top axis

# Ensure the axes form a complete frame, making sure the top and right sides are visible.
axes[1].spines['top'].set_color('black')
axes[1].spines['right'].set_color('black')

axes[1].legend(fontsize=10)
axes[1].set_xlim(left=0)

plt.tight_layout()
plt.savefig("report/results/stresses")
plt.show()
