import numpy as np
from matplotlib import pyplot as plt

# Set matplotlib to use serif fonts and LaTeX rendering for text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'Times', 'Palatino', 'serif']
plt.rcParams['text.usetex'] = True  # Enable LaTeX for all text rendering

# Set the file name and location path
name = "R"
loc = "report/input" + name + "/"

# Load nodes
with open(loc + "nodes" + name + ".txt", "r") as f:
    lines = f.readlines()
    nodes = {
        int(parts[0]): (float(parts[1]), float(parts[2]))
        for parts in (line.strip().split() for line in lines[1:])
        if len(parts) >= 3
    }

# Load elements
with open(loc + "elements" + name + ".txt", "r") as f:
    lines = f.readlines()
    elements = [
        (int(parts[0]), list(map(int, parts[1:4])))
        for parts in (line.strip().split() for line in lines[1:])
        if len(parts) >= 4
    ]

# Load displacements and apply deformation
alpha = 0.10  # scaling factor
updated_nodes = {}
with open(loc + "output.txt", "r") as f:
    lines = f.readlines()
    for parts in (line.strip().split() for line in lines[2:]):
        if len(parts) >= 5:
            node_id = int(parts[0])
            u, v = float(parts[3]), float(parts[4])
            if node_id in nodes:
                x, y = nodes[node_id]
                updated_nodes[node_id] = (x + alpha * u, y + alpha * v)

# Utility to count nodes lying on a specific axis
def count_nodes_on_axis(node_ids, axis="x", value=0.0):
    return sum(
        1 for nid in node_ids
        if axis == "x" and np.isclose(nodes[nid][1], value) or
           axis == "y" and np.isclose(nodes[nid][0], value)
    )

# Identify elements with nodes on X or Y axis
x_axis_elements = [
    eid for eid, nids in elements if count_nodes_on_axis(nids, "x", 0.0) >= 2
]
y_axis_elements = [
    eid for eid, nids in elements if count_nodes_on_axis(nids, "y", 0.0) >= 2
]

print("Elements with at least 2 nodes on X-axis (Y=0):", x_axis_elements)
print("Elements with at least 2 nodes on Y-axis (X=0):", y_axis_elements)

# Plotting original and deformed mesh
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original mesh
axes[0].set_title("Original Mesh", fontsize=18)
for _, node_ids in elements:
    coords = [nodes[nid] for nid in node_ids] + [nodes[node_ids[0]]]  # close loop
    x_vals, y_vals = zip(*coords)
    axes[0].plot(x_vals, y_vals, 'b-')
    axes[0].fill(x_vals, y_vals, edgecolor='black', fill=False, linewidth=0.5)

axes[0].set_aspect('equal')
axes[0].grid(True)
axes[0].set_xlabel(r"$\frac{x}{R}$", fontsize=18)
axes[0].set_ylabel(r"$\frac{y}{R}$", fontsize=18)

# Deformed mesh
axes[1].set_title(r"Deformed Mesh ($\alpha = {:.2f}$)".format(alpha), fontsize=20)
for _, node_ids in elements:
    if all(nid in updated_nodes for nid in node_ids):
        coords = [updated_nodes[nid] for nid in node_ids] + [updated_nodes[node_ids[0]]]
        x_vals, y_vals = zip(*coords)
        axes[1].plot(x_vals, y_vals, 'r-')
        axes[1].fill(x_vals, y_vals, edgecolor='black', fill=False, linewidth=0.5)

axes[1].set_aspect('equal')
axes[1].grid(True)
axes[1].set_xlabel(r"$\frac{x}{R}$", fontsize=18)
axes[1].set_ylabel(r"$\frac{y}{R}$", fontsize=18)

plt.tight_layout()
plt.show()

plt.savefig(loc+"mesh")