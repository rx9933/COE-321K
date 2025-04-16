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
# Initialize lists to hold σxx and σyy for X-axis and Y-axis elements
sigma_xx_xaxis = []
sigma_yy_xaxis = []
sigma_xx_yaxis = []
sigma_yy_yaxis = []

with open(loc + "output.txt", "r") as f:
    lines = f.readlines()

# Find start of "Element stresses" block
stress_start = None
for i, line in enumerate(lines):
    if "Element stresses" in line:
        stress_start = i + 2  # Skip header line
        break

# Parse element stress data and match with x_axis_elements and y_axis_elements
if stress_start is not None:
    for line in lines[stress_start:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                element_id = int(parts[0])
                sigma_xx = float(parts[1])
                sigma_yy = float(parts[2])

                if element_id in x_axis_elements:
                    sigma_xx_xaxis.append((element_id, sigma_xx))
                    sigma_yy_xaxis.append((element_id, sigma_yy))
                if element_id in y_axis_elements:
                    sigma_xx_yaxis.append((element_id, sigma_xx))
                    sigma_yy_yaxis.append((element_id, sigma_yy))
            except ValueError:
                break  # Stop on malformed or unexpected line

# # Print results
# print("\nσxx and σyy for X-axis elements:")
# for (eid_xx, sxx), (eid_yy, syy) in zip(sigma_xx_xaxis, sigma_yy_xaxis):
#     print(f"Element {eid_xx}: σxx = {sxx:.6f}, σyy = {syy:.6f}")

# print("\nσxx and σyy for Y-axis elements:")
# for (eid_xx, sxx), (eid_yy, syy) in zip(sigma_xx_yaxis, sigma_yy_yaxis):
#     print(f"Element {eid_xx}: σxx = {sxx:.6f}, σyy = {syy:.6f}")
# Compute average x-position for elements along the X-axis
# Average x-position for X-axis elements
avg_x_positions_xaxis = []
for eid in x_axis_elements:
    node_ids = next(nids for (element_id, nids) in elements if element_id == eid)
    x_coords = [nodes[nid][0] for nid in node_ids]
    avg_x = sum(x_coords) / len(x_coords)
    avg_x_positions_xaxis.append((eid, avg_x))

# Average y-position for Y-axis elements
avg_y_positions_yaxis = []
for eid in y_axis_elements:
    node_ids = next(nids for (element_id, nids) in elements if element_id == eid)
    y_coords = [nodes[nid][1] for nid in node_ids]
    avg_y = sum(y_coords) / len(y_coords)
    avg_y_positions_yaxis.append((eid, avg_y))

# # Optional: print results
# print("\nAverage x-positions for elements along X-axis:")
# for eid, x in avg_x_positions_xaxis:
#     print(f"Element {eid}: avg x = {x:.6f}")

# print("\nAverage y-positions for elements along Y-axis:")
# for eid, y in avg_y_positions_yaxis:
#     print(f"Element {eid}: avg y = {y:.6f}")

output_filename = loc + "x_y_axis_stress_data.txt"
with open(output_filename, "w") as f:
    f.write("X-Axis Elements (average_x, sigma_xx, sigma_yy):\n")
    for i, (eid, avg_x) in enumerate(avg_x_positions_xaxis):
        sigma_xx = sigma_xx_xaxis[i][1]
        sigma_yy = sigma_yy_xaxis[i][1]
        f.write(f"{avg_x:.6f}, {sigma_xx:.6f}, {sigma_yy:.6f}\n")

    f.write("\nY-Axis Elements (average_y, sigma_xx, sigma_yy):\n")
    for i, (eid, avg_y) in enumerate(avg_y_positions_yaxis):
        sigma_xx = sigma_xx_yaxis[i][1]
        sigma_yy = sigma_yy_yaxis[i][1]
        f.write(f"{avg_y:.6f}, {sigma_xx:.6f}, {sigma_yy:.6f}\n")

print(f"\nData written to: {output_filename}")



