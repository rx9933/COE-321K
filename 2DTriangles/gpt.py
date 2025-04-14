import numpy as np
import os

### PRE PROCESSING ###
loc = "testcase/"
filenames = [loc+"nodes.txt", loc+"forces.txt", loc+"displacements.txt", loc+"elements.txt"]

# Function to read non-empty lines from a file
def read_non_empty_lines(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Read nodes
node_data = read_non_empty_lines(filenames[0])
node = [list(map(float, line.split())) for line in node_data[1:]]
nodes = len(node)
ndim = 2  # 2D problem

# Read forces
force_data = read_non_empty_lines(filenames[1])
nfbcs = int(force_data[0])
force_values = [list(map(float, line.split())) for line in force_data[1:]]
fnode = [int(row[0]) for row in force_values]
fdof = [int(row[1]) for row in force_values]  # 1=x, 2=y
fval = [row[2] for row in force_values]

# Read displacements
disp_data = read_non_empty_lines(filenames[2])
ndbcs = int(disp_data[0])
disp_values = [list(map(float, line.split())) for line in disp_data[1:]]
dbcnd = [int(row[0]) for row in disp_values]
dbcdof = [int(row[1]) for row in disp_values]  # 1=x, 2=y
dbcval = [row[2] for row in disp_values]

# Read elements
elem_data = read_non_empty_lines(filenames[3])
firstrow = elem_data[0].split()
neles = int(firstrow[0])
E = float(firstrow[1])
v = float(firstrow[2])
element_info = [list(map(float, line.split())) for line in elem_data[1:]]
ele = np.array(element_info)

### MATERIAL PROPERTIES ###
C = np.array([
    [E/(1-v**2), v*E/(1-v**2), 0],
    [v*E/(1-v**2), E/(1-v**2), 0],
    [0, 0, E/(2*(1+v))]
])

### DOF NUMBERING ###
# Simple DOF numbering: 2 DOFs per node (u,v)
totdofs = 2 * nodes

# Create fixed DOFs list properly
fixed_dofs = []
for n, d in zip(dbcnd, dbcdof):
    fixed_dofs.append(2*(n-1) + (d-1))  # Convert to 0-based indexing

# Create free DOFs list
free_dofs = [i for i in range(totdofs) if i not in fixed_dofs]
ndofs = len(free_dofs)

### ELEMENT STIFFNESS MATRIX ###
def compute_K_matrix(elenum):
    nodes = ele[elenum, 1:4].astype(int)
    x1, y1 = node[nodes[0]-1][1:]
    x2, y2 = node[nodes[1]-1][1:]
    x3, y3 = node[nodes[2]-1][1:]
    
    # Element area
    A = 0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    
    # Shape function derivatives
    N1x = (y2-y3)/(2*A)
    N1y = (x3-x2)/(2*A)
    N2x = (y3-y1)/(2*A)
    N2y = (x1-x3)/(2*A)
    N3x = (y1-y2)/(2*A)
    N3y = (x2-x1)/(2*A)
    
    # B matrix
    B = np.array([
        [N1x, 0, N2x, 0, N3x, 0],
        [0, N1y, 0, N2y, 0, N3y],
        [N1y, N1x, N2y, N2x, N3y, N3x]
    ])
    
    # Element stiffness matrix
    return A * B.T @ C @ B

### ASSEMBLY ###
K_global = np.zeros((totdofs, totdofs))
F_global = np.zeros(totdofs)

# Assemble global stiffness matrix
for i in range(neles):
    nodes = ele[i, 1:4].astype(int)
    Ke = compute_K_matrix(i)
    
    # Get global DOF indices for this element [u1,v1, u2,v2, u3,v3]
    dofs = []
    for n in nodes:
        dofs.append(2*(n-1))    # u DOF
        dofs.append(2*(n-1)+1)  # v DOF
    
    # Add to global matrix
    for i_local in range(6):
        i_global = dofs[i_local]
        for j_local in range(6):
            j_global = dofs[j_local]
            K_global[i_global, j_global] += Ke[i_local, j_local]

# Apply forces
for i in range(nfbcs):
    dof = 2*(fnode[i]-1) + (fdof[i]-1)
    F_global[dof] += fval[i]

### BOUNDARY CONDITIONS ###
# Create reduced system
K_red = K_global[np.ix_(free_dofs, free_dofs)]
F_red = F_global[free_dofs]


print("K_global",K_global)
# Apply displacement BCs by modifying RHS
for i, dof in enumerate(fixed_dofs):
    F_red -= K_global[np.ix_(free_dofs, [dof])].flatten() * dbcval[i]
print("F_red",F_red)
### SOLVE ###
try:
    u_red = np.linalg.solve(K_red, F_red)
except np.linalg.LinAlgError as e:
    print("Error solving system:", e)
    print("Matrix rank:", np.linalg.matrix_rank(K_red))
    print("Matrix condition number:", np.linalg.cond(K_red))
    raise

# Combine solutions
u_full = np.zeros(totdofs)
u_full[free_dofs] = u_red
for i, dof in enumerate(fixed_dofs):
    u_full[dof] = dbcval[i]
print(u_full)

### POST-PROCESSING ###
# Nodal displacements
u_2d = np.zeros((nodes, 2))
for n in range(nodes):
    u_2d[n, 0] = u_full[2*n]    # u displacement
    u_2d[n, 1] = u_full[2*n+1]  # v displacement

# Output results
write_file = loc + 'output.txt'
with open(write_file, 'w') as f:
    # Nodal displacements
    f.write("Nodal displacements\n")
    f.write("node#      x          y          u          v\n")
    for i in range(nodes):
        f.write(f"{i+1:5d}  {node[i][1]:10.6f}  {node[i][2]:10.6f}  {u_2d[i,0]:10.6f}  {u_2d[i,1]:10.6f}\n")

print(f"Results written to {write_file}")