import numpy as np
import os
from scipy.sparse import lil_matrix, csr_matrix
### PRE PROCESSING ###
loc = "report/inputR/"
add_on = "R"
filenames = [loc+"nodes" + add_on + ".txt", loc + "forces" + add_on + ".txt", loc+"displacements"+ add_on+".txt", loc+"elements"+add_on+".txt"]

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
totdofs = 2 * nodes

# Create fixed DOFs list properly
fixed_dofs = []
for n, d in zip(dbcnd, dbcdof):
    fixed_dofs.append(2*(n-1) + (d-1))  # Convert to 0-based indexing

# Create free DOFs list
free_dofs = [i for i in range(totdofs) if i not in fixed_dofs]
ndofs = len(free_dofs)

# Create mapping from global DOF to reduced DOF
dof_map = {dof:i for i, dof in enumerate(free_dofs)}

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
# Initialize sparse matrix for reduced system
K_red = lil_matrix((ndofs, ndofs))
F_red = np.zeros(ndofs)


# Get fixed displacement values in a dictionary for quick lookup
fixed_disp = {dof:dbcval[i] for i, dof in enumerate(fixed_dofs)}


# Assemble directly into reduced system
for i in range(neles):
    nodes = ele[i, 1:4].astype(int)
    Ke = compute_K_matrix(i)
    
    # Get global DOF indices for this element [u1,v1, u2,v2, u3,v3]
    dofs = []
    for n in nodes:
        dofs.append(2*(n-1))    # u DOF
        dofs.append(2*(n-1)+1)  # v DOF
    
    # Process each DOF pair
    for i_local in range(6):
        i_global = dofs[i_local]
         
        # If this DOF is free, add to reduced system
        if i_global in dof_map:
            row = dof_map[i_global]
            
            # Add to force vector (including effect of fixed DOFs)
            for j_local in range(6):
                j_global = dofs[j_local]
                if j_global in fixed_disp:
                    F_red[row] -= Ke[i_local, j_local] * fixed_disp[j_global]
                elif j_global in dof_map:
                    col = dof_map[j_global]
                    K_red[row, col] += Ke[i_local, j_local]

# Now add external forces directly to reduced system
for i in range(nfbcs):
    dof = 2*(fnode[i]-1) + (fdof[i]-1)
    if dof in dof_map:  # Only add if DOF is free
        F_red[dof_map[dof]] += fval[i]


### SOLVE ###
try:
    # Convert to CSR format for efficient solving
    K_red_csr = K_red.tocsr()
    u_red = np.linalg.solve(K_red_csr.toarray(), F_red)
except np.linalg.LinAlgError as e:
    print("Error solving system:", e)
    print("Matrix rank:", np.linalg.matrix_rank(K_red_csr.toarray()))
    print("Matrix condition number:", np.linalg.cond(K_red_csr.toarray()))
    raise

# Combine solutions
u_full = np.zeros(totdofs)
for dof, idx in dof_map.items():
    u_full[dof] = u_red[idx]
for dof, val in fixed_disp.items():
    u_full[dof] = val

print(u_full)

### POST-PROCESSING ###
# Nodal displacements
u_2d = np.zeros((len(node), 2))
for n in range(len(node)):
    u_2d[n, 0] = u_full[2*n]    # u displacement
    u_2d[n, 1] = u_full[2*n+1]  # v displacement

# Element stresses
element_stresses = []
for i in range(neles):
    nodes = ele[i, 1:4].astype(int)
    x1,y1 = node[nodes[0]-1][1:]
    x2,y2 = node[nodes[1]-1][1:]
    x3,y3 = node[nodes[2]-1][1:]
    A = 0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    
    # B matrix
    N1x = (y2-y3)/(2*A)
    N1y = (x3-x2)/(2*A)
    N2x = (y3-y1)/(2*A)
    N2y = (x1-x3)/(2*A)
    N3x = (y1-y2)/(2*A)
    N3y = (x2-x1)/(2*A)
    B = np.array([
        [N1x, 0, N2x, 0, N3x, 0],
        [0, N1y, 0, N2y, 0, N3y],
        [N1y, N1x, N2y, N2x, N3y, N3x]
    ])
    
    # Element displacements [u1,v1, u2,v2, u3,v3]
    u_elem = np.concatenate([u_2d[n-1] for n in nodes])
    
    # Stress = C * B * u
    stress = C @ B @ u_elem
    element_stresses.append(stress)

# Write output
write_file = loc + 'output.txt'
with open(write_file, 'w') as f:
    f.write("Nodal displacements\n")
    f.write("node#      x          y          u          v\n")
    for i in range(len(node)):
        f.write(f"{i+1:5d}  {node[i][1]:10.6f}  {node[i][2]:10.6f}  {u_2d[i,0]:10.6f}  {u_2d[i,1]:10.6f}\n")
    
    f.write("\nElement stresses\n")
    f.write("ele#       σxx        σyy        τxy\n")
    for i in range(neles):
        f.write(f"{i+1:5d}  {element_stresses[i][0]:10.6f}  {element_stresses[i][1]:10.6f}  {element_stresses[i][2]:10.6f}\n")

print(f"Results written to {write_file}")