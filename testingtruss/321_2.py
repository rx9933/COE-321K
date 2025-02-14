import numpy as np
from math import sqrt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os

filenames = ["nodes", "displacements","forces","elements"]
for filename in filenames:

    # Check if the file exists without an extension
    if os.path.exists(filename) and not os.path.exists(filename + ".txt"):
        os.rename(filename, filename + ".txt")
        print(f'Renamed "{filename}" to "nodes.txt"')
    else:
        print(f'"{filename}" does not exist or "nodes.txt" already exists.')



# Function to read non-empty lines from a file
def read_non_empty_lines(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Read nodes
node_data = read_non_empty_lines('nodes.txt')
ndim = int(node_data[0])
node = [list(map(float, line.split())) for line in node_data[1:]]
nodes = int(node[0][0])
node = node[1:]

# Read forces
force_data = read_non_empty_lines('forces.txt')
nfbcs = int(force_data[0])
force_values = [list(map(float, line.split())) for line in force_data[1:]]
fnode = [int(row[0]) for row in force_values]
fdof = [int(row[1]) for row in force_values]
fval = [row[2] for row in force_values]
print("dbcval",fval)
# Read displacements
displacement_data = read_non_empty_lines('displacements.txt')
ndbcs = int(displacement_data[0])
displacement_values = [list(map(float, line.split())) for line in displacement_data[1:]]
dbcnd = [int(row[0]) for row in displacement_values]
dbcdof = [int(row[1]) for row in displacement_values]
dbcval = [row[2] for row in displacement_values]

# Read elements
element_data = read_non_empty_lines('elements.txt')
neles = int(element_data[0])
element_info = [list(map(float, line.split())) for line in element_data[1:]]
youngs_modulus, area =  element_info[3], element_info[4]
# ele = [list(map(int, line.split())) for line in element_data[1:]]
ele = [list(map(int, line.split()[:-2])) + list(map(float, line.split()[-2:])) for line in element_data[1:]]
ndpn = ndim

gconold = np.array([[node, dof, ndpn * (node - 1) + dof] for node in range(1, nodes + 1) for dof in range(1, ndpn + 1)])
totndofs = ndpn * nodes # total dofs

gcon = gconold.copy()
for ndbcNum in range(ndbcs):#ndbcs):#3
    for row in range(2*nodes): #8
        if gconold[row][0] == dbcnd[ndbcNum] and gconold[row][1] == dbcdof[ndbcNum]:
            rowIndex= row
            break
    maxNum=gconold[rowIndex][2]
    for gIndex in range(len(gconold)):
        if gconold[gIndex][2] < maxNum:
            gcon[gIndex][2] = gconold[gIndex][2]
        elif gconold[gIndex][2] == maxNum:
            gcon[gIndex][2] = len(gconold)
        else:
            gcon[gIndex][2] = gconold[gIndex][2]-1
    gconold = gcon
    # print("gn", gcon)
    gcon = gconold.copy()


#######################################################################


# Initialize length for bars, cosine/direction matrix
LO = []  
dircos = np.zeros((neles, ndim))
Bele = np.zeros((neles, 2*ndim))
Kele = np.zeros((neles, 2*ndim, 2*ndim))



row_idx, col_idx, values = [], [], []
for i in range(neles):  # Loop through elements
    lsquared = sum((node[ele[i][2]-1][j+1] - node[ele[i][1]-1][j+1])**2 for j in range(ndim))
    LO.append(sqrt(lsquared)) 
    
    for j in range(ndim):  # Compute direction cosines
        diff = node[ele[i][2]-1][j+1] - node[ele[i][1]-1][j+1]
        dircos[i][j] = diff / LO[i]
        Bele[i][j] = -dircos[i][j]
        Bele[i][j+ndim] = -Bele[i][j]

    Kele[i] = (ele[i][3] * ele[i][4] / LO[i]) * np.outer(Bele[i], Bele[i])  # EA/L * Bele^T * Bele

    node1, node2 = ele[i][1], ele[i][2]
    glist = [0] * (2 * ndim)  # Stores global DOF indices for this element

    for row in range(ndim * nodes):
        if gcon[row][0] in {node1, node2}:
            for dim in range(1, ndim + 1):
                if gcon[row][0] == node1 and gcon[row][1] == dim:
                    glist[dim - 1] = gcon[row][2] - 1
                elif gcon[row][0] == node2 and gcon[row][1] == dim:
                    glist[ndim + dim - 1] = gcon[row][2] - 1

    ndofs = totndofs - ndbcs  # Active DOFs only

    # Store only nonzero values in sparse format
    for row in range(len(glist)):
        for col in range(len(glist)):
            value = Kele[i][row, col]
            if glist[row] < ndofs and glist[col] < ndofs and abs(value) > 1e-8:  # Avoid storing explicit zeros
                row_idx.append(glist[row])
                col_idx.append(glist[col])
                values.append(value)



# Create sparse matrix in COO format and convert to CSR
Kred = coo_matrix((values, (row_idx, col_idx)), shape=(ndofs, ndofs)).tocsr()

###
# Set up force array
Fred = np.zeros(ndofs)

# Compute force contributions from elements
for i in range(neles):
    node1, node2 = ele[i][1], ele[i][2]
    glist = [0] * (2 * ndim)  # Store global DOF indices

    for row in range(ndim * nodes):
        if gcon[row][0] in {node1, node2}:
            for dim in range(1, ndim + 1):
                if gcon[row][0] == node1 and gcon[row][1] == dim:
                    glist[dim - 1] = gcon[row][2] - 1
                elif gcon[row][0] == node2 and gcon[row][1] == dim:
                    glist[ndim + dim - 1] = gcon[row][2] - 1

    # Subtract known displacement contributions
    for row in range(2 * ndim):
        if glist[row] < ndofs:  # Only for unknown DOFs
            for col in range(2 * ndim):
                if glist[col] >= ndofs:  # Only for known DOFs
                    Fred[glist[row]] -= Kele[i][row, col] * dbcval[glist[col] - ndofs]

# Apply forces to the reduced force vector
for i in range(nfbcs):
    for row in range(ndim * nodes):
        if gcon[row][0] == fnode[i] and gcon[row][1] == fdof[i]:
            gdofi = gcon[row][2] - 1  # Map to zero-based index for reduced DOF
            if gdofi < ndofs:  # Apply force only to active DOFs
                Fred[gdofi] += fval[i]  # Apply the force value at the correct index

# Solve for unknown displacements
u_red = spsolve(Kred.tocsr(), Fred)
Kred_matrix = Kred.toarray()  # Convert sparse matrix to a dense matrix
print("Kred as matrix:")
print(Kred)



# Append prescribed displacements
u = np.concatenate([u_red, dbcval])
print("Uu", u)
# convert u to regular (non gdof) list
u_flat = [0]*ndim*nodes
for i in range(ndim*nodes):
    uflatindex =gcon[i,2] -1
    u_flat[i] = u[uflatindex]
print(u_flat)
"""
# Calculate internal forces Nbar using Bele and local displacements (u_local)
Nbar = np.zeros(neles)  

for i in range(neles):
    node1, node2 = ele[i][1], ele[i][2]
    u_local = np.zeros(2 * ndim) 
    for j in range(ndim): 
        u_local[j] = u_flat[ndpn * (node1 - 1) + j]  
        u_local[ndim + j] = u_flat[ndpn * (node2 - 1) + j]  
    Nbar[i] = (ele[i][3] * ele[i][4] / LO[i]) * np.dot(Bele[i], u_local)  # EA/L * Bele^T * u_local

ele = np.array(ele)
Fext = np.zeros((nodes, ndpn))
for iele in range(neles):
    for localnode in range(2):
        for localdof in range(ndpn):
            globalnode=int(ele[iele, localnode+1])
            Fext[globalnode-1, localdof] += Nbar[iele] * Bele[iele, ndpn*(localnode)+localdof]


stress = Nbar/ele[:,4]

print("U",u_flat)
print("N", Nbar)
print("Fext", Fext)
print("stress",stress)

#### process into soln document# Generate solution document
with open('solution.txt', 'w') as f:
    f.write("Nodal displacements\n")
    f.write("node# x y z u v w\n")
    for i in range(nodes):
        f.write(f"{i+1:4d} {node[i][1]:10.6f} {node[i][2]:10.6f} {node[i][3]:10.6f} {u_flat[3*i]:10.6f} {u_flat[3*i+1]:10.6f} {u_flat[3*i+2]:10.6f}\n")
    
    f.write("\nExternal forces\n")
    f.write("node# x y fx fy fz\n")
    for i in range(nfbcs):
        f.write(f"{fnode[i]:4d} {node[fnode[i]-1][1]:10.6f} {node[fnode[i]-1][2]:10.6f} {fval[i]:10.6f}\n")
    
    f.write("\nElement axial strains and forces\n")
    f.write("ele# strain force L\n")
    for i in range(neles):
        f.write(f"{i+1:4d} {stress[i]:10.6f} {Nbar[i]:10.6f} {LO[i]:10.6f}\n")
"""