# 10/31
## all 1 input
import numpy as np
from math import sqrt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os




### PRE PROCESSING ###


loc = "../testingtruss/3dtestcase/"
filenames = [loc+"nodes", loc+"forces", loc+"displacements", loc+"elements"]
for filename in filenames:

    # Check if the file exists without an extension
    if os.path.exists(filename) and not os.path.exists(filename + ".txt"):
        os.rename(filename, filename + ".txt")
        print(f'Renamed "{filename}" to "nodes.txt"')
    else:
        print(f'"{filename}" does not exist or "nodes.txt" already exists.')


filenames = [loc+"nodes.txt", loc+"forces.txt", loc+"displacements.txt", loc+"elements.txt"]

# Function to read non-empty lines from a file
def read_non_empty_lines(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Read nodes
node_data = read_non_empty_lines(filenames[0]) # nodes
ndim = int(node_data[0])
node = [list(map(float, line.split())) for line in node_data[1:]]
nodes = int(node[0][0])
node = node[1:]

# Read forces
force_data = read_non_empty_lines(filenames[1]) # forces
nfbcs = int(force_data[0])
force_values = [list(map(float, line.split())) for line in force_data[1:]]
fnode = [int(row[0]) for row in force_values]
fdof = [int(row[1]) for row in force_values]
fval = [row[2] for row in force_values]

# Read displacements
displacement_data = read_non_empty_lines(filenames[2]) # displacements
ndbcs = int(displacement_data[0])
displacement_values = [list(map(float, line.split())) for line in displacement_data[1:]]
dbcnd = [int(row[0]) for row in displacement_values]
dbcdof = [int(row[1]) for row in displacement_values]
dbcval = [row[2] for row in displacement_values]

# Read elements
element_data = read_non_empty_lines(filenames[3]) # elements
neles = int(element_data[0])
element_info = [list(map(float, line.split())) for line in element_data[1:]]
ele = [list(map(int, line.split()[:-2])) + list(map(float, line.split()[-2:])) for line in element_data[1:]]
ndpn = ndim




gconold = np.array([[node, dof, ndpn * (node - 1) + dof] for node in range(1, nodes + 1) for dof in range(1, ndpn + 1)])
totndofs = ndpn * nodes # total dofs

gcon = gconold.copy()
for ndbcNum in range(ndbcs):#ndbcs):#3
    for row in range(ndim*nodes): #8
        if gconold[row][0] == dbcnd[ndbcNum] and gconold[row][1] == dbcdof[ndbcNum]:
            rowIndex= row
    maxNum=gconold[rowIndex][2]
    for gIndex in range(len(gconold)):
        if gconold[gIndex][2] < maxNum:
            gcon[gIndex][2] = gconold[gIndex][2]
        elif gconold[gIndex][2] == maxNum:
            gcon[gIndex][2] = len(gconold)
        else:
            gcon[gIndex][2] = gconold[gIndex][2]-1
    gconold = gcon
    gcon = gconold.copy()




#######################################################################


# Initialize length for bars, cosine/direction matrix
LO = []  
dircos = np.zeros((neles, ndim))
Bele = np.zeros((neles, 2*ndim))
Kele = np.zeros((neles, 2*ndim, 2*ndim))
ndofs = totndofs - ndbcs  # Active DOFs only

# Apply known forces to construct Fk
Fred = np.zeros(ndofs)
for i in range(nfbcs):
    r = np.argwhere(gcon[:,0] == fnode[i])
    c = np.argwhere(gcon[:,1] == fdof[i])
    for r1 in r:
        if r1 in c:
            row = r1
            break
    dof = gcon[row,2] -1 # select 3rd col
    Fred[dof] +=fval[i]

# Apply known forces to construct uk (need to subtract out of Fk)
uinit = np.zeros((nodes,ndpn))
for i in range(ndbcs):
    uinit[dbcnd[i]-1, dbcdof[i]-1] = dbcval[i]

# Compute element stiffness matrix
Kred = np.zeros((ndofs, ndofs))  
for i in range(neles): 
    lsquared = sum((node[ele[i][2]-1][j+1] - node[ele[i][1]-1][j+1])**2 for j in range(ndim))
    LO.append(np.sqrt(lsquared)) 
    
    # Compute direction cosines
    for j in range(ndim):
        diff = node[ele[i][2]-1][j+1] - node[ele[i][1]-1][j+1]
        dircos[i][j] = diff / LO[i]
        Bele[i][j] = -dircos[i][j]
        Bele[i][j+ndim] = -Bele[i][j]
 
    for id in range(2*ndpn):
        for jd in range(2*ndpn):
            Kele[i][id,jd] += (ele[i][3] * ele[i][4] / LO[i]) * Bele[i][id]* Bele[i][jd]

# construct global stiffness (reduced): Kred + construct final RHS: Fk - K2*uk
row_idx, col_idx, values = [], [], []
for i in range(neles):  
    node1, node2 = ele[i][1], ele[i][2]
    glist = [0] * (2 * ndim)  # Stores global DOF indices for this element

    for row in range(ndim * nodes):
        if gcon[row][0] in {node1, node2}:
            for dim in range(1, ndim + 1):
                if gcon[row][0] == node1 and gcon[row][1] == dim:
                    glist[dim - 1] = gcon[row][2] - 1
                elif gcon[row][0] == node2 and gcon[row][1] == dim:
                    glist[ndim + dim - 1] = gcon[row][2] - 1

    for row in range(2 * ndim):
        if glist[row] < ndofs:  # Only for unknown DOFs
            for col in range(2 * ndim):
                if glist[col] >= ndofs:  # Only for known DOFs
                    Fred[glist[row]] -= Kele[i][row, col] * dbcval[glist[col] - ndofs]
    

    glist = np.atleast_1d(glist) 

    for row in range(len(glist)):
        for col in range(len(glist)):
            l1index = row
            l2index = col
            value = Kele[i][l1index, l2index]
            if glist[row] < ndofs and glist[col] < ndofs:# and abs(value) > 1e-8:  # Avoid storing explicit zeros
                Kred[glist[row],glist[col]] += value

u_red = np.linalg.solve(Kred, Fred)
# Append prescribed displacements
u = np.concatenate([u_red, dbcval])

u_flat = [0]*ndim*nodes
for i in range(ndim*nodes):
    uflatindex =gcon[i,2]-1
    u_flat[i] = u[uflatindex]

# Calculate internal forces Nbar using Bele and local displacements (u_local)
Nbar = np.zeros(neles)  
strain = np.zeros(neles) 
for i in range(neles):
    node1, node2 = ele[i][1], ele[i][2]
    u_local = np.zeros(2 * ndim) 
    for j in range(ndim): 
        u_local[j] = u_flat[ndpn * (node1 - 1) + j]  
        u_local[ndim + j] = u_flat[ndpn * (node2 - 1) + j]  
    strain[i] =  np.dot(Bele[i], u_local)/LO[i]
    Nbar[i] = ele[i][3] * ele[i][4] * strain[i] # EA/L * Bele^T * u_local

ele = np.array(ele)
Fext = np.zeros((nodes, ndpn))
for iele in range(neles):
    for localnode in range(2):
        for localdof in range(ndpn):
            globalnode=int(ele[iele, localnode+1])
            Fext[globalnode-1, localdof] += Nbar[iele] * Bele[iele, ndpn*(localnode)+localdof]


## Post processing

# create file to write output
write_file = loc + 'output.txt'

with open(write_file, 'w') as f:
    # Define labels dynamically based on `ndim`
    coord_labels = ["x", "y", "z"][:ndim]  # Trim to 2D or 3D
    disp_labels = ["u", "v", "w"][:ndim]
    force_labels = ["fx", "fy", "fz"][:ndim]

    # Write headers
    f.write("Nodal displacements\n")
    f.write(f"node#  {'  '.join(coord_labels)}  {'  '.join(disp_labels)}\n")
    
    # Write nodal displacements
    for i in range(nodes):
        f.write(f"{i+1:4d} " + 
                " ".join(f"{node[i][j+1]:10.6f}" for j in range(ndim)) + " " + 
                " ".join(f"{u_flat[ndim*i + j]:10.6f}" for j in range(ndim)) + "\n")

    f.write("\nExternal forces\n")
    f.write(f"node#  {'  '.join(coord_labels)}  {'  '.join(force_labels)}\n")

    # Write external forces
    for i in range(nodes):
        f.write(f"{i+1:4d} " + 
                " ".join(f"{node[i][j+1]:10.6f}" for j in range(ndim)) + " " + 
                " ".join(f"{Fext[i][j]:10.6f}" for j in range(ndim)) + "\n")

    f.write("\nElement axial strains and forces\n")
    f.write("ele#      strain       force        L\n")

    # Write element strains and forces
    for i in range(neles):
        f.write(f"{i+1:4d} {strain[i]:10.6f} {Nbar[i]:10.6f} {LO[i]:10.6f}\n")

print("Fext",Fext)

LF = np.zeros(neles)  # Keep as NumPy array

delta = np.array(u_flat).reshape(nodes, ndim)
add = np.hstack((np.zeros((nodes, 1)), delta))  # Ensure shape (nodes, ndim+1)
nodef = np.array(node) + add  # Ensure node is a NumPy array

for i in range(neles): 
    lfsquared = sum((nodef[int(ele[i][2]-1), j+1] - nodef[int(ele[i][1]-1), j+1])**2 for j in range(ndim))
    LF[i] = np.sqrt(lfsquared)  # Assign instead of append
print(LO)