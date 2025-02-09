# 10/31

import numpy as np
from math import sqrt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def split_cols(lines, index):
    data = []
    while index < len(lines):
        if lines[index].strip():
            columns = lines[index].split()
            data.append(columns)
            index += 1
        else:
            break
    return data

with open('hw2p1.txt', 'r') as file: # 2d or 3d input text file
    lines = file.readlines()

lineno = 0

for line in lines:
    if "Number of spatial dimensions:" in line:
        ndim = int(lines[lineno + 1])
        ndpn = ndim
    elif "Number of joints/nodes:" in line:
        nodes = int(lines[lineno + 1])
    elif "Node #, x-location, y-location, (z-location if 3D)" in line:
        node = []
        index = lineno + 1
        data = split_cols(lines, index)
        for columns in data:
            nodeno = int(columns[0])
            if ndim == 2:
                x, y = float(columns[1]), float(columns[2])
                node.append([nodeno, x, y])
            else:
                x, y, z = float(columns[1]), float(columns[2]), float(columns[3])
                node.append([nodeno, x, y, z])
    elif "Number of bars/elements:" in line:
        neles = int(lines[lineno + 1])
    elif "Element#" in line:
        ele = []
        index = lineno + 1
        data = split_cols(lines, index)
        for columns in data:
            element_number = int(columns[0])
            local_node1 = int(columns[1])
            local_node2 = int(columns[2])
            youngs_modulus = float(columns[3])
            area = float(columns[4])
            ele.append([element_number, local_node1, local_node2, youngs_modulus, area])
    elif "Number of applied forces" in line:
        nfbcs = int(lines[lineno + 1])
    elif "Node#, Force direction, Force value" in line:
        fnode = []
        fdof = []
        fval = []
        force = []
        index = lineno + 1
        data = split_cols(lines, index)
        for columns in data:
            fnode.append(int(columns[0]))
            fdof.append(int(columns[1]))
            fval.append(float(columns[2]))
            force.append([int(columns[0]), int(columns[1]), float(columns[2])]) # array of forces info
    elif "Number of known/applied displacements" in line:
        ndbcs = int(lines[lineno + 1])
    elif "Node#, Displacement direction, Displacement value" in line:
        index = lineno + 1
        data = split_cols(lines, index)
        dbcnd=[]
        dbcdof=[]
        dbcval=[]
        dbc=[] # array of displacment info
        for columns in data:
            dbcnd.append(int(columns[0]))
            dbcdof.append(int(columns[1]))
            dbcval.append(float(columns[2]))
            dbc.append([int(columns[0]), int(columns[1]), float(columns[2])])
    lineno += 1

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
            gcon[gIndex][2] = gconold[gIndex - 1][2]
    gconold = gcon
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
            if glist[row] < ndofs and glist[col] < ndofs and abs(value) > 1e-12:  # Avoid storing explicit zeros
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

# Append prescribed displacements
u = np.concatenate([u_red, dbcval])

print(u)

