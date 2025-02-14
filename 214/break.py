# 10/31
## all 1 input
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

with open("../testingtruss/tetr.txt", 'r') as file: # 2d or 3d input text file
# with open("../hw2p1.txt", 'r') as file: # 2d or 3d input text file
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

ndofs = totndofs - ndbcs  # Active DOFs only

# Initialize global stiffness matrix (assuming it's not too large)
Kred = np.zeros((totndofs - ndbcs, totndofs - ndbcs))  
Fred = np.zeros(ndofs)

for i in range(nfbcs):
    dof = gcon[fnode[i]-1, fdof[i]-1]
    Fred[dof] +=fval[i]

uinit = np.zeros((nodes,ndpn))
for i in range(ndbcs):
    uinit[dbcnd[i]-1, dbcdof[i]-1] = dbcval[i]

for i in range(neles):  # Loop through elements
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
        # Compute element stiffness matrix
            Kele[i][id,jd] += (ele[i][3] * ele[i][4] / LO[i]) * Bele[i][id]* Bele[i][jd]
   
    for inode in range(2):
        for idof in range(ndpn):
            idoflocal = (inode)*ndpn+idof
            idofglobal = gcon[ele[i][inode],idof]-1
            if idofglobal<=ndofs:
                for jnode in range(2):
                    for jdof in range(ndpn):
                        jdoflocal = (jnode)*ndpn+jdof
                        jdofglobal=gcon[ele[i][jnode],jdof]-1
                        if jdofglobal<ndofs:
                            # if i == 0:
                                # print(Kele[i][idoflocal,jdoflocal], i,idoflocal,jdoflocal, idofglobal, jdofglobal)
                            # print(i, idofglobal,jdofglobal,Kele[i][idoflocal,jdoflocal])
                            Kred[idofglobal,jdofglobal]+=Kele[i][idoflocal,jdoflocal]
                            # print(Kred)
                        else:
                            Fred[idofglobal] -=Kele[i][idoflocal,jdoflocal] * uinit[ele[i][jnode+1]-1,jdof]
    # if i>=1:
    #     break
Kred = np.zeros((totndofs - ndbcs, totndofs - ndbcs))  
row_idx, col_idx, values = [], [], []
for i in range(neles):  # Loop through elements
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
    glist = np.atleast_1d(glist) 
    print(Kele[i])
    print("GLIST",glist)
    # Store only nonzero values in sparse format
    for row in range(len(glist)):
        for col in range(len(glist)):
            # l1index = np.argwhere(glist==row)
            # l2index = np.argwhere(glist==col)
            l1index = row
            l2index = col
     
            value = Kele[i][l1index, l2index]
      
            if glist[row] < ndofs and glist[col] < ndofs:# and abs(value) > 1e-8:  # Avoid storing explicit zeros
                print("AA",l1index,l2index,value)
                Kred[glist[row],glist[col]] += value
                print("Kred",Kred)
# Kred = coo_matrix((values, (row_idx, col_idx)), shape=(ndofs, ndofs)).tocsr()
print("K",Kred)  # Print dense matrix
print(Fred)
print(gcon)
# Solve for unknown displacements
u_red = spsolve(Kred, Fred)

# Append prescribed displacements
u = np.concatenate([u_red, dbcval])

# convert u to regular (non gdof) list
u_flat = [0]*ndim*nodes
for i in range(ndim*nodes):
    uflatindex =gcon[i,2]-1
    u_flat[i] = u[uflatindex]

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


print("U",u_flat)
print("N", Nbar)
print("Fext", Fext)