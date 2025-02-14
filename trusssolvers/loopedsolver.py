# 10/31

import numpy as np
from math import sqrt

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

with open("../testingtruss/1.txt", 'r') as file: # 2d or 3d input text file
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

# Initialize length for bars, cosine/direction matrix, 
LO = []  
dircos = np.zeros((neles, ndim))
Bele = np.zeros((neles, 2*ndim))
Kele = np.zeros((neles, 2*ndim, 2*ndim))
Kglobal = np.zeros((ndim*nodes, ndim*nodes)) #final stiffness matrix 

for i in range(neles): # for each bar
    lsquared = 0.0
    for j in range(ndim):
        diff = node[ele[i][2]-1][j+1] - node[ele[i][1]-1][j+1]
        lsquared += diff**2
    LO.append(lsquared**(1/2)) 
    for j in range(ndim): # for each dimension
        diff = node[ele[i][2]-1][j+1] - node[ele[i][1]-1][j+1]
        dircos[i][j] =diff/LO[i]
        Bele[i][j] = -dircos[i][j]
        Bele[i][j+ndim] = -Bele[i][j]

    Kele[i] = ele[i][3]*ele[i][4]/LO[i] * np.transpose(Bele[i]).reshape(-1, 1)*Bele[i]
 
    node1 = ele[i][1]
    node2 = ele[i][2]
    glist = [0] * 2*ndim # gcon values for particular element -- dictates location in Kglobal

    for row in range(ndim * nodes):
        if gcon[row][0] == int(node1) or gcon[row][0] == int(node2):
            for dim in range(1, ndim + 1):
                if gcon[row][0] == int(node1) and gcon[row][1] == dim:
                    glist[dim - 1] = gcon[row][2]
                elif gcon[row][0] == int(node2) and gcon[row][1] == dim:
                    glist[ndim + dim - 1] = gcon[row][2]

    # move each Kele element into global stiffness matrix
    for row in range(len(glist)):
        for col in range(len(glist)):
            Kglobal[glist[row]-1][glist[col]-1]+=Kele[i][row][col]

ndofs = totndofs - ndbcs #active number of degrees of freedom
Kred = Kglobal[:ndofs, :ndofs] #reduced/portion of global stiffness matrix
Fred = np.zeros(ndim*nodes) #all forces minus K*x 

# Set up force array
for i in range(nfbcs):
    for row in range(ndim * nodes):
        if gcon[row][0] == fnode[i] and gcon[row][1] == fdof[i]:
            gdofi = gcon[row][2]
            Fred[gdofi - 1] = fval[i]
            
# Apply displacement conditions 
for i in range(ndbcs):
    for row in range(ndofs):
        if gcon[row][0] == dbcnd[i] and gcon[row][1] == dbcdof[i]:
            gdofi = gcon[row][2]
            Fred -= Kglobal[:, gdofi - 1] * dbcval[i]

# Solve for displacements
# Kinverse = np.linalg.inv(Kred)
u = np.linalg.solve(Kred, Fred[:ndofs]) 
u=np.append(u, dbcval) # add on remaining displacements(per input file), in same order; final displacement list

print(u)


