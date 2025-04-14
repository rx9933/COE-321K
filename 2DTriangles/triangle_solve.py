
import numpy as np
import os




### PRE PROCESSING ###

 
loc = "testcase/"
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
ndim = 2#int(node_data[0])
node = [list(map(float, line.split())) for line in node_data[1:]]
nodes = np.shape(np.array(node))[0]
# nodes = int(node[0][0])
# node = node[1:]

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
firstrow = element_data[0].split(" ")[::2]
neles = int(firstrow[0])
E = float(firstrow[1])
v = float(firstrow[2])
element_info = [list(map(float, line.split())) for line in element_data[1:]]
ele = np.array(element_info)[:,:]
ndpn = ndim +1


gconold = np.array([[node, dof, ndpn * (node - 1) + dof] for node in range(1, nodes + 1) for dof in range(1, ndpn + 1)])
totndofs = ndpn * nodes # total dofs

gcon = gconold.copy()
for ndbcNum in range(ndbcs):#ndbcs):#3
    for row in range(ndpn*nodes): #8
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

C = np.array([[E/(1-v**2), v*E /(1-v**2), 0],[v*E /(1-v**2), E /(1-v**2), 0], [0, 0, E/(2*(1+v))]])


def compute_K_matrix(elenum):
    """
    Compute the 6x6 stiffness matrix K[i] for a beam element.
    
    Parameters:
    EA  - Axial rigidity
    EI  - Flexural rigidity
    L   - Element length
    phi - Angle (in radians)
    
    Returns:
    6x6 numpy array representing the stiffness matrix.
    
    """
    print("element",ele[elenum])
    element_nodes = ele[elenum][1:]
    print("nodes", node)
    x1,y1 = node[int(element_nodes[0]-1)][1:]
    x2,y2 = node[int(element_nodes[1]-1)][1:]
    x3,y3 = node[int(element_nodes[2]-1)][1:]
    print("all nodes", x1, y1, x2, y2, x3, y3)
    A = 1/2 * np.linalg.det(np.array([[x2-x1, y2-y1, 0], [x3-x1, y3-y1, 0], [0, 0, 1]]))
    N1x = 1/(2*A) * (y2-y3)
    N1y = 1/(2*A) * (x3-x2)

    N2x = 1/(2*A) * (y3-y1)
    N2y = 1/(2*A) * (x1-x3)

    N3x = 1/(2*A) * (y1-y2)
    N3y = 1/(2*A) * (x2-x1)
    B = np.array([[N1x, 0, N2x, 0, N3x, 0], [0, N1y, 0, N2y, 0, N3y], [N1y, N1x, N2y, N2x, N3y, N3x]])
    # Constructing the 6x6 stiffness matrix
    print(A)
    K = A * B.T @ C @ B
    print(C)
    # print( K)
    # K =np.ones((ndpn, 2*ndpn))
    assert (K.T == K).all
    return K




# Initialize length for bars, cosine/direction matrix
LO = []  
dircos = np.zeros((neles, ndim))
Kele = np.zeros((neles, ndpn*ndim, ndpn*ndim))
ndofs = totndofs - ndbcs  # Active DOFs only
print("ndofs", ndofs)
print("ndbcs", ndbcs)
print("nfbcs", nfbcs)
print("nodes", nodes)
print("ndpn", ndpn)
print("ndim", ndim)
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
ele =  np.column_stack((ele[:, :2].astype(int), ele[:, 2:]))

# Compute element stiffness matrix
Kred = np.zeros((ndofs, ndofs))  
for i in range(neles):
    Kele[i] = compute_K_matrix(i) # EA, EI, L, phi

    ###########

# construct global stiffness (reduced): Kred + construct final RHS: Fk - K2*uk
row_idx, col_idx, values = [], [], []
for i in range(neles):
    # Get the three nodes for this triangular element
    node1, node2, node3 = int(ele[i][1]), int(ele[i][2]), int(ele[i][3])
    
    # Initialize global DOF list (6 DOFs: u1,v1, u2,v2, u3,v3)
    glist = np.zeros(6, dtype=int)
    
    # Map each node's DOFs to global DOF numbers
    for row in range(ndpn * nodes):
        node_num = gcon[row, 0]
        dof_num = gcon[row, 1]
        
        if node_num == node1:
            if dof_num == 1: glist[0] = gcon[row, 2] - 1  # u1
            if dof_num == 2: glist[1] = gcon[row, 2] - 1  # v1
        elif node_num == node2:
            if dof_num == 1: glist[2] = gcon[row, 2] - 1  # u2
            if dof_num == 2: glist[3] = gcon[row, 2] - 1  # v2
        elif node_num == node3:
            if dof_num == 1: glist[4] = gcon[row, 2] - 1  # u3
            if dof_num == 2: glist[5] = gcon[row, 2] - 1  # v3
    print(glist)
    # Apply boundary conditions (modify RHS for fixed DOFs)
    for row in range(ndim*ndpn):  # 6 DOFs per element
        if glist[row] < ndofs:  # This is a free DOF
            for col in range(ndim*ndpn):
                if glist[col] >= ndofs:  # This is a fixed DOF
                    # Find which boundary condition this corresponds to
                    bc_index = glist[col] - ndofs
                    Fred[glist[row]] -= Kele[i][row, col] * dbcval[bc_index]
    
    # Assemble the global stiffness matrix
    for row in range(ndim*ndpn):
        for col in range(ndim*ndpn):
            if glist[row] < ndofs and glist[col] < ndofs:  # Both DOFs are free
                Kred[glist[row], glist[col]] += Kele[i][row, col]

u_red = np.linalg.solve(Kred, Fred)
# Append prescribed displacements
u = np.concatenate([u_red, dbcval])

u_flat = [0]*ndpn*nodes
for i in range(ndpn*nodes):
    uflatindex =gcon[i,2]-1
    u_flat[i] = u[uflatindex]

u_2d = np.array(u_flat).reshape((int(len(u_flat)/ndpn), ndpn))

# Calculate internal forces Nbar using Bele and local displacements (u_local)
Nbar, V, M1, M2 = np.zeros(neles), np.zeros(neles), np.zeros(neles), np.zeros(neles)  

for i in range(neles):

    node1ind, node2ind = int(ele[i][1]-1), int(ele[i][2]-1)
    theta1 = u_2d[node1ind][-1].item()
    theta2 = u_2d[node2ind][-1].item()
    u1v1 = u_2d[node1ind][:2]
    u2v2 = u_2d[node2ind][:2]
    u1a = np.dot(u1v1,dircos[i])
    u1t = np.dot(np.array((-u1v1[0], u1v1[1])),dircos[i][::-1])

    u2a = np.dot(u2v2,dircos[i])
    u2t = np.dot(np.array((-u2v2[0], u2v2 [1])),dircos[i][::-1])
    
    Nbar[i] = ele[i][3] * ele[i][4] * (u2a-u1a) * 1/LO[i]
    V[i]  = 12 * ele[i][3] * ele[i][5] / LO[i]**3 * (u1t - u2t) + 6 * ele[i][3] * ele[i][5] / LO[i]**2 * (theta1 + theta2)
    M1[i] = 6 * ele[i][3] * ele[i][5] / LO[i]**2 * (u2t - u1t) - 2 * ele[i][3] * ele[i][5] / LO[i] * (2* theta1 + theta2)
    M2[i] = 6 * ele[i][3] * ele[i][5] / LO[i]**2 * (u1t - u2t) + 2 * ele[i][3] * ele[i][5] / LO[i] * (theta1 + 2*theta2)


ele = np.array(ele)
Fext = np.zeros((nodes, ndpn))
for i in range(neles):
    node1ind, node2ind = int(ele[i][1]-1), int(ele[i][2]-1)
    Fext[node1ind, 0] += np.dot([Nbar[i], V[i]], -dircos[i])
    Fext[node2ind, 0] += np.dot([Nbar[i], V[i]], dircos[i])

    Fext[node1ind, 1] += np.dot([Nbar[i], V[i]], np.array([-dircos[i][1], dircos[i][0]]))
    Fext[node2ind, 1] += np.dot([Nbar[i], V[i]], np.array([dircos[i][1], -dircos[i][0]]))
    
    Fext[node1ind, 2] += -M1[i]
    Fext[node2ind, 2] += M2[i]

###################
## Post processing
write_file = loc + 'output.txt'

with open(write_file, 'w') as f:
    # Writing nodal displacements
    f.write("Nodal displacements\n")
    f.write("node#      x          y          u          v       theta\n")
    for i in range((u_2d.shape)[0]):
        f.write(f"{i+1:5d}  {node[i][1]:10.6f}  {node[i][2]:10.6f}  {u_2d[i,0]:10.6f}  {u_2d[i,1]:10.6f}  {u_2d[i,2]:10.6f}\n")

    f.write("\nExternal forces\n")
    f.write("node#      x          y         Fx         Fy          M\n")
    for i in range(len(Fext)):
        f.write(f"{i+1:5d}  {node[i][1]:10.6f}  {node[i][2]:10.6f}  {Fext[i,0]:10.6f}  {Fext[i,1]:10.6f}  {Fext[i,2]:10.6f}\n")

    f.write("\nElement forces and moments\n")
    f.write("ele#       N          V         M1         M2\n")
    for i in range(len(Nbar)):
        f.write(f"{i+1:5d}  {Nbar[i]:10.6f}  {V[i]:10.6f}  {M1[i]:10.6f}  {M2[i]:10.6f}\n")

print(f"Results written to {write_file}")

