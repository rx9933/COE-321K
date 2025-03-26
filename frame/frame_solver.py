# 10/31
## 4 inputs
import numpy as np
from math import sqrt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os




### PRE PROCESSING ###


loc = "hw5/"
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
neles = int(element_data[0])
element_info = [list(map(float, line.split())) for line in element_data[1:]]
ele = np.array(element_info)[:,:]
I = np.array(element_info)[:,-1]
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
import numpy as np

def compute_K_matrix(EA, EI, L, phi):
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
    cos_phi = phi[0]
    sin_phi = phi[1]
    # Constructing the 6x6 stiffness matrix
    K = np.array([
        [(EA * cos_phi**2) / L + (12 * EI * sin_phi**2) / L**3, -((12 * EI * cos_phi * sin_phi) / L**3) + (EA * cos_phi * sin_phi) / L, -((6 * EI * sin_phi) / L**2), -((EA * cos_phi**2) / L) - (12 * EI * sin_phi**2) / L**3, (12 * EI * cos_phi * sin_phi) / L**3 - (EA * cos_phi * sin_phi) / L, -((6 * EI * sin_phi) / L**2)],
        [-((12 * EI * cos_phi * sin_phi) / L**3) + (EA * cos_phi * sin_phi) / L, (12 * EI * cos_phi**2) / L**3 + (EA * sin_phi**2) / L, (6 * EI * cos_phi) / L**2, (12 * EI * cos_phi * sin_phi) / L**3 - (EA * cos_phi * sin_phi) / L, -((12 * EI * cos_phi**2) / L**3) - (EA * sin_phi**2) / L, (6 * EI * cos_phi) / L**2],
        [-((6 * EI * sin_phi) / L**2), (6 * EI * cos_phi) / L**2, (4 * EI) / L, (6 * EI * sin_phi) / L**2, -((6 * EI * cos_phi) / L**2), (2 * EI) / L],
        [-((EA * cos_phi**2) / L) - (12 * EI * sin_phi**2) / L**3, (12 * EI * cos_phi * sin_phi) / L**3 - (EA * cos_phi * sin_phi) / L, (6 * EI * sin_phi) / L**2, (EA * cos_phi**2) / L + (12 * EI * sin_phi**2) / L**3, -((12 * EI * cos_phi * sin_phi) / L**3) + (EA * cos_phi * sin_phi) / L, (6 * EI * sin_phi) / L**2],
        [(12 * EI * cos_phi * sin_phi) / L**3 - (EA * cos_phi * sin_phi) / L, -((12 * EI * cos_phi**2) / L**3) - (EA * sin_phi**2) / L, -((6 * EI * cos_phi) / L**2), -((12 * EI * cos_phi * sin_phi) / L**3) + (EA * cos_phi * sin_phi) / L, (12 * EI * cos_phi**2) / L**3 + (EA * sin_phi**2) / L, -((6 * EI * cos_phi) / L**2)],
        [-((6 * EI * sin_phi) / L**2), (6 * EI * cos_phi) / L**2, (2 * EI) / L, (6 * EI * sin_phi) / L**2, -((6 * EI * cos_phi) / L**2), (4 * EI) / L]
    ])

    return K




# Initialize length for bars, cosine/direction matrix
LO = []  
dircos = np.zeros((neles, ndim))
Kele = np.zeros((neles, ndpn*ndim, ndpn*ndim))
ndofs = totndofs - ndbcs  # Active DOFs only
# print("ndofs", ndofs)
# print("ndbcs", ndbcs)
# print("nfbcs", nfbcs)
# print("nodes", nodes)
# print("ndpn", ndpn)
# print("ndim", ndim)
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
    lsquared = sum((node[int(ele[i][2])-1][j+1] - node[int(ele[i][1])-1][j+1])**2 for j in range(ndim))
    LO.append(np.sqrt(lsquared)) 
    
    # # Compute direction cosines

    for j in range(ndim):
        diff = node[int(ele[i][2])-1][j+1] - node[int(ele[i][1])-1][j+1]
        dircos[i][j] = diff / LO[i]
    Kele[i] = compute_K_matrix(ele[i][3] * ele[i][4] , ele[i][3] * I[i], LO[i], dircos[i]) # EA, EI, L, phi
    
    ###########

# construct global stiffness (reduced): Kred + construct final RHS: Fk - K2*uk
row_idx, col_idx, values = [], [], []
for i in range(neles):  
    node1, node2 = ele[i][1], ele[i][2]
    glist = [0] * (ndpn * ndim)  # Stores global DOF indices for this element

    for row in range(ndpn * nodes):
        if gcon[row][0] in {node1, node2}:
            for dim in range(1, ndpn + 1):
                if gcon[row][0] == node1 and gcon[row][1] == dim:
                    glist[dim - 1] = gcon[row][2] - 1
                elif gcon[row][0] == node2 and gcon[row][1] == dim:
                    glist[ndpn + dim - 1] = gcon[row][2] - 1
    for row in range(ndpn * ndim):
        if glist[row] < ndofs:  # Only for unknown DOFs
            for col in range(ndpn * ndim):
                if glist[col] >= ndofs:  # Only for known DOFs
                    Fred[glist[row]] -= Kele[i][row, col] * dbcval[glist[col] - ndofs]
    

    glist = np.atleast_1d(glist) 

    for row in range(len(glist)):
        for col in range(len(glist)):
            l1index = row
            l2index = col
            value = Kele[i][l1index, l2index]
            if glist[row]  < ndofs and glist[col] < ndofs:# and abs(value) > 1e-8:  # Avoid storing explicit zeros
                Kred[glist[row],glist[col]] += value






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
    
    Nbar[i] = ele[i][3] * ele[i][4] * (u2a-u1a)
    V[i]  = 12 * ele[i][3] * ele[i][5] / LO[i]**3 * (u1t - u2t) + 6 * ele[i][3] * ele[i][5] / LO[i]**2 * (theta1 + theta2)
    M1[i] = 6 * ele[i][3] * ele[i][5] / LO[i]**2 * (u2t - u1t) - 2 * ele[i][3] * ele[i][5] / LO[i] * (2* theta1 + theta2)
    M2[i] = -6 * ele[i][3] * ele[i][5] / LO[i]**2 * (u2t - u1t) + 2 * ele[i][3] * ele[i][5] / LO[i] * (theta1 + 2*theta2)


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
