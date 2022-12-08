import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def placeSource(X,Y,f,power):
    Y = Ny - Y
    temp = int(Ny*X + Y)
    f[temp] = power
    return f

def addContour(material,thickness,GHz,n):
    radius = thickness//2

    wall = generateWall(0,radius//2,int(Nx),radius,True)
    n = addWall(n,wall,2.4,material)
    printWall(wall)

    wall = generateWall(0,Ny-radius//2,int(Nx),radius,True)
    n = addWall(n,wall,2.4,material)
    printWall(wall)

    wall = generateWall(radius//2,0,int(Ny),radius,False)
    n = addWall(n,wall,2.4,material)
    printWall(wall)

    wall = generateWall(Nx-radius//2,0,int(Ny),radius,False)
    n = addWall(n,wall,2.4,material)
    printWall(wall)
    return n

def printWall(wall):
    return
    # plt.scatter(np.array(wall).T[0],np.array(wall).T[1], color='white', alpha=0.5)

def generateWall(startPointX, startPointY, length, thickness, xAxes):
    wall = []
    if xAxes:
        for i in range(length):
            for j in range(thickness):
                wall.append([
                    startPointX+i,
                    int(startPointY-thickness/2+j)
                ])
    else:
        for i in range(length):
            for j in range(thickness):
                wall.append([
                    int(startPointX-thickness/2+j),
                    startPointY+i
                ])
    return wall

def addWall(n,wall,GHz,material):
    a,b,c,d = material
    for i in range(len(wall)):
        n[wall[i][0],wall[i][1]] = np.sqrt(a*GHz**b - 1j*17.98*c*GHz**d/GHz)
    return n

def generateMatrix(Nx,Ny,dx,dy,n,k):
    def idx1D(i , j, N):
        return i * N + j

    temp_x = []
    temp_y = []
    temp_var = []

    for i in range(Nx):
        for j in range(Ny):
            temp_x.append(idx1D(i, j, Ny))
            temp_y.append(idx1D(i, j, Ny))
            temp_var.append(-2/dx**2 -2/dy**2 + (k**2)*n[i,j]**2)

    for i in range(Nx):
        for j in range(1,Ny):
            temp_x.append(idx1D(i, j, Ny))
            temp_y.append(idx1D(i, j-1, Ny))
            temp_var.append(1/dy**2)

    for i in range(Nx):
        for j in range(Ny-1):
            temp_x.append(idx1D(i, j, Ny))
            temp_y.append(idx1D(i, j+1, Ny))
            temp_var.append(1/dy**2)

    for i in range(Nx-1):
        for j in range(Ny):
            temp_x.append(idx1D(i, j, Ny))
            temp_y.append(idx1D(i+1, j, Ny))
            temp_var.append(1/dx**2)

    for i in range(1,Nx):
        for j in range(Ny):
            temp_x.append(idx1D(i, j, Ny))
            temp_y.append(idx1D(i-1, j, Ny))
            temp_var.append(1/dx**2)

    return temp_var, temp_x, temp_y

Ny = 401 #cm
Nx = 1201 #cm
dx = 0.005 #m
dy = 0.005 #m
k = 6.28/0.12
n = np.ones((Nx,Ny),dtype=np.csingle)
f = np.zeros((Nx*Ny),dtype=np.csingle)
E_2D = np.zeros((Nx,Ny),dtype=np.csingle)

concrete =  (5.24,  0,  0.0462, 0.7822)
brick =     (3.91,  0,  0.0238, 0.16)
wood =      (1.99,  0,  0.0047, 1.0718)
glass =     (6.31,  0,  0.0036, 1.3394)
metal =     (1,     0,  1e7,    0)

placeSource(Nx//2,Ny//2,f,1000)
placeSource(Nx//2+1,Ny//2,f,1000)
placeSource(Nx//2,Ny//2+1,f,1000)
placeSource(Nx//2+1,Ny//2+1,f,1000)

addContour(concrete,10,2.4,n)

wall = generateWall(Nx//3,0,Ny//3*2,5,False)
n = addWall(n,wall,2.4,concrete)
printWall(wall)

wall = generateWall(Nx//3*2,Ny//3,Ny//3*2,5,False)
n = addWall(n,wall,2.4,concrete)
printWall(wall)

var, coordinates_x, coordinates_y = generateMatrix(Nx,Ny,dx,dy,n,k)
matrix_sparse = csc_matrix((var,(coordinates_x, coordinates_y)))
E_1D = spsolve(matrix_sparse,f)
E_2D = np.reshape(E_1D,(Nx,Ny))

# f = open("data_2.txt", "a")
# for i in range(Nx):
#     f.write(str(E_2D[i]))
# f.close()

plt.imshow((abs(E_2D)).T)
# plt.colorbar()
plt.xlabel('x [cm]')
plt.ylabel('y [cm]')
plt.title('abs($\Psi$) dla dx={}'.format(dx*100))
plt.savefig('E_{}x{}cm_dx_{}cm.png'.format(Nx,Ny,dx*100))


