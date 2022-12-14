import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def placeSource(X,Y,f,power):
    Y = Ny - Y
    temp = int(Ny*X + Y)
    f[temp] = power
    return f

def addContour(material,thickness,GHz,n):
    radius = thickness//2

    wall = generateWall(0,radius//2,int(Nx),radius,True)
    n = addWall(n,wall,GHz,material)
    walls = wall

    wall = generateWall(0,Ny-radius//2,int(Nx),radius,True)
    n = addWall(n,wall,GHz,material)
    walls = np.concatenate((walls,wall))

    wall = generateWall(radius//2,0,int(Ny),radius,False)
    n = addWall(n,wall,GHz,material)
    walls = np.concatenate((walls,wall))

    wall = generateWall(Nx-radius//2,0,int(Ny),radius,False)
    n = addWall(n,wall,GHz,material)
    walls = np.concatenate((walls,wall))

    return n, walls

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

def printAbs(E_2D,GHz,X,Y,material):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow((np.abs(E_2D)).T, cmap=newcmp, extent=[-X//2, X//2, -Y//2, Y//2])
    plt.xlabel('x [m]', fontsize=10)
    plt.ylabel('y [m]', fontsize=10)
    plt.title('|??| dla f={:1} GHz, {}, dx={}'.format(GHz,material,dx), fontsize=13)
    plt.xlim(-1.75,1.75)
    plt.ylim(-1.75,1.75)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return ax

def calculateFild(Nx,Ny,dx,dy,n,k,f):
    var, coordinates_x, coordinates_y = generateMatrix(Nx,Ny,dx,dy,n,k)
    matrix_sparse = csc_matrix((var,(coordinates_x, coordinates_y)))
    
    plt.imshow(abs(matrix_sparse.toarray()),'binary')
    plt.colorbar()
    plt.title('Macierz uk??adu r??wna?? w 2D dla $N_x = 6$')
    plt.xlabel('Index macierzy [-]')
    plt.ylabel('Index macierzy [-]')
    plt.savefig('./macierz_uk??adu_r??wna??_2D.png', dpi=300)

    # E_1D = spsolve(matrix_sparse,f)
    # E_2D = np.reshape(E_1D,(Nx,Ny))
    # return E_2D

def printWalls(walls,dx,ax):
    print('no walls')
    # ax.scatter(np.array(walls).T[0]*dx - X//2,-(np.array(walls).T[1]*dx - Y//2), color='cornflowerblue', marker=',',lw=0, s=1)

def saveGraph(GHz,X,Y,dx,material,room):
    plt.savefig('./2D_{:1}/abs_{:1}_{:1}_{:3}_{}_{}.png'.format(GHz,X,Y,dx,material,room),dpi=300)

X = 4 #m
Y = 4 #m
dx = 0.01 #m
dy = 0.01 #m
# Ny = int(Y/dy)
# Nx = int(X/dx)
Ny = 6
Nx = 6
k = 6.28/0.125
n = np.ones((Nx,Ny),dtype=np.csingle)
f = np.zeros((Nx*Ny),dtype=np.csingle)
E_2D = np.zeros((Nx,Ny),dtype=np.csingle)

concrete =  (5.24,  0,  0.0462, 0.7822)
brick =     (3.91,  0,  0.0238, 0.16)
wood =      (1.99,  0,  0.0047, 1.0718)
glass =     (6.31,  0,  0.0036, 1.3394)
metal =     (1,     0,  1e7,    0)

binary = cm.get_cmap('bwr', 128)
newcolors = binary(np.linspace(0, 1, 128))
for i in range(len(newcolors)//2):
    newcolors[i] = [
        (2*i/128)**0.5,
        (2*i/128)**0.5,
        (2*i/128)**0.5,
        1]
newcmp = colors.ListedColormap(newcolors)

material = 'beton'
room = 'box'
GHz = 2.4

match material:
    case "beton":
        material_factors = concrete 
    case "ceg??a":
        material_factors = brick
    case "drewno":
        material_factors = wood
    case "szk??o":
        material_factors = glass
    case "metal":
        material_factors = metal
    case _:
        material_factors = 0

# f = placeSource(3*Nx//8,3*Ny//8,f,100000)
# n, walls = addContour(material_factors,Nx//4,GHz,n)

# wall = generateWall(0,Ny//4,Nx//2,Ny//2,True)
# n = addWall(n,wall,2.4,concrete)
# walls = np.concatenate((walls,wall))


E_2D = calculateFild(Nx,Ny,dx,dy,n,k,f)
# ax = printAbs(E_2D,GHz ,X ,Y, material)

# printWalls(walls,dx,ax)
# saveGraph(GHz,X,Y,dx,material,room)

# generateWall(startPointX, startPointY, length, thickness, xAxes):

