import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def generateMatrix(N,dx,n,k):
    temp_x = []
    temp_y = []
    temp_var = []

    for i in range(N):
        temp_x.append(i)
        temp_y.append(i)
        temp_var.append(-2/(dx**2) + k**2*n[i]**2)

    for i in range(1,N):
        temp_x.append(i-1)
        temp_y.append(i)
        temp_var.append(1/dx**2)
    
    for i in range(N-1):
        temp_x.append(i+1)
        temp_y.append(i)
        temp_var.append(1/dx**2)

    return temp_var, temp_x, temp_y


def addWall(n,wall,GHz,material):
    a,b,c,d = material
    for i in range(len(wall)):
        n[wall[i]] = np.sqrt(a*GHz**b - 1j*17.98*c*GHz**d/GHz)
    return n

concrete =  (5.24,  0,  0.0462, 0.7822)
brick =     (3.91,  0,  0.0238, 0.16)
wood =      (1.99,  0,  0.0047, 1.0718)
glass =     (6.31,  0,  0.0036, 1.3394)
metal =     (1,     0,  1e7,    0)

# X = 1 #m
# # N = 50
# E_0 = 20 #V/m
# dx = 0.001 # co ile
# N = int(X/dx) +1
# k = 6.28/0.125 #2pi/długość fali
# matrix = np.zeros((N,N),dtype=np.csingle)
# n = np.ones(N,dtype=np.csingle)
# f = np.zeros(N,dtype=np.csingle)

# f[N//2] = 10
# n = addWall(n,range(200),2.4,glass)
# n = addWall(n,range(-200,0),2.4,glass)

# var, coordinates_x, coordinates_y = generateMatrix(N,dx,n,k)
# matrix_sparse = csc_matrix((var,(coordinates_x, coordinates_y)))
# E_1D = spsolve(matrix_sparse,f)


# plt.style.use('ggplot')
# plt.figure(figsize=(10,5))
# plt.plot(np.imag(E_1D), label='Częśc urojona symulowanej fukcji falowej')
# plt.plot(np.real(E_1D), label='Częśc rzeczywistej symulowanej fukcji falowej')
# plt.yticks(np.arange(-1e-4, 1.1e-4, 5e-5))
# plt.title('Ψ', fontsize=20)
# plt.xlabel('Symulowana przestrzeń 1D [mm]')
# plt.ylabel('real(Ψ) lub imag(Ψ) [-]')
# ax = plt.gca()
# fig = plt.gcf()
# ax.ticklabel_format(axis='y',useMathText=True)
# ax.legend(bbox_to_anchor=[0.02, 0.05], loc='lower left')
# plt.savefig('./1D/test', dpi=600)

def printBoth(E_1D, material, GHz):
    plt.style.use('ggplot')
    plt.figure(figsize=(10,5))
    plt.plot(np.imag(E_1D), label='Część urojona symulowanej fukcji falowej')
    plt.plot(np.real(E_1D), label='Część rzeczywistej symulowanej fukcji falowej')
    plt.yticks(np.arange(-1e-4, 1.1e-4, 5e-5))
    plt.title('Ψ dla f={:1} GHz oraz barier na brzegach - {}'.format(GHz, material), fontsize=15)
    plt.xlabel('Symulowana przestrzeń 1D [mm]')
    plt.ylabel('real(Ψ), imag(Ψ) [-]')
    ax = plt.gca()
    fig = plt.gcf()
    ax.ticklabel_format(axis='y',useMathText=True)
    ax.legend(bbox_to_anchor=[0.02, 0.05], loc='lower left')
    plt.savefig('./1D_{:1}/barier_20_20_{}'.format(GHz,material), dpi=600)

def generateGraph(GHz, material, X, E_0, dx):
    match material:
        case "beton":
            temp = concrete 
        case "cegła":
            temp = brick
        case "drewno":
            temp = wood
        case "szkło":
            temp = glass
        case "metal":
            temp = metal
        case _:
            temp = 0

    N = int(X/dx) +1
    lenght = 299792458/(GHz * 1e9)
    k = np.pi*2/lenght                          #2pi/długość fali
    n = np.ones(N,dtype=np.csingle)
    f = np.zeros(N,dtype=np.csingle)

    f[N//2] = E_0
    n = addWall(n,range(200),GHz,temp)
    n = addWall(n,range(-200,0),GHz,temp)

    var, coordinates_x, coordinates_y = generateMatrix(N,dx,n,k)
    matrix_sparse = csc_matrix((var,(coordinates_x, coordinates_y)))
    E_1D = spsolve(matrix_sparse,f)

    printBoth(E_1D, material, GHz)


generateGraph(2.4, "beton", 1, 10, 0.001)
generateGraph(2.4, "drewno", 1, 10, 0.001)
generateGraph(2.4, "cegła", 1, 10, 0.001)
generateGraph(2.4, "metal", 1, 10, 0.001)
generateGraph(2.4, "szkło", 1, 10, 0.001)

generateGraph(5.2, "beton", 1, 10, 0.001)
generateGraph(5.2, "drewno", 1, 10, 0.001)
generateGraph(5.2, "cegła", 1, 10, 0.001)
generateGraph(5.2, "metal", 1, 10, 0.001)
generateGraph(5.2, "szkło", 1, 10, 0.001)


# plt.style.use('ggplot')
# plt.figure(figsize=(10,5))
# plt.plot(np.imag(E_1D), label='Część urojona symulowanej fukcji falowej')
# plt.plot(np.real(E_1D), label='Część rzeczywistej symulowanej fukcji falowej')
# plt.yticks(np.arange(-1e-4, 1.1e-4, 5e-5))
# plt.title('Ψ, bariery na brzegach - {}'.format('glass'), fontsize=15)
# plt.xlabel('Symulowana przestrzeń 1D [mm]')
# plt.ylabel('real(Ψ) lub imag(Ψ) [-]')
# ax = plt.gca()
# fig = plt.gcf()
# ax.ticklabel_format(axis='y',useMathText=True)
# ax.legend(bbox_to_anchor=[0.02, 0.05], loc='lower left')
# plt.savefig('./1D_2.4/barier_20_20_{}'.format('glass'), dpi=600)
