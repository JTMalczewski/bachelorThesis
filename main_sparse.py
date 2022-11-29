import numpy as np
import matplotlib.pyplot as plt
# from numba import njit
from fun import *
import scipy.sparse.linalg as sp
from scipy.sparse import csc_matrix
import pandas as pd

# @njit
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
            temp_var.append(-2/dx**2 -2/dy**2 + k**2/n[i,j]**2)

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

Ny = 81
Nx = 81
dx = 0.001
dy = 0.001
k = 6.28/0.012
n = np.ones((Nx,Ny))
f = np.zeros((Nx*Ny))
E_2D = np.zeros((Nx,Ny))

f[(Nx*Ny)//2] = 1

var, coordinates_x, coordinates_y = generateMatrix(Nx,Ny,dx,dy,n,k)
matrix_sparse = csc_matrix((var,(coordinates_x, coordinates_y)))
matrix_inv_sparse = sp.inv(matrix_sparse)
E_1D = matrix_inv_sparse.dot(f)
E_2D = np.reshape(E_1D,(Nx,Ny))

f = open("data_2.txt", "a")
for i in range(Nx):
    f.write(str(E_2D[i]))
f.close()


# plt.imshow(E_2D, 'gist_heat')
# plt.colorbar()
# plt.savefig('E_{1:d}_{1:d}_mm.png'.format(Nx,Ny))

