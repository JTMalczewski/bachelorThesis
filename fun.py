import numpy as np
from numba import njit

def idx1D(i , j, N):
        return i * N + j

@njit
def generateMatrix(Nx,Ny,dx,dy,n,k):
    def idx1D(i , j, N):
        return i * N + j

    matrix = np.zeros((Nx**2,Ny**2))

    for i in range(Nx):
        for j in range(Ny):
            matrix[idx1D(i, j, Nx), idx1D(i, j, Ny)] = -2/dx**2 -2/dy**2 + k**2/n[i,j]**2

    for i in range(Nx):
        for j in range(1,Ny):
            matrix[idx1D(i, j, Nx), idx1D(i, j - 1, Ny)] = 1/dy**2

    for i in range(Nx):
        for j in range(Ny-1):
            matrix[idx1D(i, j, Nx), idx1D(i, j + 1, Ny)] = 1/dy**2

    for i in range(Nx-1):
        for j in range(Ny):
            matrix[idx1D(i, j, Nx), idx1D(i + 1, j, Ny)] = 1/dx**2

    for i in range(1,Nx):
        for j in range(Ny):
            matrix[idx1D(i, j, Nx), idx1D(i - 1, j, Ny)] = 1/dx**2
    return matrix

@njit
def to2D(E,Nx ,Ny):
    E_2D = np.zeros((Nx,Ny))
    for j in range(np.sqrt(len(E))):
        E_2D[j] = E[j*Nx:(j+1)*Nx]
    return E_2D

# @njit
def generateFild(Nx,Ny,dx,dy,n,k,f):
    matrix = generateMatrix(Nx,Ny,dx,dy,n,k)
    E = np.linalg.solve(matrix, f)
    E_2D = to2D(E,Nx,Ny)
    return E_2D

@njit
def pinv(matrix):
    return np.linalg.pinv(matrix) 

# @njit
def matmul(M1,M2):
    return np.matmul(M1, M2)