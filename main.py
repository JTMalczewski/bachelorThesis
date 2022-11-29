import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from fun import *
import scipy.sparse.linalg as sp
from scipy.sparse import csc_matrix

@njit
def generateMatrix(Nx,Ny,dx,dy,n,k):
    def idx1D(i , j, N):
        return i * N + j

    temp = np.zeros((Nx*Ny,Ny*Nx))

    for i in range(Nx):
        for j in range(Ny):
            temp[idx1D(i, j, Ny), idx1D(i, j, Ny)] = -2/dx**2 -2/dy**2 + k**2/n[i,j]**2

    for i in range(Nx):
        for j in range(1,Ny):
            temp[idx1D(i, j, Ny), idx1D(i, j - 1, Ny)] = 1/dy**2

    for i in range(Nx):
        for j in range(Ny-1):
            temp[idx1D(i, j, Ny), idx1D(i, j + 1, Ny)] = 1/dy**2

    for i in range(Nx-1):
        for j in range(Ny):
            temp[idx1D(i, j, Ny), idx1D(i + 1, j, Ny)] = 1/dx**2

    for i in range(1,Nx):
        for j in range(Ny):
            temp[idx1D(i, j, Ny), idx1D(i - 1, j, Ny)] = 1/dx**2
    return temp

@njit
def pinv(matrix):
    return np.linalg.pinv(matrix) 

def matmul(M1,M2):
    return np.matmul(M1, M2)

x = 21
y = 41

print('Plese write box shape:\nx: ')
x = int(input())
print('y: ')
y = int(input())

print('Any border?\nfrom y:')
border_start = int(input())
print('what with?')
border_with = int(input())
print('How long?')
border_long = int(input())
print('a value:')
border_value = int(input())



Ny = y if y > x else x
Nx = x if y > x else y
dx = 0.001
dy = 0.001
k = 6.28/0.012 #2pi/długość fali
n = np.ones((Nx,Ny))
f = np.zeros((Nx*Ny))
f[(Nx*Ny)//2] = 1
E_2D_test = np.zeros((Nx,Ny))

for i in range(border_long):
    for j in range(border_start, border_start + border_with):
        n[i,j] = border_value

print('generating matrix...')
matrix = generateMatrix(Nx,Ny,dx,dy,n,k)

print('pseudoinverting matrix...')
matrix_inv = pinv(matrix)         # Pseudoinverse
# matrix_inv = sp.inv(
#     csc_matrix((
#         np.reshape(matrix,(1,Nx*Ny*Ny*Nx))[0],
#         range(Nx*Ny*Ny*Nx),
#         range(Nx*Ny*Ny*Nx)),
#         dtype=np.int8
#     ))

print('reshaping matrix...')
print(np.shape(matrix_inv))

print('multiplication...')
E_test = matmul(matrix_inv, f)

print('reshaping result...')
print(np.shape(E_test))

E_2D_test = np.reshape(E_test,(Nx,Ny))#[:(Nx-Ny)]
print('generate img...')

plt.imshow(E_2D_test, 'gist_heat')
# plt.imshow(matrix, 'gist_heat')
plt.colorbar()
print('saving img...')
plt.savefig('test_3.png')







