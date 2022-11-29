import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import matplotlib.patheffects as fx
import seaborn as sns
import matplotlib.animation as animation

def idx1D(i , j, N):
        return i * N + j

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

x = 31
y = 61
Ny = y if y > x else x
Nx = x if y > x else y
dx = 0.001
dy = 0.001
k = 6.28/0.012 #2pi/długość fali
n = np.ones((Nx,Ny))
f = np.zeros(Nx*Ny)
k = 6.28/0.012
print('let\'s goooo')

for i in range(Nx):
    f = np.zeros((Nx*Ny))
    f[idx1D(i,Ny//2,Ny)] = 1
    matrix = generateMatrix(Nx,Ny,dx,dy,n,k)
    matrix_inv = pinv(matrix)
    E = matmul(matrix_inv, f)
    E_2D = np.reshape(E,(Nx,Ny))
    plt.imshow(E_2D, 'gist_heat')
    plt.title('(x,y) = ({:2.0f},{:2.0f}) [mm]'.format(Ny//2,i))
    plt.savefig('./graphs/E_f{:d}'.format(i))
    print('{:2.0f}/{:2.0f}'.format(i,Nx))

print('done B|')

fig, ax = plt.subplots()

# Adjust figure so GIF does not have extra whitespace
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax.axis('off')
ims = []

pics = []
for i in range(Nx):
    pics.append('E_f{:d}'.format(i))

for pic in pics:
    im = ax.imshow(plt.imread(f'./graphs/{pic}.png'), animated = True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=200)
ani.save('E_moving_f.gif')




# ------- Generacja gifu z różnym k 

# K = np.linspace(0,6.28/0.12,200)
# f = np.zeros(Nx**2)
# f[(Nx**2)//2]=1
# E_2D = np.zeros((Nx,Ny))

# for i in range(len(K)):
#     matrix = generateMatrix(Nx,Ny,n,K[i])
#     E = np.linalg.solve(matrix, f)
#     for j in range(Nx-1):
#         E_2D[j] = E[j*Nx:(j+1)*Nx]
#     plt.imshow(E_2D, 'gist_heat')
#     plt.title('k = {:f}'.format(K[i]))
#     plt.savefig('./graphs/E_k{:d}'.format(i))

# fig, ax = plt.subplots()

# # Adjust figure so GIF does not have extra whitespace
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
# ax.axis('off')
# ims = []

# pics = []
# for i in range(200):
#     pics.append('E_k{:d}'.format(i))

# for pic in pics:
#     im = ax.imshow(plt.imread(f'./graphs/{pic}.png'), animated = True)
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=200)
# ani.save('E_81_81.gif')