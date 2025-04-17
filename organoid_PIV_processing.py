#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import h5py
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv


# In[2]:


file_path = "organoid_10_PIV.h5"

tp = "tp42-43"

with h5py.File(file_path, "r") as h5f:
    U = h5f[tp]["U"][:] # Output shape is (z, x, y)
    V = h5f[tp]["V"][:]
    W = h5f[tp]["W"][:]
    xgrid = h5f[tp]["xgrid"][:]
    ygrid = h5f[tp]["ygrid"][:]
    zgrid = h5f[tp]["zgrid"][:]


# In[3]:


pos = [0, 0, 0]

u1 = U[pos[0], pos[1], pos[2]]
v1 = V[pos[0], pos[1], pos[2]]
w1 = W[pos[0], pos[1], pos[2]]
x1 = xgrid[pos[0], pos[1], pos[2]]
y1 = ygrid[pos[0], pos[1], pos[2]]
z1 = zgrid[pos[0], pos[1], pos[2]]

# Define the vector
origin = [x1, y1, z1]  # Start point
vector = [u1, v1, w1]  # Direction and magnitude

# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vector using quiver
ax.quiver(*origin, *vector, color='r', linewidth=2)

# Set limits (optional)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title("3D Vector Plot")
plt.show()


# In[4]:


# Convert 3D arrays into array of (N, 3) shape, where N is the total number of points. 
def convert_origins_vectors(U, V, W, xgrid, ygrid, zgrid):
    # Flatten all arrays for the origins of vectors. 
    x_flat = xgrid.ravel()
    y_flat = ygrid.ravel()
    z_flat = zgrid.ravel()
    # Stack into shape (N, 3), where N is the total number of points. 
    origins = np.stack((x_flat, y_flat, z_flat), axis=1)
    
    # Flatten all arrays for the vector directions. 
    u_flat = U.ravel()
    v_flat = V.ravel()
    w_flat = W.ravel()
    # Stack into shape (N, 3), where N is the total number of points. 
    vectors = np.stack((u_flat, v_flat, w_flat), axis=1)

    return origins, vectors 


# In[11]:


def plot_3D_PIV(origins, vectors):
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each vector
    for origin, vector in zip(origins, vectors):
        ax.quiver(*origin, *vector, color='r', linewidth=2)


# In[18]:


def create_vtk(origins, vectors, vtk_file_name):
    # Create a PyVista point cloud
    point_cloud = pv.PolyData(origins, force_float=False)
    
    # Add vectors to the point cloud as point data
    point_cloud["vectors"] = vectors
    
    # Save as VTK file
    point_cloud.save(vtk_file_name + ".vtk")


# In[15]:


def filter_vectors_through_length(threshold_length, vectors):
    
    # Compute the Euclidean norm (length) of each vector
    lengths = np.linalg.norm(vectors, axis=1)
    
    # Find which vectors are too long
    mask = lengths > threshold_length
    
    # Zero out those vectors.
    filt_vectors = vectors.copy()
    filt_vectors[mask] = 0
    return filt_vectors


# In[12]:


origins, vectors = convert_origins_vectors(U, V, W, xgrid, ygrid, zgrid)
plot_3D_PIV(origins, vectors)


# In[6]:


lengths = np.linalg.norm(vectors, axis=1)
lengths


# In[19]:


filt_vectors = filter_vectors_through_length(10, vectors)
plot_3D_PIV(origins, filt_vectors)
create_vtk(origins, filt_vectors, "filt_PIV")


# In[ ]:




