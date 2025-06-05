#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import h5py
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv


# In[2]:


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


# In[3]:


def plot_3D_PIV(origins, vectors):
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each vector
    for origin, vector in zip(origins, vectors):
        ax.quiver(*origin, *vector, color='r', linewidth=2)


# In[4]:


def create_vtk(origins, vectors, vtk_file_name):
    # Create a PyVista point cloud
    point_cloud = pv.PolyData(origins, force_float=False)
    # Add vectors to the point cloud as point data
    point_cloud["vectors"] = vectors
    point_cloud.set_active_vectors("vectors")
    # Save as VTK file
    point_cloud.arrows.save(vtk_file_name + ".vtk")


# In[5]:


def filter_vectors_through_length(threshold_length, vectors):
    
    # Compute the Euclidean norm (length) of each vector
    lengths = np.linalg.norm(vectors, axis=1)
    
    # Find which vectors are too long
    mask = lengths > threshold_length
    
    # Zero out those vectors.
    filt_vectors = vectors.copy()
    filt_vectors[mask] = 0
    return filt_vectors


# In[9]:


# # Convert all PIV vector in h5 files into .vtk file for Visualisation. 

# File path of the generated piv h5 flie from Julia Quick-PIV. 
piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/organoid_10_with_mask_PIV(all_tp).h5"
with h5py.File(piv_vec_h5_path, 'r') as f:
    all_tp = list(f.keys())
    print(all_tp)

# Path for saving the vtk files. 
piv_vtk_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/PIV_organoid_10/"

for i in range(len(all_tp)):
    
    with h5py.File(piv_vec_h5_path, "r") as h5f:
        U = h5f[all_tp[i]]["U"][:] # Output shape is (z, y, x)
        V = h5f[all_tp[i]]["V"][:]
        W = h5f[all_tp[i]]["W"][:]
        xgrid = h5f[all_tp[i]]["xgrid"][:]
        ygrid = h5f[all_tp[i]]["ygrid"][:]
        zgrid = h5f[all_tp[i]]["zgrid"][:]
    
    origins, vectors = convert_origins_vectors(U, V, W, xgrid, ygrid, zgrid)
    # plot_3D_PIV(origins, vectors)
    create_vtk(origins, vectors, piv_vtk_path+"tp"+str(42+i))
    print("create"+str(all_tp[i])+".vtk")


# In[14]:


lengths = np.linalg.norm(vectors, axis=1)
plt.hist(lengths)


# In[15]:


filt_vectors = filter_vectors_through_length(10, vectors)
plot_3D_PIV(origins, filt_vectors)
create_vtk(origins, filt_vectors, "filt_PIV")


# In[ ]:




