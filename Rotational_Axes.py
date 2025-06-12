#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
import plotly
import pyvista as pv
from sklearn.decomposition import PCA
import os


# # Single timepoint experiment for calculating rotation plane, rotation axis, and rotation center.  

# ## Import the PIV vectors
# 
# Those vectors should already be filtered by a thresholding magnitude before. Those are the ones that are visualized in ParaView. 

# In[2]:


piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/organoid_10_with_mask_PIV(all_tp).h5"
i = 1
with h5py.File(piv_vec_h5_path, 'r') as f:
    all_tp = list(f.keys()) # all time points/the labels of the top layer. 
    U = f[all_tp[i]]["U"][:] # Output shape is (z, y, x)
    V = f[all_tp[i]]["V"][:]
    W = f[all_tp[i]]["W"][:]
    xgrid = f[all_tp[i]]["xgrid"][:]
    ygrid = f[all_tp[i]]["ygrid"][:]
    zgrid = f[all_tp[i]]["zgrid"][:]   


# ## 1. Combine the matrices into 3D vectors
# 
# First, convert your 3D arrays into a list of 3D vectors. For PCA, we need a 2D matrix of shape N×3, where each row is a vector at one spatial location:

# In[3]:


# Stack the vector directions to form an (N, 3) array
vectors = np.vstack((U.flatten(), V.flatten(), W.flatten())).T

# Stack the origins of the vectors to form an (N, 3) array.
origins = np.vstack((xgrid.flatten(), ygrid.flatten(), zgrid.flatten())).T


# In[4]:


origins


# ## 2. Filter out zero or near-zero vectors (background noise). 
# 
# In many PIV fields, there might be regions with little or no motion. These can skew PCA:

# In[5]:


# Keep only vectors with sufficient magnitude
magnitudes = np.linalg.norm(vectors, axis=1) # magnitudes < threshold_piv in Julia already. 
threshold = 1e-3
mask = magnitudes > threshold  # e.g., threshold = 1e-3. For filtering out the small 
vectors_filtered = vectors[mask]

# Get the corresponding origins of the vectors left. 
origins_filtered = origins[mask]


# ## 3. Center the data
# 
# This removes any net translation, isolating rotational structure:

# In[6]:


vectors_centered = vectors_filtered - vectors_filtered.mean(axis=0)


# ## 4. Apply PCA
# 
# Getting the first 3 principle components, and the corresponding eigenvalues/variances. 
# The first PC1 and PC2 span the rotation plane, perpendicular to the PC3, which is the rotational axis. 
# 
# The eigenvalues/variances how the strength of the distribution in the corresponding PC direction. 
# For a "good" rotation on the rotation plane, the variance of PC1 and PC2 should be significantly larger than the variance for PC3. But during the flipping (or before a flip), the variance of PC3 could increase. 

# In[7]:


pca = PCA(n_components=3)
pca.fit(vectors_centered)

# PCA axes (principal directions)
rotation_plane_axes = pca.components_[0:2]  # First two: dominant plane
rotation_axis = pca.components_[2]         # Third: least variance → rotation axis

# Get the variance/eigenvalues for the first 3 PCs.
eigenvalues = pca.explained_variance_[0:3]


# In[8]:


rotation_plane_axes, rotation_axis


# In[9]:


eigenvalues


# ## 5. Determine the rotaional center (the origin for the rotational axis)
# 
# #### Quick & Dirty: 
# Use the filtered PIV vectors. 
# The origins of the PIV vectors are the centers of the corresponding interogation areas. Determine the spatial middle point of all the interogation areas as the approximation of the real rotational center. 
# 
# However, the estimation is not really precise and need the assumption that the rotation of organoid is significantly distinguished from the background. 

# In[10]:


# The middle point of the left vectors after filtering.
rotation_center = np.round(np.mean(origins_filtered, axis=0))
np.array([rotation_center])


# #### A better way: Still to be done!
# Detemrine the cells position and the corresponding cell centers. Binding the cell centres to get the convex hull round the organoid. Then the spatial middle point of the convex hull will be the more precise middle point/rotation centre. 

# In[ ]:





# # Summary as a function
# Apply PCA on the PIV vectors for getting the rotational plane (PC1 and PC2), rotational axis (PC3) and the corresponding variances. Besides, the rotational center will also be calculated (but first in a quick and dirty way -- has to be optimized). 

# In[2]:


def pca_piv_vectors(piv_file_path, tp):
    # Getting the input PIV data. 
    with h5py.File(piv_file_path, 'r') as f:
        U = f[tp]["U"][:] # Output shape is (z, y, x)
        V = f[tp]["V"][:]
        W = f[tp]["W"][:]
        xgrid = f[tp]["xgrid"][:]
        ygrid = f[tp]["ygrid"][:]
        zgrid = f[tp]["zgrid"][:]   

    # Combine vectors and origins in (N,3) arrays. 
    # Stack the vector directions to form an (N, 3) array
    vectors = np.vstack((U.flatten(), V.flatten(), W.flatten())).T
    # Stack the origins of the vectors to form an (N, 3) array.
    origins = np.vstack((xgrid.flatten(), ygrid.flatten(), zgrid.flatten())).T

    # Keep only vectors with sufficient magnitude
    magnitudes = np.linalg.norm(vectors, axis=1) # magnitudes < threshold_piv in Julia already. 
    threshold = 1e-3
    mask = magnitudes > threshold  # e.g., threshold = 1e-3. For filtering out the small 
    vectors_filtered = vectors[mask]
    # Get the corresponding origins of the vectors left. 
    origins_filtered = origins[mask]

    # Centering the data. This removes any net translation, isolating rotational structure.
    vectors_centered = vectors_filtered - vectors_filtered.mean(axis=0)

    # Apply PCA.
    pca = PCA(n_components=3)
    pca.fit(vectors_centered)
    # PCA axes (principal directions)
    rotation_plane_axes = pca.components_[0:2]  # First two: dominant plane
    rotation_axis = pca.components_[2]         # Third: least variance → rotation axis
    # Get the variance/eigenvalues for the first 3 PCs.
    eigenvalues = pca.explained_variance_[0:3]

    # Calculate the rotation center. 
    # The middle point of the left vectors after filtering.
    rotation_center = np.round(np.mean(origins_filtered, axis=0))

    return rotation_plane_axes, rotation_axis, eigenvalues, rotation_center


# ## Plot the vector field and the rotational axis. 

# In[11]:


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

def plot_3D_PIV(origins, vectors, extra_origin, extra_vec):
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each vector
    for origin, vector in zip(origins, vectors):
        ax.quiver(*origin, *vector, color='r', linewidth=2)
        
    ax.quiver(*extra_origin, *extra_vec, color="b")


# In[13]:


origin, vectors = convert_origins_vectors(U, V, W, xgrid, ygrid, zgrid)
plot_3D_PIV(origin, vectors, rotation_center, rotation_axis*30)


# ## Store the rotational vectors into .h5 files and visualise in ParaView through .vtk 

# In[ ]:


# piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/organoid_10_with_mask_PIV(all_tp).h5"

# h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/"

# vtk_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/"

# organoid = "organoid_10" 

# with h5py.File(h5_file_path+organoid+"_rotate_axis.h5", "w") as file:
#     time_point_layer = file.create_group("tp"+str(i))
#     time_point_layer.create_dataset("PC1, PC2", data = rotation_plane_axes) # Rotation plane.
#     time_point_layer.create_dataset("PC3", data = rotation_axis) # Rotational axis.
#     time_point_layer.create_dataset("variances", data = eigenvalues) # The variances for PC1, PC2, PC3. 
#     time_point_layer.create_dataset("rotation center", data = rotation_center) # The origin of the rotation axis. 
        


# In[3]:


def create_vtk(origins, vectors, vtk_file_name):
    # Create a PyVista point cloud
    point_cloud = pv.PolyData(origins, force_float=False)
    # Add vectors to the point cloud as point data
    point_cloud["vectors"] = vectors
    point_cloud.set_active_vectors("vectors")
    # Save as VTK file
    point_cloud.arrows.save(vtk_file_name + ".vtk")


# In[16]:


create_vtk(np.array([rotation_center]), np.array([rotation_axis*30]), vtk_file_path+str(i))


# ## Iteration over all time points. 

# In[4]:


piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/organoid_10_with_mask_PIV(all_tp).h5"

h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/"

vtk_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/"

organoid = "organoid_10" 

with h5py.File(piv_vec_h5_path, 'r') as f:
    all_tp = list(f.keys()) # all time points/the labels of the top layer. 

# Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
new_vtk_folder = vtk_file_path + organoid + "_rotata_axis/" # Notice the mask in the file name. 
os.makedirs(new_vtk_folder)

with h5py.File(h5_file_path+organoid+"_rotate_axis.h5", "w") as file:
    for i in range(len(all_tp)):
        rotation_plane_axes, rotation_axis, eigenvalues, rotation_center = pca_piv_vectors(piv_vec_h5_path, all_tp[i])

        time_point_layer = file.create_group("tp"+str(i))
        time_point_layer.create_dataset("PC1, PC2", data = rotation_plane_axes) # Rotation plane.
        time_point_layer.create_dataset("PC3", data = rotation_axis) # Rotational axis.
        time_point_layer.create_dataset("variances", data = eigenvalues) # The variances for PC1, PC2, PC3. 
        time_point_layer.create_dataset("rotation center", data = rotation_center) # The origin of the rotation axis. 
        
        create_vtk(np.array([rotation_center]), np.array([rotation_axis*30]), new_vtk_folder+"tp"+str(i))


# In[ ]:




