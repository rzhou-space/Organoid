#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
import plotly


# In[4]:


piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/organoid_10_with_mask_PIV(all_tp).h5"
i = 10
with h5py.File(piv_vec_h5_path, 'r') as f:
    all_tp = list(f.keys()) # all time points/the labels of the top layer. 
    U = f[all_tp[i]]["U"][:] # Output shape is (z, y, x)
    V = f[all_tp[i]]["V"][:]
    W = f[all_tp[i]]["W"][:]
    xgrid = f[all_tp[i]]["xgrid"][:]
    ygrid = f[all_tp[i]]["ygrid"][:]
    zgrid = f[all_tp[i]]["zgrid"][:]   


# In[6]:


U.shape


# In[9]:


np.column_stack((U, V, W)).shape


# ## 1. Combine U, V, W into 3D vectors
# 
# First, convert your 3D arrays into a list of 3D vectors. For PCA, we need a 2D matrix of shape N×3, where each row is a vector at one spatial location:

# In[12]:


# U, V, W are 3D arrays with shape (Z, Y, X)
U_flat = U.flatten()
V_flat = V.flatten()
W_flat = W.flatten()

# Stack to form an (N, 3) array
vectors = np.vstack((U_flat, V_flat, W_flat)).T


# In[13]:


vectors


# ## 2. Filter out zero or near-zero vectors (optional but helpful)
# 
# In many PIV fields, there might be regions with little or no motion. These can skew PCA:

# In[14]:


# Keep only vectors with sufficient magnitude
magnitudes = np.linalg.norm(vectors, axis=1) # magnitudes < threshold_piv in Julia already. 
threshold = 1e-3
mask = magnitudes > threshold  # e.g., threshold = 1e-3. For filtering out the small 
vectors_filtered = vectors[mask]


# ## 3. Center the data
# 
# This removes any net translation, isolating rotational structure:

# In[17]:


vectors_centered = vectors_filtered - vectors_filtered.mean(axis=0)


# ## 4. Apply PCA

# In[18]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(vectors_centered)

# PCA axes (principal directions)
rotation_plane_axes = pca.components_[0:2]  # First two: dominant plane
rotation_axis = pca.components_[2]         # Third: least variance → rotation axis


# In[19]:


rotation_axis


# ## 5. Determine the rotaional center (the origin for the rotational axis)
# One idea: Defined as the circle middle of the plane with the maximal PIV vector velocity. 

# In[ ]:





# In[ ]:





# In[ ]:





# ## Plot the vector could and the rotational axis. 

# In[42]:


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


# In[47]:


origin, vectors = convert_origins_vectors(U, V, W, xgrid, ygrid, zgrid)
plot_3D_PIV(origin, vectors, [30, 80, 0], rotation_axis*30)


# ## TODO: Plotly dont have quiver function (has to set up arrow self) 
# Not possible for overlapping. Easier to visualise in Paraview. Can also just store the rotational axies alone and overlap with other layers, e.g the organoid and the piv vectors.

# In[ ]:




