#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import h5py
import plotly
import pyvista as pv
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

# In[3]:


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
        


# In[4]:


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

# In[5]:


piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/new_padding/organoid_34/organoid_34_with_mask_PIV(all_tp).h5"

h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/organoid_34/"

vtk_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/new_padding/organoid_34/"

organoid = "organoid_34" 

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


# # Analysis of the Rotational Axis and the corresponding variance in Correlation with Flipping dynamics.

# ## Import the file with rotation axis information. 

# In[4]:


organoid = "organoid_10"
rotation_axis_file = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/new_padding/"+organoid+"/"+organoid+"_rotate_axis.h5"

with h5py.File(rotation_axis_file, 'r') as f:
    all_axis = []
    all_variances = []
    all_centers = []
    all_tp = list(f.keys()) # all time points/the labels of the top layer. 
    # Sort the tp order through integer order instead of string order. 
    all_tp_sorted = sorted(all_tp, key=lambda x: int(x[2:]))
    print(all_tp_sorted)

    for i in all_tp_sorted:
        rotation_axis = f[i]["PC3"][:]
        rotation_center = f[i]["rotation center"][:]
        variances = f[i]["variances"][:]
        
        all_axis.append(rotation_axis)
        all_centers.append(rotation_center)
        all_variances.append(variances)


# ## Plot the temporal rotation axis in 3D and colored by the size of corresponding PC variance. 

# In[5]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

# Normalize and get colormap
values = np.array(all_variances)[:,2]
norm = Normalize(vmin=np.min(values), vmax=np.max(values))
cmap = plt.colormaps.get_cmap('viridis')  # you can also try 'plasma', 'coolwarm', etc.
colors = cmap(norm(values))

# Create figure
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')

origins = np.array(all_centers)
vectors = np.array(all_axis)

# Plot each vector
for origin, vector, color in zip(origins, vectors, colors):
    ax.quiver(*origin, *vector, color=color, linewidth=2)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='PC3 Variances')

# Normalize and colormap for the dashed line. 
norm_2 = Normalize(vmin=np.min(0), vmax=np.max(len(origins)))
cmap_2 = plt.colormaps.get_cmap("Greys")
colors_2 = cmap_2(norm_2(range(len(origins))))

# Plot colored dashed lines between consecutive origins
for i in range(len(origins)-1):
    x = [origins[i][0], origins[i+1][0]]
    y = [origins[i][1], origins[i+1][1]]
    z = [origins[i][2], origins[i+1][2]]
    ax.plot(x, y, z, linestyle='--', color=colors_2[i])

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,1])
ax.view_init(elev=30, azim=45) 
plt.show()


# In[11]:


import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def create_circle_around_vector(origin, vector, radius=0.2, n_points=50):
    # Normalize vector
    v = vector / np.linalg.norm(vector)

    # Find two orthogonal vectors perpendicular to v. 
    # Alternative: could also use normalised PC1 and PC2. 
    if np.allclose(v, [0, 0, 1]):
        perp1 = np.cross(v, [1, 0, 0])
    else:
        perp1 = np.cross(v, [0, 0, 1])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(v, perp1)

    # Create points of the circle
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle_points = np.array([
        origin + radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
        for t in theta
    ])
    return circle_points


# In[12]:


# multiple vectors with origins
origins = np.array(all_centers)
vectors = np.array(all_axis)

# Define the colormap normalisation for the vector lines. 
values = np.array(all_variances)[:,2]
norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
colormap = plt.colormaps.get_cmap('viridis')  # or 'plasma', 'coolwarm', etc.
# Convert to hex color strings for Plotly
colors = [mcolors.to_hex(colormap(norm(v))) for v in values]

# Define the colormap normalization for the dashed lines between two consecutive origins. 
# Generate "time" or "index" values to map to color
values_dash = np.linspace(0, 1, len(origins) - 1)  # one value per segment
# Normalize and get colormap
norm_dash = mcolors.Normalize(vmin=np.min(values_dash), vmax=np.max(values_dash))
colormap_dash = plt.colormaps.get_cmap('spring')  # or viridis, inferno, etc.
colors_dash = [mcolors.to_hex(colormap_dash(norm_dash(v))) for v in values_dash]

# Start plot
fig = go.Figure()

# Plot each vector and circle
for origin, vector, color in zip(origins, vectors, colors):
    # Line segment for the vector
    end = origin + vector
    fig.add_trace(go.Scatter3d(
        x=[origin[0], end[0]], # start points and end points of the lines. 
        y=[origin[1], end[1]],
        z=[origin[2], end[2]],
        mode='lines',
        line=dict(color=color, width=5),
        showlegend=False
    ))

    # Circle around the vector
    circle = create_circle_around_vector(origin, vector, radius=0.2)
    fig.add_trace(go.Scatter3d(
        x=circle[:, 0],
        y=circle[:, 1],
        z=circle[:, 2],
        mode='lines',
        line=dict(color='red', dash='dash'),
        showlegend=False
    ))

# Add colorbar manually using dummy scatter
# (Optional: helpful for interpreting color mapping)
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None],
    mode='markers',
    marker=dict(
        colorscale='viridis',
        cmin=np.min(values),
        cmax=np.max(values),
        colorbar=dict(title="Variance PC3"), # Eigenvalues. Absolute variance for PC. 
        color=values, size=0  # invisible, just for colorbar
    ),
    showlegend=False
))

# Add dashed lines between origins. 
for i in range(len(origins)-1):
    p1, p2 = origins[i], origins[i + 1]
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p2[0]],
        y=[p1[1], p2[1]],
        z=[p1[2], p2[2]],
        mode='lines',
        line=dict(color=colors_dash[i], width=5, dash='dash'),
        showlegend=False
    ))

# Layout settings
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    title='3D Vectors with Perpendicular Circles',
    margin=dict(l=0, r=0, b=0, t=30)
)

fig.write_html("filename.html", auto_open=True)


# In[ ]:




