#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import pandas


# In[12]:


piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/organoid_10_with_mask_PIV(all_tp).h5"
i = 10
with h5py.File(piv_vec_h5_path, 'r') as f:
    all_tp = list(f.keys()) # all time points/the labels of the top layer. 
    U = f[all_tp[i]]["U"][:] # Output shape is (z, y, x)
    V = f[all_tp[i]]["V"][:]
    W = f[all_tp[i]]["W"][:]
    xgrid = f[all_tp[i]]["xgrid"][:]
    ygrid = f[all_tp[i]]["ygrid"][:]
    zgrid = f[all_tp[i]]["zgrid"][:]   


# In[15]:


xgrid


# ## Generating pseudo tracks based on piv results

# In[4]:


def average_voxel_cube(volume, center, n):
    """
    Computes the average of an n x n x n cube of voxel values centered at a given position in a 3D volume.
    
    Parameters:
        volume (np.ndarray): 3D numpy array representing the volume.
        center (tuple): A tuple of (x, y, z) coordinates for the center voxel.
        n (int): Size of the cube (must be odd for symmetric centering).
        
    Returns:
        float: The average of the voxel values in the cube.
    """
    if n % 2 == 0:
        raise ValueError("n must be an odd number for symmetric centering.")

    x, y, z = center # The position starts with 0 in python!
    half = n // 2

    # Calculate the bounds and clamp them to volume limits. The single PIV volume at tp t has (z, y, x) dimensions. 
    x_min = max(x - half, 0)
    x_max = min(x + half + 1, volume.shape[2]) # +1 because the Python slicing below is excl. the upper bound. 
    y_min = max(y - half, 0)
    y_max = min(y + half + 1, volume.shape[1])
    z_min = max(z - half, 0)
    z_max = min(z + half + 1, volume.shape[0])

    cube = volume[x_min:x_max, y_min:y_max, z_min:z_max]
    return cube.mean()


# In[12]:


# vol = np.random.randint(2, size=(5, 4, 6))
# print(vol)
# center = (3, 3, 3)
# n = 3
# avg_value = average_voxel_cube(vol, center, n)
# print(f"cube: {avg_value}")


# In[20]:


def pseudo_tracking_piv_grid_single(piv_file_path, start_x, start_y, start_z, scale=(1, 1, 1)):
    """
    Calculate the pseudo trajectory with single start point (x, y, z). 
    # U, V, W: (#total_time, z, y, x) dimensional
    # t_interval: [t0, t1] two element list
    # start_x: [x0] one element list. x, y in image coordinate.
    # start_y: [y0] one element list
    # start_z: [z0] one element list. 
    # scale = scales for (z, y, x)
    # step_width = (step_x, step_y, step_z)
    """
    
    trajectory_x = [start_x] # starting points in image coordinates.
    trajectory_y = [start_y]
    trajectory_z = [start_z]

    with h5py.File(piv_vec_h5_path, 'r') as f:
         all_tp = list(f.keys()) # all time points/the labels of the top layer. 
        
    for i in range(len(all_tp)): # Iteration over the whole time point.    
        with h5py.File(piv_vec_h5_path, 'r') as f:
            U = f[all_tp[i]]["U"][:] # Output shape is (z, y, x)
            V = f[all_tp[i]]["V"][:]
            W = f[all_tp[i]]["W"][:]
            xgrid = f[all_tp[i]]["xgrid"][:]
            ygrid = f[all_tp[i]]["ygrid"][:]
            zgrid = f[all_tp[i]]["zgrid"][:]     
        # Calculate the step width at this time point. (The step width is generally dynamic). 
        x_step_width = xgrid[0, 0, 1] - xgrid[0, 0, 0]
        y_step_width = ygrid[0, 1, 0] - ygrid[0, 0, 0]
        z_step_width = zgrid[1, 0, 0] - zgrid[0, 0, 0]
        
        # Transform the points into vector field grid coordinates. 
        x_cor = np.round(trajectory_x[i]/x_step_width - 1).astype(int)
        y_cor = np.round(trajectory_y[i]/y_step_width - 1).astype(int)
        z_cor = np.round(trajectory_z[i]/z_step_width - 1).astype(int)

        center = (x_cor, y_cor, z_cor)
        n = 3 # The size of the voxel that will be avgeraged in the PIV volume.
        
        # Determine the vector directions through averaging surrounding neighbours.
        # dx, dy, and dz are changes in img coordinates!

        dx = scale[2] * average_voxel_cube(V, center, n) # Indexing in python begins with 0. 
        dy = scale[1] * average_voxel_cube(U, center, n)
        dz = scale[0] * average_voxel_cube(W, center, n)

        # Update of the start positions in image coordinate. --> final updated position in img coordinate.
        # If necessary can be round to the nearest integer. As written in outcommented code.
        update_x_img = trajectory_x[i] + dx # round(trajectory_x[i-1] + dx)
        update_y_img = trajectory_y[i] + dy # round(trajectory_y[i-1] + dy)
        update_z_img = trajectory_z[i] + dz # round(trajectory_y[i-1] + dy)

        # Tranform the updates position in 
        update_x_grid = np.round(update_x_img/x_step_width - 1)
        update_y_grid = np.round(update_y_img/y_step_width - 1)
        update_z_grid = np.round(update_z_img/z_step_width - 1)

        # Stays in the last inside position if comes over the image boundary.  
        if update_x_grid < 0 or update_x_grid >= U.shape[2]:
            trajectory_x.append(trajectory_x[i])
            trajectory_y.append(trajectory_y[i])
            trajectory_z.append(trajectory_z[i])
            continue
        elif update_y_grid < 0 or update_y_grid >= U.shape[1]:
            trajectory_x.append(trajectory_x[i])
            trajectory_y.append(trajectory_y[i])
            trajectory_z.append(trajectory_z[i])
            continue
        elif update_z_grid < 0 or update_z_grid >= U.shape[0]:
            trajectory_x.append(trajectory_x[i])
            trajectory_y.append(trajectory_y[i])
            trajectory_z.append(trajectory_z[i])
            continue

        trajectory_x.append(update_x_img)
        trajectory_y.append(update_y_img)
        trajectory_z.append(update_z_img)

    return np.array(trajectory_x), np.array(trajectory_y), np.array(trajectory_z) # Final results in image coordinate.


# # TODO: Single cell tracking probably makes not very much sense...? 

# In[26]:


piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/organoid_10_with_mask_PIV(all_tp).h5"
x, y, z =pseudo_tracking_piv_grid_single(piv_vec_h5_path, 20, 30, 30)
x1, y1, z1 = pseudo_tracking_piv_grid_single(piv_vec_h5_path, 25, 30, 30)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the 3D line
ax.plot(x, y, z, marker='o')
ax.plot(x1, y1, z1, marker='o')

plt.show()


# ## Input the PIV data. 

# In[24]:


# Generation of pseudo tracks with multiple start conditions. 

def pseudo_tracking_piv_grid_v2(U, V, W, t_interval_array, start_x_array, start_y_array, start_z_array,
                                piv_step_width, scale=(1, 1, 1)):
    """
    # t_interval_array: [[t0, t1], ...] list of two element lists
    # start_x_array: [[x0], ...] list of one element lists, image coordinates.
    # start_y_array: [[y0], ...] list of one element lists, image coordinates.
    # piv_step_width: (step_x, step_y, step_z)
    """
    all_x_trajectory = []
    all_y_trajectory = []
    all_z_trajectory = []

    for i in range(len(start_x_array)):
        # Transform the x and y in image coordinates into vector field grid.
        start_x = start_x_array[i]
        start_y = start_y_array[i]
        start_z = start_z_array[i]
        t_interval = t_interval_array[i]

        x_track, y_track, z_track = pseudo_tracking_piv_grid_single_v2(U, V, W, t_interval, start_x, start_y, start_z, 
                                                                       piv_step_width, scale)
        # Get the track results in vector field grid and transform into image coordinate.
        all_x_trajectory.append(x_track)
        all_y_trajectory.append(y_track)
        all_z_trajectory.append(z_track)

    return all_x_trajectory, all_y_trajectory, all_z_trajectory


# In[25]:


# Plot the pseudo track results. TODO: modify it to 3D plot. And later also in vtk for ParaView visualisation. 
def plot_pseudo_track(track_id, x_track, y_track, t_track, fig_size):
    plt.style.use('dark_background')
    colors = list(mcolors.CSS4_COLORS.keys())
    
    plt.figure(figsize=(10, 10))
    for i in range(len(track_id)):
        
        x_cor = x_track[i]
        y_cor = y_track[i]
        t_interval = range(t_track[i][0], t_track[i][-1]+1)
        
        plt.plot(x_cor, y_cor, color=colors[i], zorder=1)
        cmap = def_cmap(colors[i])
        plt.scatter(x_cor, y_cor, c=t_interval, zorder=2, cmap=cmap)
        plt.annotate(str(track_id[i]), xy = (x_cor[0], y_cor[0]), c=colors[i])
    
    plt.xlim(0, fig_size)
    plt.ylim(0, fig_size)
    plt.gca().invert_yaxis()
    #plt.savefig("pseudotrack_img_cor", dpi=300)
    plt.show()


# In[ ]:





# In[ ]:




