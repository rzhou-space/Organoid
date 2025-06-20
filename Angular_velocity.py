#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import h5py


# # Single Organoid Angular Velocity Calculation

# In[7]:


# # Import the organid center information.

# rotate_axis_file = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/new_padding/organoid_10/organoid_10_rotate_axis.h5"

# i = 10

# with h5py.File(rotate_axis_file, 'r') as f:
#     all_tp = list(f.keys())
#     # Sort the tp order through integer order instead of string order. 
#     all_tp_sorted = sorted(all_tp, key=lambda x: int(x[2:]))
#     tp = all_tp_sorted[i]
#     print(tp)
#     center = f[tp]["rotation center"][:]


# In[8]:


# # Import the PIV vectors information for the estimation of the angular velocity.

# piv_vec_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/new_padding/organoid_10/organoid_10_with_mask_PIV(all_tp).h5"

# i = 10

# with h5py.File(piv_vec_h5_path, 'r') as f:
#     all_tp = list(f.keys()) # all time points/the labels of the top layer. 
#     print(all_tp[i])
#     U = f[all_tp[i]]["U"][:] # Output shape is (z, y, x)
#     V = f[all_tp[i]]["V"][:]
#     W = f[all_tp[i]]["W"][:]
#     xgrid = f[all_tp[i]]["xgrid"][:]
#     ygrid = f[all_tp[i]]["ygrid"][:]
#     zgrid = f[all_tp[i]]["zgrid"][:]   

# # Combine the 3D information into (N, 3) Array. 
# # Stack the vector directions to form an (N, 3) array
# vectors = np.vstack((U.flatten(), V.flatten(), W.flatten())).T
# # Stack the origins of the vectors to form an (N, 3) array.
# origins = np.vstack((xgrid.flatten(), ygrid.flatten(), zgrid.flatten())).T

# # Keep only vectors with sufficient magnitude
# magnitudes = np.linalg.norm(vectors, axis=1) # magnitudes < threshold_piv in Julia already. 
# threshold = 1e-3
# mask = magnitudes > threshold  # e.g., threshold = 1e-3. For filtering out the small 
# vectors_filtered = vectors[mask]
# # Get the corresponding origins of the vectors left. 
# origins_filtered = origins[mask]


# In[9]:


# Calculate the single angular speed.

def single_point_angular_speed(center, origin, vector, time_width): 
    # center: The rotation center of the organoid. 
    # origin: the point (middle of one IA) where the piv vector is. 
    # vector: the piv vector given the moment velocity at the origin. With unit pxl/time_width
    r = origin - center
    omega = np.cross(r, vector) / np.dot(r, r) # Unit = (pxl^2/time_width)/pxl^2 = 1/time_width
    angular_speed = np.linalg.norm(omega)
    # change the angular velocity with rad/h as unit, so that the numbers are not too small. 
    unit_h = 60/time_width
    return angular_speed*unit_h # The final unit would be 1/h = rad/h pysically in the context of angular speed.


# In[10]:


# single_point_angular_speed(center, origins_filtered[0], vectors_filtered[0], 30)


# In[11]:


# Cauclate the mean angular speed of all IA (PIV vectors) at one single time point. 

def one_time_point_angular_speed(center, origins, vectors, time_width):
    # center: the center point of the organoid at one certain time point. 
    # origins: the list of all considered origins of the PIV vectors the calculation for angular speed. 
    # vectors: all the PIV vectors that will be considered for the calculation of angular speed. 
    all_angular_speed = []
    for i in range(len(origins)): 
        origin = origins[i]
        vector = vectors[i]
        angular_speed = single_point_angular_speed(center, origin, vector, time_width)
        all_angular_speed.append(angular_speed)
    return np.mean(np.array(all_angular_speed)), np.array(all_angular_speed) # Due to the function of single_point_angular_speed()


# In[12]:


# Calculate the angular speed for all time points for a given organoid/the given path. 

def angular_speed_over_time(organoid):
    # The piv vector files and rotation axis files for the given organoid. 
    piv_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/new_padding/"+organoid+"/"+organoid+"_with_mask_PIV(all_tp).h5"
    rotate_axis_file = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/new_padding/"+organoid+"/"+organoid+"_rotate_axis.h5"
    
    # Collect all the centers at each time point. 
    all_centers = []
    with h5py.File(rotate_axis_file, 'r') as f:
        all_tp = list(f.keys())
        # Sort the tp order through integer order instead of string order. 
        all_tp_sorted = sorted(all_tp, key=lambda x: int(x[2:]))
        for tp in all_tp_sorted:
            center = f[tp]["rotation center"][:]
            all_centers.append(center)
    
    # For each time point, calculate the angular speed. 
    all_mean_angular_speed = [] # Store the mean angular speed at each time point. 
    all_variance = [] # Store the variance at each time point for marking the error bar. 
    all_angular_speed = [] # Store all the angular speed at each time point. 
    
    with h5py.File(piv_file_path, 'r') as f:
        all_tp = list(f.keys()) # all time points/the labels of the top layer. 
        
        for i in range(len(all_centers)):
            
            U = f[all_tp[i]]["U"][:] # Output shape is (z, y, x)
            V = f[all_tp[i]]["V"][:]
            W = f[all_tp[i]]["W"][:]
            xgrid = f[all_tp[i]]["xgrid"][:]
            ygrid = f[all_tp[i]]["ygrid"][:]
            zgrid = f[all_tp[i]]["zgrid"][:]   
    
            # Combine the 3D information into (N, 3) Array. 
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
    
            # Calculation of the angular speed for the i-th time point. 
            center = all_centers[i]
            mean_angular_speed, angular_speed = one_time_point_angular_speed(center, origins_filtered, vectors_filtered, 30)
            
            all_mean_angular_speed.append(mean_angular_speed)
            all_variance.append(np.var(angular_speed))
            all_angular_speed.append(angular_speed)

    return np.array(all_mean_angular_speed), np.array(all_variance), all_angular_speed


# In[13]:


mean_angular_speed_org10, variance_org10, angular_speed_org10 = angular_speed_over_time("organoid_10") # Small organoid. 
mean_angular_speed_org34, variance_org34, angular_speed_org34 = angular_speed_over_time("organoid_34") # Large organoid. 


# In[14]:


max(np.concatenate(angular_speed_org10))


# In[15]:


# Compare the angular speed over time between large (organoid 34) and small (organod 10) organoids. And add axis labels.

import scipy as sp # For the median filter. 

time_width = 30 # The time distance between frames. 
time_convert = time_width/60 # Convert time into hours. 

plt.figure(dpi=300)
plt.plot([i*time_convert for i in range(len(mean_angular_speed_org10))], mean_angular_speed_org10, color="blue", alpha=0.3)
# Apply the meadian filter for smoothing the curve and show the variance as error bars. 
plt.errorbar([i*time_convert for i in range(len(mean_angular_speed_org10))], 
             sp.signal.medfilt(mean_angular_speed_org10, 5), yerr=variance_org10, fmt="-o", capsize=3,
             color="blue", ecolor="lightblue", label="organoid_10(Small)")

plt.plot([i*time_convert for i in range(len(mean_angular_speed_org34))], mean_angular_speed_org34, color="orange", alpha = 0.3)
# Apply the median filter for smoothing the curve. 
plt.errorbar([i*time_convert for i in range(len(mean_angular_speed_org34))], 
             sp.signal.medfilt(mean_angular_speed_org34, 5), yerr=variance_org34, fmt="-o", capsize=3,
             color="orange", ecolor="bisque", label="organoid_34(Large)")

plt.legend()
plt.xlabel("Time (h)")
plt.ylabel("Angular Velocity (rad/h)")
plt.xticks([0, 5, 10, 15, 20])
plt.show()


# In[ ]:




