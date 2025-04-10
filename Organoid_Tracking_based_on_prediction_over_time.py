#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import cv2
from matplotlib import pyplot as plt 
import numpy as np
# import zipfile
import os
from natsort import os_sorted
import pyvista as pv


# In[ ]:


pip install --force-reinstall -v "pyvista==0.44.1"


# # Converting tif into .h5 files. 
# Firstly convert .tif video into stack of .tif images in a saparate folder. 

# In[ ]:


data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_image/20181113_HCC_d0-2_t42c1_ORG/"

img_stack = []

# Open the file where single .tif 2D images are. (extracted with ImageJ)
items = [file for file in os_sorted(os.listdir(data_dir)) if file.lower().endswith(".tif")]
for file in items:
    img = cv2.imread(data_dir + file, 0) 
    # Normalisation with cv2, so that the minimal intensity 0 and maximal intensity 255. 
    # img_scaled = cv2.normalize(img, dst=True, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    img_stack.append(np.array(img).astype(np.float32))
    print("ok"+ file)

# Volume as a 3D numpy array at the end. 
volume = np.stack(np.array(img_stack), axis=0)


# In[3]:


# save volume as .h5 file

with h5py.File("HCC_t42.h5", "w") as file:
    file.create_dataset("img", data=volume)


# # Cropping single organoid based on the bounding box from Mask-RCNN predictions 

# In[2]:


# open h5 files.
h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_h5_video/HCC_t42.h5"
with h5py.File(h5_file_path, "r") as h5f:
    images_t1 = h5f["img"][:]


# In[3]:


h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_h5_video/HCC_t43.h5"
with h5py.File(h5_file_path, "r") as h5f:
    images_t2 = h5f["img"][:]


# In[4]:


bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

with h5py.File(bbox_h5_path, "r") as h5f:
    t1_x1, t1_y1, t1_z1, t1_x2, t1_y2, t1_z2 = h5f["organoid_10"]["42"]["bbox"][:]
    t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2 = h5f["organoid_10"]["43"]["bbox"][:]


# In[5]:


plt.imshow(images_t1[t1_z1+10][t1_y1:t1_y2, t1_x1:t1_x2])


# In[6]:


# Crop a single organoid based on the bbox from prediction. 

def crop_organoid(img_stack, x1, y1, z1, x2, y2, z2):
    org_stack = []
    for z in range(z1, z2+1): # Include the z2 slice. 
        org_stack.append(img_stack[z1][y1:y2, x1:x2]) # Attention the orientation of coordinates. 
    # Volume as a 3D numpy array at the end. 
    volume = np.stack(np.array(org_stack), axis=0)
    return volume


# In[7]:


volume_t1 = crop_organoid(images_t1, t1_x1, t1_y1, t1_z1, t1_x2, t1_y2, t1_z2)
volume_t2 = crop_organoid(images_t2, t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2)


# In[8]:


# Store the volume as .h5 file. 

with h5py.File("organoid10.h5", "w") as file:
    file.create_dataset("t42", data=volume_t1)
    file.create_dataset("t43", data=volume_t2)


# In[11]:


# Convert the vloume in .vti file for visualisation e.g. in Paraview. 
# ---------------  pyvista version 0.42.3 ---------------------
def convert_vtk(volume, vtk_name):
    depth, height, width = volume.shape
    
    # Create a PyVista UniformGrid (3D volume)
    grid = pv.UniformGrid()
    grid.dimensions = [width, height, depth]  # Make sure this is (X, Y, Z) order
    grid.spacing = [1, 1, 2]  # You can adjust the spacing as needed
    
    # Flatten the 3D volume and assign it to cell data (or point data)
    grid.point_data["values"] = volume.flatten()  
    
    # Add the scalar values to the grid
    
    # Save the volume as a .vti file
    grid.save(vtk_name + ".vti")
    print("Volume saved as output_example.vti!")

# # ----------- pv version 0.44.1 --------------- .vti file not working for the whole stack, only a part of.
#     # Create the ImageData
#     grid = pv.ImageData()
#     # Set the dimension. Make sure this is (X, Y, Z) order so that slice along Z-axis.
#     # depth, height, width = volume.shape
#     grid.dimensions = np.array(volume.shape) + 1
#     # Edit the spatial reference
#     grid.origin = (0, 0, 0)
#     grid.spacing = (1, 1, 1)
#     # Flatten the 3D volume and assign it to cell data (or point data)
#     grid.cell_data["values"] = volume.flatten(order="F")
#     # grid information.
#     print(grid)


# In[12]:


convert_vtk(volume_t1, "tp42")
convert_vtk(volume_t2, "tp43")


# In[ ]:




