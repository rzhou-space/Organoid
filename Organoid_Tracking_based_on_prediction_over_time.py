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
import tifffile


# In[1]:


# # Force package to be a certain version. 
# pip install --force-reinstall -v "pyvista==0.44.1"


# # Converting tif into .h5 files. -- Not Necessary..
# Firstly convert .tif video into stack of .tif images in a saparate folder. 

# In[11]:


# # Option 1: Manually convert the .tif video into single .tif images through ImageJ. --> Unefficient for large amount of data. 
# data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_image/20181113_HCC_d0-2_t42c1_ORG/"

# img_stack = []

# # Open the file where single .tif 2D images are. (extracted with ImageJ)
# items = [file for file in os_sorted(os.listdir(data_dir)) if file.lower().endswith(".tif")]
# for file in items:
#     img = cv2.imread(data_dir + file, 0) 
#     # Normalisation with cv2, so that the minimal intensity 0 and maximal intensity 255. 
#     # img_scaled = cv2.normalize(img, dst=True, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#     img_stack.append(np.array(img).astype(np.float32))
#     print("ok"+ file)

# # Volume as a 3D numpy array at the end. 
# volume = np.stack(np.array(img_stack), axis=0)

# save volume as .h5 file

# with h5py.File("HCC_t42.h5", "w") as file:
#     file.create_dataset("img", data=volume)


# In[12]:


# # Option 2: Convert .tif videos direct into .h5 file. As 3D array with single video frame as 2D array. 

# # Access all .tif video file name from the data folder. 
# folder_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/"

# # Get all .tif file names
# tif_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]

# # For each .tif video file in the folder generate the .h5 file. 
# for file_name in tif_files: 
#     tif_path = folder_path + file_name
#     with tifffile.TiffFile(tif_path) as tif:
#         frames = tif.asarray()  # shape: (num_frames, height, width)
            
#     # Save each frame as a slice in the .h5 file
#     new_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/"
#     name_without_suffix = os.path.splitext(file_name)[0]
#     h5_path = new_folder + name_without_suffix + ".h5"
#     with h5py.File(h5_path, "w") as h5f:
#         # Option 1: Store all frames in one dataset (like a 3D array)
#         h5f.create_dataset("img", data=frames, dtype=frames.dtype)


# # Cropping single organoid based on the bounding box from Mask-RCNN predictions 

# In[5]:


# Option 1: Firstly convert it into .h5 file then open. 
# # open h5 files.
# h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/20181113_HCC_d0-2_t43c1_ORG.h5"
# with h5py.File(h5_file_path, "r") as h5f:
#     images_t1 = h5f["img"][:]


# # Option 2: Directly apply the .tif file. 
# tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t42c1_ORG.tif"
# with tifffile.TiffFile(tif_path) as tif:
#     images_t1 = tif.asarray()  # shape: (num_frames, height, width)


# In[6]:


# h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/20181113_HCC_d0-2_t43c1_ORG.h5"
# with h5py.File(h5_file_path, "r") as h5f:
#     images_t2 = h5f["img"][:]


# # Option 2: Directly apply the .tif file. 
# tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t43c1_ORG.tif"
# with tifffile.TiffFile(tif_path) as tif:
#     images_t2 = tif.asarray()  # shape: (num_frames, height, width)


# In[7]:


# bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

# with h5py.File(bbox_h5_path, "r") as h5f:
#     t1_x1, t1_y1, t1_z1, t1_x2, t1_y2, t1_z2 = h5f["organoid_10"]["42"]["bbox"][:]
#     t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2 = h5f["organoid_10"]["43"]["bbox"][:]

# plt.imshow(images_t1[t1_z1+10][t1_y1:t1_y2, t1_x1:t1_x2])


# In[2]:


# Crop a single organoid based on the bbox from prediction. 

def crop_organoid(img_stack, x1, y1, z1, x2, y2, z2):
    org_stack = []
    for z in range(z1, z2+1): # Include the z2 slice. 
        org_stack.append(img_stack[z][y1:y2, x1:x2]) # Attention the orientation of coordinates. 
    # Volume as a 3D numpy array at the end. 
    volume = np.stack(np.array(org_stack), axis=0)
    return volume


# In[10]:


# volume_t1 = crop_organoid(images_t1, t1_x1, t1_y1, t1_z1, t1_x2, t1_y2, t1_z2)
# volume_t2 = crop_organoid(images_t2, t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2)

# np.shape(volume_t1)
# plt.imshow(volume_t1[10][:])


# In[12]:


# # Store the volume as .h5 file. 

# with h5py.File("organoid10.h5", "w") as file:
#     file.create_dataset("t42", data=volume_t1)
#     file.create_dataset("t43", data=volume_t2)


# In[3]:


# Convert the vloume in .vti file for visualisation e.g. in Paraview. 
# ---------------  pyvista version 0.42.3 ---------------------
# def convert_vtk(volume, vtk_name):
#     depth, height, width = volume.shape
    
#     # Create a PyVista UniformGrid (3D volume)
#     grid = pv.UniformGrid()
#     grid.dimensions = [width, height, depth]  # Make sure this is (X, Y, Z) order
#     grid.spacing = [1, 1, 2]  # You can adjust the spacing as needed
    
#     # Flatten the 3D volume and assign it to cell data (or point data)
#     grid.point_data["values"] = volume.flatten()  
    
#     # Add the scalar values to the grid
    
#     # Save the volume as a .vti file
#     grid.save(vtk_name + ".vti")
#     print("Volume saved as output_example.vti!")

# ----------- pv version 0.44.1 --------------- .vti file not working for the whole stack, only a part of.
def convert_vtk(volume, vtk_name):
    # Create the ImageData
    grid = pv.ImageData()
    # Set the dimension. 
    # depth, height, width = volume.shape
    grid.dimensions = np.array(volume.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)
    grid.spacing = (2, 1, 1) # Scaling fator 2 in the depth direction. 
    # Flatten the 3D volume and assign it to cell data (or point data)
    grid.cell_data["values"] = volume.flatten(order="F")
    # Save the volume as a .vti file
    grid.save(vtk_name + ".vti")
    print("Volume saved as " + vtk_name + ".vti! Open in ParaView!")
    # grid information.
    print(grid)


# In[23]:


# convert_vtk(volume_t1, "tp42")
# convert_vtk(volume_t2, "tp43") # TODO: Run for pyvista version 0.44.1


# In[4]:


# Path to the predictions. 
bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

# Path to the folder containing the original organoid .tif files. 
data_file_name = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/"

# Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
vtk_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/"

# Path for saving the volume numerical data as .h5 file. 
new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/" 

# Generate multiple vtk files over multiple time points. 
def single_organoid_all_tp(organoid): # organoid: str. from the tree of hd5f prediction. 
    # Extract all existing time points for one organoid. 
    with h5py.File(bbox_h5_path, "r") as h5f:
        single_organoid_layer = h5f[organoid]
        all_tp = list(single_organoid_layer.keys())
    # Test with the first 3 time points (t42, t43, t44). 
    test_tp = all_tp[0:3]
    
    # Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
    new_vtk_folder = vtk_folder + organoid
    os.makedirs(new_vtk_folder)
    
    with h5py.File(new_h5_folder+organoid+".h5", "w") as file:
        for tp in test_tp: 
            # Access to the original .tif video for timepoint tp. 
            tif_file_name = "20181113_HCC_d0-2_t"+tp+"c1_ORG.tif"
            tif_file_path = data_file_name + tif_file_name
            with tifffile.TiffFile(tif_file_path) as tif:
                images = tif.asarray()  # shape: (num_frames, height, width)
    
            # Extract the bounding box for single organoid at one timepoint tp. 
            with h5py.File(bbox_h5_path, "r") as h5f:
                x1, y1, z1, x2, y2, z2 = h5f[organoid][tp]["bbox"][:]
    
            # Crop the organoid over time based on the bounding box range from predictions. 
            volume = crop_organoid(images, x1, y1, z1, x2, y2, z2)
            file.create_dataset("tp"+tp, data=volume)
            
            convert_vtk(volume, new_vtk_folder+"/tp"+tp)


# In[5]:


single_organoid_all_tp("organoid_10")


# In[ ]:




