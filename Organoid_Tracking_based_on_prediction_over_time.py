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
import pandas as pd


# In[2]:


# # Force package to be a certain version. 
# pip install --force-reinstall -v "pyvista==0.44.1"


# # Converting tif into .h5 files. -- Not Necessary..
# Firstly convert .tif video into stack of .tif images in a saparate folder. 

# In[3]:


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


# In[4]:


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


# Option 2: Directly apply the .tif file. 
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
# #     t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2 = h5f["organoid_10"]["43"]["bbox"][:]

# plt.imshow(images_t1[t1_z1 + 20][t1_y1:t1_y2, t1_x1:t1_x2])


# In[8]:


# Crop a single organoid based on the bbox from prediction. 

def crop_organoid(img_stack, x1, y1, z1, x2, y2, z2):
    org_stack = []
    for z in range(z1, z2+1): # Include the z2 slice. 
        org_stack.append(img_stack[z][y1:y2, x1:x2]) # Attention the orientation of coordinates. 
    # Volume as a 3D numpy array at the end. 
    volume = np.stack(np.array(org_stack), axis=0)
    return volume # Z, Y, X dimensions. 


# In[9]:


# volume_t1 = crop_organoid(images_t1, t1_x1, t1_y1, t1_z1, t1_x2, t1_y2, t1_z2)
# # volume_t2 = crop_organoid(images_t2, t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2)

# np.shape(volume_t1)


# In[10]:


# # Store the volume as .h5 file. 

# with h5py.File("organoid10.h5", "w") as file:
#     file.create_dataset("t42", data=volume_t1)
#     file.create_dataset("t43", data=volume_t2)


# # Padding the volumes based on the maximal moving range. 
# Padding the volumes or all the 2D stacks so that all the images/volumes are included in the region. Hereby find out the most lowerst corner (minimal of three coordinates) of all bounding boxes and the upperst corner (maximal of three coordinates). Only padd the part of empty space while keeping the original position of single image/volume unchanged. 

# In[11]:


# bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"
# # Get all the time points. 
# organoid = "organoid_10"
# with h5py.File(bbox_h5_path, "r") as h5f:
#     single_organoid_layer = h5f[organoid]
#     all_tp = list(single_organoid_layer.keys()) # Get all the time points under this organoid. 
    
# Organoid of intrest. 
def maximal_region(organoid, bbox_h5_path, all_tp): # Find the corners of the maximal region that contains all the organoids. 
    # Create a pandas DataFrame for the bounding boxes corners. 
    bbox_df = pd.DataFrame(columns=["x1", "y1", "z1", "x2", "y2", "z2"])
    # Get all the timepoints under the chosen organoid and extract the bounding box into a pandas Dataframe. 
    with h5py.File(bbox_h5_path, "r") as h5f:
        for tp in all_tp: 
            x1, y1, z1, x2, y2, z2 = h5f[organoid][tp]["bbox"][:] # Coordinates of the bounding box. 
            bbox_df.loc[len(bbox_df)] = [x1, y1, z1, x2, y2, z2]
    
    # Find out the lowest corner and the uppest corner over all bounding boxes coordinates. 
    min_corner = np.array([bbox_df["x1"].min(), bbox_df["y1"].min(), bbox_df["z1"].min()])
    max_corner = np.array([bbox_df["x2"].max(), bbox_df["y2"].max(), bbox_df["z2"].max()])

    return min_corner, max_corner, bbox_df

# min_corner, max_corner, df = maximal_region("organoid_10", bbox_h5_path, all_tp)
# print(max_corner - min_corner)
# print(df)


# In[12]:


# Padding the volume into the maximal movement region with zeros, while preserving the absolute position in space. 
def pad_volume(organoid, tp, image_stack, bbox_h5_path, lower_large, upper_large): 
    # One of the bounding box volumes. (Contained by the maximal movement region.)
    with h5py.File(bbox_h5_path, "r") as h5f:
        x1, y1, z1, x2, y2, z2 = h5f[organoid][tp]["bbox"][:]
        
    volume_small = crop_organoid(image_stack, x1, y1, z1, x2, y2, z2)

    # Coordinates of bounding boxes
    # lower_large, upper_large, _ = maximal_region(organoid, bbox_h5_path, all_tp) # Coordinates of the maximal movement region. 
    
    lower_small = np.array([x1, y1, z1])   # Coordinates of the current bounding box. 
    upper_small = np.array([x2, y2, z2]) 
    
    # Calculate size of large box / maximal movement region.
    shape_large = upper_large - lower_large  # (dx, dy, dz)
    
    # Calculate offset of the small volume inside the large one. Determine the size pad befor and after the original object. 
    pre_pad = lower_small - lower_large       # Number of voxels to pad before (z, y, x)
    post_pad = upper_large - upper_small     # Padding after
    
    # Convert to (z, y, x) order.
    pre_pad = pre_pad[::-1]
    post_pad = post_pad[::-1]
    
    # Combine padding in ((before, after), ...) for each axis. 
    padding = tuple(zip(pre_pad, post_pad))  # Format: ((z1, z2), (y1, y2), (x1, x2))
    
    # Apply padding
    volume_padded = np.pad(volume_small, pad_width=padding, mode='constant', constant_values=0) # Result volume in (z, y, x) dimension. 

    return volume_padded

# min_corner, max_corner, df = maximal_region("organoid_10", bbox_h5_path, all_tp)
# volume_padded = pad_volume("organoid_10", "50", images_t1, bbox_h5_path, min_corner, max_corner)
# print("New shape:", volume_padded.shape)
# plt.imshow(volume_padded[20])
# volume_padded


# # Generate the vtk files with VTK packages 
# 
# The bounding boxes of the objects stay still fixed based on the first object outline. As the later points the object moving around or getting larger, the object will be cutted by the outline from the first object. 

# In[13]:


# Convert the volume into .vtk with VTK package. 
import vtk
from vtk.util import numpy_support

# Create a vtkImageData object 
def apply_vtk(volume, file_path):
    # Get the dimensions
    depth = volume.shape[0]
    height = volume.shape[1]
    width = volume.shape[2]
    
    # Convert to 1D array for VTK
    flat_array = volume.flatten()  # Note: Transpose ZYX to match VTK order
    
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=flat_array)
    
    # Create vtkImageData
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(width, height, depth)
    image_data.SetSpacing(1.0, 1.0, 1.0)  # Set spacing as needed
    image_data.GetPointData().SetScalars(vtk_data_array)
    
    # Save as .vtk file 
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(file_path + ".vti")
    print("Volume saved as " + file_path + ".vti! Open in ParaView!")
    writer.SetInputData(image_data)
    writer.Write()


# In[14]:


# apply_vtk(volume_padded, "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/hi")


# # Alternative: Pyvista Package. 
# But at the end the object is not as class vtkImageData. This then influence the further adaptation into Blender with SciBlend, which need vtkImageData. This could also be the reason why the bounding box during animation noch dynamically changed with the object. 

# In[15]:


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

# # ----------- pv version 0.44.1 --------------- .vti file not working for the whole stack, only a part of.
# def convert_vtk(volume, vtk_name):
#     # Create the ImageData
#     grid = pv.ImageData()
#     # Set the dimension. 
#     # depth, height, width = volume.shape
#     grid.dimensions = np.array(volume.shape) + 1
#     # Edit the spatial reference
#     grid.origin = (0, 0, 0)
#     grid.spacing = (2, 1, 1) # Scaling fator 2 in the depth direction. 
#     # Flatten the 3D volume and assign it to cell data (or point data)
#     grid.cell_data["values"] = volume.flatten(order="F")
#     # Save the volume as a .vti file
#     grid.save(vtk_name + ".vtk")
#     print("Volume saved as " + vtk_name + ".vtk! Open in ParaView!")
#     # grid information.
#     print(grid)


# In[16]:


# convert_vtk(volume_t1, "tp42")
# convert_vtk(volume_t2, "tp43") # TODO: Run for pyvista version 0.44.1


# # Summary the whole function for cropping the organoid volumes. 
# For cropping single organoid while keeping the bounding box consistent. 

# In[17]:


# Path to the predictions. 
bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

# Path to the folder containing the original organoid .tif files. 
data_file_name = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/"

# Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
vtk_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"

# Path for saving the volume numerical data as .h5 file. 
new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/" 

# Generate multiple vtk files over multiple time points. 
def single_organoid_all_tp(organoid): # organoid: str. from the tree of hd5f prediction. 
    # Extract all existing time points for one organoid. 
    with h5py.File(bbox_h5_path, "r") as h5f:
        single_organoid_layer = h5f[organoid]
        all_tp = list(single_organoid_layer.keys())
    # Test with the first 3 time points (t42, t43, t44). 
    test_tp = ["42", "43", "44", "82"]

    # Get the maximal movement region of the organoid, defined by minimal and maximal corners.
    min_corner, max_corner, _ = maximal_region(organoid, bbox_h5_path, all_tp)

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
    
            # # Extract the bounding box for single organoid at one timepoint tp. 
            # with h5py.File(bbox_h5_path, "r") as h5f:
            #     x1, y1, z1, x2, y2, z2 = h5f[organoid][tp]["bbox"][:]
            # print(x1, y1, z1, x2, y2, z2)
            # # Crop the organoid over time based on the bounding box range from predictions. 
            # volume = crop_organoid(images, x1, y1, z1, x2, y2, z2)

            # Pad the single volume at time tp to the maximal movement region while preserve the absolute position. 
            volume_padded = pad_volume(organoid, tp, images, bbox_h5_path, min_corner, max_corner)
            print(volume_padded.shape)
    
            file.create_dataset("tp"+tp, data = volume_padded)
            
            apply_vtk(volume_padded, new_vtk_folder+"/tp"+tp)


# In[18]:


single_organoid_all_tp("organoid_10")


# # Loading Masks and Padding masks
# The volume of the mask is the same as the one defined by bounding boxes corner. 

# In[19]:


def pad_volume_mask(organoid, tp, bbox_h5_path, lower_large, upper_large): 
    # One of the bounding box volumes. (Contained by the maximal movement region.)
    with h5py.File(bbox_h5_path, "r") as h5f:
        x1, y1, z1, x2, y2, z2 = h5f[organoid][tp]["bbox"][:]
        volume_small = h5f[organoid][tp]["mask"][:].astype(int)

    # Coordinates of bounding boxes
    # lower_large, upper_large, _ = maximal_region(organoid, bbox_h5_path, all_tp) # Coordinates of the maximal movement region. 
    
    lower_small = np.array([x1, y1, z1])   # Coordinates of the current bounding box. 
    upper_small = np.array([x2, y2, z2]) 
    
    # Calculate size of large box / maximal movement region.
    shape_large = upper_large - lower_large  # (dx, dy, dz)
    
    # Calculate offset of the small volume inside the large one. Determine the size pad befor and after the original object. 
    pre_pad = lower_small - lower_large       # Number of voxels to pad before (z, y, x)
    post_pad = upper_large - upper_small     # Padding after
    
    # Convert to (z, y, x) order.
    pre_pad = pre_pad[::-1]
    post_pad = post_pad[::-1]
    
    # Combine padding in ((before, after), ...) for each axis. 
    padding = tuple(zip(pre_pad, post_pad))  # Format: ((z1, z2), (y1, y2), (x1, x2))
    
    # Apply padding
    volume_padded = np.pad(volume_small, pad_width=padding, mode='constant', constant_values=0) # Result volume in (z, y, x) dimension. 

    return volume_padded


# In[20]:


# mask_padded = pad_volume_mask("organoid_10", "42", bbox_mask_h5_path, min_corner, max_corner)
# plt.imshow(mask_padded[20])


# In[21]:


# The mask over time. 

# Path to the predictions. 
bbox_mask_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

# Path to the folder containing the original organoid .tif files. 
data_file_name = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/"

# Path for saving the .vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
vtk_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"

# Path for saving the volume numerical data as .h5 file. 
new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"

# Generate multiple vtk files over multiple time points. 
def single_organoid_mask_all_tp(organoid): # organoid: str. from the tree of hd5f prediction. 
    # Extract all existing time points for one organoid. 
    with h5py.File(bbox_mask_h5_path, "r") as h5f:
        single_organoid_layer = h5f[organoid]
        all_tp = list(single_organoid_layer.keys())
    # Test with certain time points.  
    test_tp = ["42", "43", "44", "82"]

    # Get the maximal movement region for padding. 
    min_corner, max_corner, _ = maximal_region(organoid, bbox_mask_h5_path, all_tp)
    
    # Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
    new_vtk_folder = vtk_folder + organoid + "_mask" # Notice the mask in the file name. 
    os.makedirs(new_vtk_folder)
    
    with h5py.File(new_h5_folder+organoid+"_mask.h5", "w") as file:
        for tp in test_tp: 
            # Extract the bounding mask for single organoid at one timepoint tp. 
            with h5py.File(bbox_mask_h5_path, "r") as h5f:
                mask = h5f[organoid][tp]["mask"][:].astype(int)

            mask_padded = pad_volume_mask(organoid, tp, bbox_mask_h5_path, min_corner, max_corner)
            print(mask_padded.shape)
    
            file.create_dataset("tp"+tp, data=mask_padded)
            apply_vtk(mask_padded, new_vtk_folder+"/tp"+tp)


# In[22]:


single_organoid_mask_all_tp("organoid_10") # TODO: The vtk files are sometimes not correctly generated .... Have to generate twice.


# # Overlap the Mask and the Original Intensity 
# Apply hereby the padded volumes, which also have the same shapes.

# In[23]:


# Overlap the original image and mask at one time point. 

def overlap_organoid_with_mask(organoid):
    # File paths:
    organoid_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"+organoid+".h5"
    mask_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"+organoid+"_mask.h5"
    new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"
    vtk_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"
        
    # Extract all existing time points for one organoid. 
    with h5py.File(organoid_h5_path, "r") as h5f:
        all_tp = list(h5f.keys()) # List of stings "tp+number".
        
    # Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
    new_vtk_folder = vtk_folder + organoid + "_with_mask" # Notice the mask in the file name. 
    os.makedirs(new_vtk_folder)
    
    # Based on that the original organoids data and the mask data are both stored in .h5 files. Can read out the information and overlap the arrays. 
    with h5py.File(new_h5_folder+organoid+"_with_mask.h5", "w") as file:
        for tp in all_tp:
            # The original cropped organoid bbox data. 
            with h5py.File(organoid_h5_path, "r") as h5f:
                original_organoid = h5f[tp][:]
            # The mask of the corresponding organoid. 
            with h5py.File(mask_h5_path, "r") as h5f:
                mask_organoid = h5f[tp][:]
            # Overlap the original organoid data with masks. 
            organoid_with_mask = original_organoid * mask_organoid
        
            # Store it in .h5 file. 
            file.create_dataset(tp, data=organoid_with_mask)

            # Convert the organoid with mask into .vti file and visualise it in ParaView. 
            print(organoid_with_mask.shape)
            apply_vtk(organoid_with_mask, new_vtk_folder+"/"+tp)
        


# In[24]:


overlap_organoid_with_mask("organoid_10")


# In[ ]:




