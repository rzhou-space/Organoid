#!/usr/bin/env python
# coding: utf-8

# In[11]:


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

# In[22]:


# Option 1: Firstly convert it into .h5 file then open. 
# # open h5 files.
# h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/20181113_HCC_d0-2_t43c1_ORG.h5"
# with h5py.File(h5_file_path, "r") as h5f:
#     images_t1 = h5f["img"][:]


# Option 2: Directly apply the .tif file. 
tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t42c1_ORG.tif"
with tifffile.TiffFile(tif_path) as tif:
    images_t1 = tif.asarray()  # shape: (num_frames, height, width)


# In[5]:


# h5_file_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/20181113_HCC_d0-2_t43c1_ORG.h5"
# with h5py.File(h5_file_path, "r") as h5f:
#     images_t2 = h5f["img"][:]


# # Option 2: Directly apply the .tif file. 
# tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t43c1_ORG.tif"
# with tifffile.TiffFile(tif_path) as tif:
#     images_t2 = tif.asarray()  # shape: (num_frames, height, width)


# In[23]:


bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

with h5py.File(bbox_h5_path, "r") as h5f:
    t1_x1, t1_y1, t1_z1, t1_x2, t1_y2, t1_z2 = h5f["organoid_10"]["42"]["bbox"][:]
#     t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2 = h5f["organoid_10"]["43"]["bbox"][:]

plt.imshow(images_t1[t1_z1 + 20][t1_y1:t1_y2, t1_x1:t1_x2])


# In[24]:


# Crop a single organoid based on the bbox from prediction. 

def crop_organoid(img_stack, x1, y1, z1, x2, y2, z2):
    org_stack = []
    for z in range(z1, z2+1): # Include the z2 slice. 
        org_stack.append(img_stack[z][y1:y2, x1:x2]) # Attention the orientation of coordinates. 
    # Volume as a 3D numpy array at the end. 
    volume = np.stack(np.array(org_stack), axis=0)
    return volume # Z, Y, X dimensions. 


# In[33]:


volume_t1 = crop_organoid(images_t1, t1_x1, t1_y1, t1_z1, t1_x2, t1_y2, t1_z2)
# volume_t2 = crop_organoid(images_t2, t2_x1, t2_y1, t2_z1, t2_x2, t2_y2, t2_z2)

np.shape(volume_t1)


# In[34]:


volume_t1


# In[12]:


# # Store the volume as .h5 file. 

# with h5py.File("organoid10.h5", "w") as file:
#     file.create_dataset("t42", data=volume_t1)
#     file.create_dataset("t43", data=volume_t2)


# # Padding the volumes based on the maximal moving range. 
# Padding the volumes or all the 2D stacks so that all the images/volumes are included in the region. Hereby find out the most lowerst corner (minimal of three coordinates) of all bounding boxes and the upperst corner (maximal of three coordinates). Only padd the part of empty space while keeping the original position of single image/volume unchanged. 

# In[20]:


# Organoid of intrest. 
organoid = "organoid_10"

def maximal_region(organoid): # Find the corners of the maximal region that contains all the organoids. 
    # Create a pandas DataFrame for the bounding boxes corners. 
    bbox_df = pd.DataFrame(columns=["x1", "y1", "z1", "x2", "y2", "z2"])
    # Get all the timepoints under the chosen organoid and extract the bounding box into a pandas Dataframe. 
    with h5py.File(bbox_h5_path, "r") as h5f:
        single_organoid_layer = h5f[organoid]
        all_tp = list(single_organoid_layer.keys()) # Get all the time points under this organoid. 
        for tp in all_tp: 
            x1, y1, z1, x2, y2, z2 = h5f[organoid][tp]["bbox"][:] # Coordinates of the bounding box. 
            bbox_df.loc[len(bbox_df)] = [x1, y1, z1, x2, y2, z2]
    
    # Find out the lowest corner and the uppest corner over all bounding boxes coordinates. 
    min_corner = np.array([bbox_df["x1"].min(), bbox_df["y1"].min(), bbox_df["z1"].min()])
    max_corner = np.array([bbox_df["x2"].max(), bbox_df["y2"].max(), bbox_df["z2"].max()])

    return min_corner, max_corner, bbox_df

maximal_region("organoid_10")


# In[31]:


# Padding the volume into the maximal movement region with zeros, while preserving the absolute position in space. 
def pad_volume(organoid, tp, image_stack, bbox_h5_path): 
    # One of the bounding box volumes. (Contained by the maximal movement region.)
    with h5py.File(bbox_h5_path, "r") as h5f:
        x1, y1, z1, x2, y2, z2 = h5f[organoid][tp]["bbox"][:]
        
    volume_small = crop_organoid(image_stack, x1, y1, z1, x2, y2, z2)

    # Coordinates of bounding boxes
    lower_large, upper_large, _ = maximal_region(organoid) # Coordinates of the maximal movement region. 
    
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

volume_padded = pad_volume("organoid_10", "42", images_t1, bbox_h5_path)
print("New shape:", volume_padded.shape)
print(shape_large)
plt.imshow(volume_padded[20])


# # Generate the vtk files with VTK packages 
# 
# The bounding boxes of the objects stay still fixed based on the first object outline. As the later points the object moving around or getting larger, the object will be cutted by the outline from the first object. 

# In[40]:


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


# # Alternative: Pyvista Package. 
# But at the end the object is not as class vtkImageData. This then influence the further adaptation into Blender with SciBlend, which need vtkImageData. This could also be the reason why the bounding box during animation noch dynamically changed with the object. 

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
    grid.save(vtk_name + ".vtk")
    print("Volume saved as " + vtk_name + ".vtk! Open in ParaView!")
    # grid information.
    print(grid)


# In[23]:


# convert_vtk(volume_t1, "tp42")
# convert_vtk(volume_t2, "tp43") # TODO: Run for pyvista version 0.44.1


# # Summary the whole function. 

# In[41]:


# Path to the predictions. 
bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

# Path to the folder containing the original organoid .tif files. 
data_file_name = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/"

# Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
vtk_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/test_results/"

# Path for saving the volume numerical data as .h5 file. 
new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/test_results/" 

# Generate multiple vtk files over multiple time points. 
def single_organoid_all_tp(organoid): # organoid: str. from the tree of hd5f prediction. 
    # Extract all existing time points for one organoid. 
    with h5py.File(bbox_h5_path, "r") as h5f:
        single_organoid_layer = h5f[organoid]
        all_tp = list(single_organoid_layer.keys())
    # Test with the first 3 time points (t42, t43, t44). 
    test_tp = ["42", "43", "44", "82"]
    
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
            print(x1, y1, z1, x2, y2, z2)
    
            # Crop the organoid over time based on the bounding box range from predictions. 
            volume = crop_organoid(images, x1, y1, z1, x2, y2, z2)
            file.create_dataset("tp"+tp, data=volume)
            
            convert_vtk(volume, new_vtk_folder+"/tp"+tp)


# In[42]:


single_organoid_all_tp("organoid_10")


# In[ ]:




