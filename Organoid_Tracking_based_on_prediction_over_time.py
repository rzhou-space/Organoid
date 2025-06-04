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



# Crop a single organoid based on the bbox from prediction. 
def crop_organoid(img_stack, x1, y1, z1, x2, y2, z2):
    org_stack = []
    for z in range(z1, z2+1): # Include the z2 slice. 
        org_stack.append(img_stack[z][y1:y2, x1:x2]) # Attention the orientation of coordinates. 
    # Volume as a 3D numpy array at the end. 
    volume = np.stack(np.array(org_stack), axis=0)
    return volume # Z, Y, X dimensions. 


    
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



# # Cropping the organoid based on the bounding box and paddin to its maximal movement region. 

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

            # Pad the single volume at time tp to the maximal movement region while preserve the absolute position. 
            volume_padded = pad_volume(organoid, tp, images, bbox_h5_path, min_corner, max_corner)
            print(volume_padded.shape)
    
            file.create_dataset("tp"+tp, data = volume_padded)
            
            apply_vtk(volume_padded, new_vtk_folder+"/tp"+tp)


single_organoid_all_tp("organoid_10")



# # Loading Masks and Padding masks to the maximal movement regions. 

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


# # The padded mask over time. 

# Path to the predictions. 
bbox_mask_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

# Path to the folder containing the original organoid .tif files. 
data_file_name = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/"

# Path for saving the .vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
vtk_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"

# Path for saving the volume numerical data as .h5 file. 
new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/Code/Organoid/notebooks/"

# Generate multiple vtk files over multiple time points. 
def single_organoid_mask_all_tp(organoid): # organoid: string from the tree of hd5f prediction. 
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


single_organoid_mask_all_tp("organoid_10")


# # Overlap the Mask and the Original Intensity 
# # Apply hereby the padded volumes, which also have the same shapes.

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
        


overlap_organoid_with_mask("organoid_10")






