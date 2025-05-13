#!/usr/bin/env python
# coding: utf-8

# In[41]:


import h5py
import numpy as np
import matplotlib.pyplot as plt 
import pyvista as pv
import os 


# # Extract Mask from the prediction for a single organoid

# In[27]:


mask_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

with h5py.File(mask_h5_path, "r") as h5f:
    mask_1 = h5f["organoid_10"]["42"]["mask"][:].astype(int)
# Shape of mask (Δz, Δy, Δz)    
np.shape(mask_1)
# Show a single slice along z stack. 
plt.imshow(mask_1[20, :, :])
print(mask_1[0])


# In[31]:


# # Crop a single organoid based on the bbox from prediction. 

# def mask_organoid(img_stack):
#     z_range = np.shape(img_stack)[0]
#     org_stack = []
#     for z in range(z_range): # Include the z2 slice. 
#         org_stack.append(img_stack[z]) # Attention the orientation of coordinates. 
#     # Volume as a 3D numpy array at the end. 
#     volume = np.stack(np.array(org_stack), axis=0)
#     return volume


# In[28]:


# Export the volume mask as .vtk for visualising in ParaView. 

# ----------- pv version 0.44.1 --------------- .vti file not working for the whole stack, only a part of.
def convert_vtk(volume, vtk_name):
    # Create the ImageData
    grid = pv.ImageData()
    # Set the dimension. 
    # depth, height, width = volume.shape
    grid.dimensions = np.array(volume.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)
    grid.spacing = (2, 1, 1) # Scaling fator 2 in the z direction. 
    # Flatten the 3D volume and assign it to cell data (or point data)
    grid.cell_data["values"] = volume.flatten(order="F")
    # Save the volume as a .vti file
    grid.save(vtk_name + ".vti")
    print("Volume saved as " + vtk_name + ".vti! Open in ParaView!")
    # grid information.
    print(grid)


# In[37]:


convert_vtk(mask_1, "mask_organoid10_tp42")


# In[42]:


# The mask over time. 

# Path to the predictions. 
mask_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

# Path to the folder containing the original organoid .tif files. 
data_file_name = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/"

# Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
vtk_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_vti/"

# Path for saving the volume numerical data as .h5 file. 
new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/" 

# Generate multiple vtk files over multiple time points. 
def single_organoid_mask_all_tp(organoid): # organoid: str. from the tree of hd5f prediction. 
    # Extract all existing time points for one organoid. 
    with h5py.File(mask_h5_path, "r") as h5f:
        single_organoid_layer = h5f[organoid]
        all_tp = list(single_organoid_layer.keys())
    # Test with the first 3 time points (t42, t43, t44). 
    test_tp = all_tp[0:3]
    
    # Path for saving the vtk for the organoid. Generate for each organoid a new folder for storing the corresponding .vtk files. 
    new_vtk_folder = vtk_folder + organoid + "_mask" # Notice the mask in the file name. 
    os.makedirs(new_vtk_folder)
    
    with h5py.File(new_h5_folder+organoid+"_mask.h5", "w") as file:
        for tp in test_tp: 
            # Extract the bounding mask for single organoid at one timepoint tp. 
            with h5py.File(mask_h5_path, "r") as h5f:
                mask = h5f[organoid][tp]["mask"][:].astype(int)
    
            file.create_dataset("tp"+tp, data=mask)
            convert_vtk(mask, new_vtk_folder+"/tp"+tp)
            


# In[43]:


single_organoid_mask_all_tp("organoid_10")


# # Overlap the mask with the original image intensities. 

# In[69]:


# Overlap the original image and mask at one time point. 

def overlap_organoid_with_mask(organoid):
    # File adresses:
    organoid_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/"+organoid+".h5"
    mask_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/"+organoid+"_mask.h5"
    new_h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/"
    
    # Extract all existing time points for one organoid. 
    with h5py.File(organoid_h5_path, "r") as h5f:
        all_tp = list(h5f.keys()) # List of stings "tp+number".
    
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
            
            # Convert the organoid with mask into .vti file and visualise it in ParaView. 
            convert_vtk(organoid_with_mask, organoid+"_with_mask")
        
            # Store it in .h5 file. 
            file.create_dataset(tp, data=organoid_with_mask)
        


# In[70]:


org10_with_mask = overlap_organoid_with_mask("organoid_10")


# In[72]:


h5_folder = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_h5_video/organoid_10_with_mask.h5"
with h5py.File(h5_folder, "r") as h5f:
    organoid = h5f["tp42"][:]
plt.imshow(organoid[30])


# In[ ]:




