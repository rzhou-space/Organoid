#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import h5py
import tifffile


# # Single time all organoids crop

# In[9]:


# # Import the original tif file at one time point.

# tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t42c1_ORG.tif"
# with tifffile.TiffFile(tif_path) as tif:
#     images = tif.asarray()  # shape: (num_frames, height, width)
    
# # Import the predictions of the detected organoids. 

# bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"
# with h5py.File(bbox_h5_path, 'r') as f:
#     # Get all top-level names (groups or datasets)
#     top_level_keys = list(f.keys())
#     print("Detected Number of Organoids:", len(top_level_keys))
#     bbx = f["organoid_10"]["42"]["bbox"][:]


# In[2]:


def find_org_at_tp(bbox_h5_path, tp):
    # Sort out the detected organoids at given time point. 
    
    org_at_tp = []
    
    with h5py.File(bbox_h5_path, 'r') as f:
        # Get all top-level names (groups or datasets)
        top_level_keys = list(f.keys())
        # Iterate over all organoids. 
        for org in top_level_keys:
            single_organoid_layer = f[org]
            # Get all time points that is under organoid org. 
            all_tp = list(single_organoid_layer.keys())
            if tp in all_tp:
                org_at_tp.append(org)
    return org_at_tp


# In[23]:


def collect_coordinates(bbox_h5_path, tp):
    # collect the coordinates of all bounding boxes and corresponding organoids. Buil list of tupples. 
    boxes = []
    # Find out the organoids that are present at the given time point tp. 
    org_at_tp = find_org_at_tp(bbox_h5_path, tp)
    
    with h5py.File(bbox_h5_path, 'r') as f:
        # Iterate over all organoids that are present at tp. 
        for org in org_at_tp:
            x1, y1, z1, x2, y2, z2 = f[org][tp]["bbox"][:] # Coordinates of the bounding box. 
            boxes.append(((x1, y1, z1), (x2, y2, z2), org)) # Save the corners of bbox and organoids as tuple. 
    return boxes


# In[15]:


def bbox_mask_single_box(bbox_h5_path, tp, images):
    # Create an empty volume with the same shape as the input original data. 
    empty_volume = np.zeros((np.shape(images)))
    # collect all the bounding boxes that present at the given time point tp. 
    boxes = collect_coordinates(bbox_h5_path, tp)
    # Assign values to each box in the volume. 
    for i, (start, end, label) in enumerate(boxes, start=1):
        x1, y1, z1 = start
        x2, y2, z2 = end
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        zmin, zmax = sorted([z1, z2])
        empty_volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = i # Each bounding box an individual value.
    return empty_volume


# In[14]:


def bbox_mask(bbox_h5_path, tp, images):
    # Create an empty volume with the same shape as the input original data. 
    empty_volume = np.zeros((np.shape(images)))
    # collect all the bounding boxes that present at the given time point tp. 
    boxes = collect_coordinates(bbox_h5_path, tp)
    # Assign values to each box in the volume. 
    for (start, end, label) in enumerate(boxes, start=1):
        x1, y1, z1 = start
        x2, y2, z2 = end
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        zmin, zmax = sorted([z1, z2])
        empty_volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = 1 # Each bounding box the same value 1.
    return empty_volume


# In[11]:


def overlap_bbox_mask(prediction_file, original_tif_file, tp):
    # The original tif images at time point tp. 
    with tifffile.TiffFile(original_tif_file) as tif:
        images = tif.asarray()  # shape: (num_frames, height, width)
    # make bounding box mask from the prediction h5 file for the given time point tp. 
    mask_volume = bbox_mask(prediction_file, tp, images)
    # Overlap the original volome at time point tp with the bounding box mask. 
    return images*mask_volume


# In[12]:


tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t42c1_ORG.tif"
bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"
overlap = overlap_bbox_mask(bbox_h5_path, tif_path, "42")
plt.imshow(overlap[200])


# In[ ]:


# TODO: Plot in 3D and Labeling! The plotly is not working!


# In[18]:


tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t42c1_ORG.tif"
with tifffile.TiffFile(tif_path) as tif:
    images = tif.asarray()  # shape: (num_frames, height, width)
bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"
volume = bbox_mask_single_box(bbox_h5_path, "42", images)


# In[22]:


plt.imshow(volume[500])


# In[ ]:


import plotly.graph_objects as go
# Present the organoid bounding boxes in 3D. 
boxes = collect_coordinates(bbox_h5_path, "42")
# Plot all labeled voxels
fig = go.Figure()
for i, (_, _, label) in enumerate(boxes, start=1):
    x, y, z = np.where(volume == i) # TODO: Can make it quicker when directly use the know coordinates of the single boxes! And the individual value assigining could also be let out!
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=4),
        name=label,
        text=[label] * len(x),  # Show label on hover
        hoverinfo='text'
    ))

fig.update_layout(
    title="Multiple Labeled 3D Volumes",
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    width=800,
    height=800
)

fig.write_html("Bounding Boxes.html", auto_open=True)


# In[ ]:





# In[ ]:




