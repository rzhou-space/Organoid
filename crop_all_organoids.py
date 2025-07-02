#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import h5py
import tifffile


# # Single time all organoids crop


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


def bbox_mask(bbox_h5_path, tp, images):
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
        empty_volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = 1 # Each bounding box the same value 1.
    return empty_volume


def overlap_bbox_mask(prediction_file, original_tif_file, tp):
    # The original tif images at time point tp. 
    with tifffile.TiffFile(original_tif_file) as tif:
        images = tif.asarray()  # shape: (num_frames, height, width)
    # make bounding box mask from the prediction h5 file for the given time point tp. 
    mask_volume = bbox_mask(prediction_file, tp, images)
    # Overlap the original volome at time point tp with the bounding box mask. 
    return np.array(images * mask_volume)


tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t42c1_ORG.tif"
bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"

box_coordinate = collect_coordinates(bbox_h5_path, "42")



# TODO: When the size of the whole 3D image is too large, can also make it possible for subregions presentation. 
# TODO: Possible to overlap with original images?


from vedo import Box, Text3D, show

boxes = []
labels = []

for box in box_coordinate:
    boxes.append(box[0]+box[1])
    labels.append(box[2][-2:])
print(boxes)
print(labels)


actors = []

for i, (x1,y1,z1,x2,y2,z2) in enumerate(boxes):
    box = Box(pos=[(x1+x2)/2, (y1+y2)/2, (z1+z2)/2],
              length=abs(x2-x1), width=abs(y2-y1), height=abs(z2-z1),
              c='lightblue', alpha=0.3)
    actors.append(box)
    
    label_pos = (x2 + 0.1, y2 + 0.1, z2 + 0.1)  # offset to be outside the box
    label = Text3D(labels[i], pos=label_pos, s=20, c='red')
    actors.append(label)

show(actors)