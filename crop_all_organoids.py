#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import h5py
import tifffile
from vedo import Box, Text3D, show
import re
import pyvista as pv


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


def find_org_at_tp(organoid_list, tp):
    # Sort out the detected organoids at given time point. 
    
    org_at_tp = []
    
    with h5py.File(bbox_h5_path, 'r') as f:
        # Get all top-level names (groups or datasets)
        # top_level_keys = list(f.keys())
        # Iterate over all organoids. 
        for org in organoid_list:
            single_organoid_layer = f[org]
            # Get all time points that is under organoid org. 
            all_tp = list(single_organoid_layer.keys())
            if tp in all_tp:
                org_at_tp.append(org)
    return org_at_tp


def collect_coordinates(organoid_list, tp):
    # collect the coordinates of all bounding boxes and corresponding organoids. Build list of tupples. 
    # organoid list: ["organoid_id", ...]
    boxes = []
    organoid_at_tp = find_org_at_tp(organoid_list, tp)

    with h5py.File(bbox_h5_path, 'r') as f:
        # Iterate over all organoids that are present at tp. 
        for org in organoid_at_tp:
            x1, y1, z1, x2, y2, z2 = f[org][tp]["bbox"][:] # Coordinates of the bounding box. 
            boxes.append(((x1, y1, z1), (x2, y2, z2), org)) # Save the corners of bbox and organoids as tuple. 
    return boxes


# def bbox_mask_single_box(bbox_h5_path, tp, images):
#     # Create an empty volume with the same shape as the input original data. 
#     empty_volume = np.zeros((np.shape(images)))
#     # collect all the bounding boxes that present at the given time point tp. 
#     boxes = collect_coordinates(bbox_h5_path, tp)
#     # Assign values to each box in the volume. 
#     for i, (start, end, label) in enumerate(boxes, start=1):
#         x1, y1, z1 = start
#         x2, y2, z2 = end
#         xmin, xmax = sorted([x1, x2])
#         ymin, ymax = sorted([y1, y2])
#         zmin, zmax = sorted([z1, z2])
#         empty_volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = i # Each bounding box an individual value.
#     return empty_volume


# def bbox_mask(bbox_h5_path, tp, images):
#     # Create an empty volume with the same shape as the input original data. 
#     empty_volume = np.zeros((np.shape(images)))
#     # collect all the bounding boxes that present at the given time point tp. 
#     boxes = collect_coordinates(bbox_h5_path, tp)
#     # Assign values to each box in the volume. 
#     for i, (start, end, label) in enumerate(boxes, start=1):
#         x1, y1, z1 = start
#         x2, y2, z2 = end
#         xmin, xmax = sorted([x1, x2])
#         ymin, ymax = sorted([y1, y2])
#         zmin, zmax = sorted([z1, z2])
#         empty_volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = 1 # Each bounding box the same value 1.
#     return empty_volume


# def overlap_bbox_mask(prediction_file, original_tif_file, tp):
#     # The original tif images at time point tp. 
#     with tifffile.TiffFile(original_tif_file) as tif:
#         images = tif.asarray()  # shape: (num_frames, height, width)
#     # make bounding box mask from the prediction h5 file for the given time point tp. 
#     mask_volume = bbox_mask(prediction_file, tp, images)
#     # Overlap the original volome at time point tp with the bounding box mask. 
#     return np.array(images * mask_volume)


# # VERY IMPORTANT ONE!
def filter_bbox(bbox_h5_path, min_duration, min_volume):
    # Filter the bounding boxes (over all predictioins and whole time period) given the thresholds of duration and volume. 
    filtered_bbox = []
    with h5py.File(bbox_h5_path, 'r') as f:
        # Get all top-level names (all organoid ids)
        top_level_keys = list(f.keys())

        # Iterate over all organoids. 
        for org in top_level_keys:
            single_organoid_layer = f[org]
            # Get the time duration that is recorded under organoid org. 
            all_tp = list(single_organoid_layer.keys())
            duration = len(all_tp)

            if duration >= min_duration: # First filtering through minminal time duration. 

                # calculate the mean volume of bounding box after the first filtering through time duration.
                all_volume = []
                for tp in all_tp:
                    x1, y1, z1, x2, y2, z2 = f[org][tp]["bbox"][:]
                    volume = abs(x2-x1) * abs(y2-y1) * abs(z2-z1) # in voxel
                    all_volume.append(volume)
                # The mean volume over all existing time points.
                mean_volume = round(np.mean(np.array(all_volume)))

                # Second filtering through the minimal volume.
                if mean_volume >= min_volume:
                    filtered_bbox.append([duration, mean_volume, org])

    return filtered_bbox # Bounding Boxes with duration and size larger than given thresholds. Recoorded in the formate [duration, mean volume in voxels, organoid_id]


def plot_annotation_bounding_box(box_coordinate):
    # Plotting the bounding boxes with its annotation (the id of organoids) in interactive 3D image. 
    # box_coordinate has the tuple formate: ((x1, y1, z1), (x2, y2, z2), 'organoid_id'), where x, y, z the coordinate of the bounding box corner coordinates. 

    boxes = []
    labels = []

    for box in box_coordinate:
        boxes.append(box[0]+box[1])
        id = re.search(r'(\d{1,3})$', box[2])
        labels.append(id.group(1))

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


def plot_filtered_bounding_box_vedo(bbox_h5_path, min_duration, min_volume, tp):
    # 3D interactive plots with annotation through vedo for the filtered bounding boxes. Only for one time point. 
    # Generate the list of boxes after filtering through duration and mean volume.
    filtered_box = filter_bbox(bbox_h5_path, min_duration, min_volume)
    # Find the corresponding bounding box coordinates based on the organoids_id in after filtering. 
    filtered_box_id = []
    for box in filtered_box:
        filtered_box_id.append(box[2])
    # Plot the filtered bounding boxes for a time point. 
    filtered_box_coord = collect_coordinates(filtered_box_id, tp)
    plot_annotation_bounding_box(filtered_box_coord)


def plot_filtered_bounding_box_paraview(bbox_h5_path, t_start, t_end, 
                                        min_duration, min_volume, vtk_save_path):
    ## 3D interactive image without annotations, only bounding box outlines. Visualization through ParaView. For all time points. 
    ## t_start, t_end: int. for the start and end time points for bounding box visualisation.
    ## min_duration, min_volume: int. the thresholds for filtering bounding boxes.
    ## vtk_save_path: string. the path for saving the resulting vtk. files. If necessary creating new folder. 

    # Generate the list of filtered bounding boxes based on minimal duratioin and mean volume. 
    filtered_box = filter_bbox(bbox_h5_path, min_duration, min_volume)
    # Get the corresponding bounding box id list. 
    filtered_box_id = []
    for box in filtered_box:
        filtered_box_id.append(box[2])

    # Iterrate over all time points from t_start to t_end and generate the outlines at each time point. 
    for tp in [str(t) for t in range(t_start, t_end+1)]: 
        # The coordinates of filtered boxes that appears at time point tp. 
        filtered_box_coord = collect_coordinates(filtered_box_id, tp)
        # Create a single MultiBlock dataset to hold all outlines
        outline_blocks = pv.MultiBlock()

        for box in filtered_box_coord: # [(x1, y1, z1), (x2, y2, z2), 'organoid_id']
            x1, y1, z1 = box[0][:]
            x2, y2, z2 = box[1][:]
            # Get the bounds of the box.
            bounds = [x1, x2, y1, y2, z1, z2]
            # Create a box from bounds
            box = pv.Box(bounds=bounds)
            # Extract the outline (wireframe)
            outline = box.extract_feature_edges()
            # Add it to the multiblock
            outline_blocks.append(outline)

        # Combine all outlines into a single mesh
        all_outlines = outline_blocks.combine()
        # Save as vtk files.
        all_outlines.save(vtk_save_path + "/tp" + tp + ".vtk")


def convert_vtk(volume, vtk_name):
    # Create the ImageData
    grid = pv.ImageData()
    # Set the dimension. 
    # depth, height, width = volume.shape
    grid.dimensions = np.array(volume.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1) # Scaling fator 2 in the depth direction. 
    # Flatten the 3D volume and assign it to cell data (or point data)
    grid.cell_data["values"] = volume.flatten(order="F")
    # Save the volume as a .vti file
    grid.save(vtk_name + ".vti")
    print("Volume saved as " + vtk_name + ".vti! Open in ParaView!")
    # grid information.
    print(grid)


def filtered_organoids_paraview(bbox_h5_path, tif_file_folder, t_start, t_end, 
                                min_duration, min_volume, vtk_save_path):
    ## Extract the organoids based on the filtering of the bounding box. 
    ## tif_file_folder: the path to the folder where all original .tif files are stored. 
    ## vtk_save_path: the path to the folder that schould store all generated .vtk files. Set new folder if necessary. 

    # Generate the list of filtered bounding boxes based on minimal duratioin and mean volume. 
    filtered_box = filter_bbox(bbox_h5_path, min_duration, min_volume)
    # Get the corresponding bounding box id list. 
    filtered_box_id = []
    for box in filtered_box:
        filtered_box_id.append(box[2])

    # Iterrate over all time points from t_start to t_end and generate the outlines at each time point. 
    for tp in [str(t) for t in range(t_start, t_end+1)]: 

        # Load the original image data. 
        file_name = "20181113_HCC_d0-2_t"+tp+"c1_ORG.tif"
        img_path = tif_file_folder + file_name
        with tifffile.TiffFile(img_path) as tif:
            img = tif.asarray()  # shape: (depth(z), height(y), width(x))

        # The coordinates of filtered boxes that appears at time point tp. 
        filtered_box_coord = collect_coordinates(filtered_box_id, tp)
            
        # Create an empty volume same shape as original, filled with zeros
        combined = np.zeros_like(img)
        for box in filtered_box_coord: # [(x1, y1, z1), (x2, y2, z2), 'organoid_id']
            x1, y1, z1 = box[0][:]
            x2, y2, z2 = box[1][:]
            # Extract the subvolume from the original image
            sub = img[z1:z2, y1:y2, x1:x2]
            # Paste it into the combined volume at the correct position
            combined[z1:z2, y1:y2, x1:x2] = sub
        
        convert_vtk(combined, vtk_save_path + "/tp"+ tp)



# -------------------"__main__"---------------------

if __name__ == "__main__":

    tif_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/20181113_HCC_d0-2_t42c1_ORG.tif"
    bbox_h5_path = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/predictions/organoid4D_tp_42_82_z_0_870_min_preds_10.h5"
    
    ##  ATTENTION: When overlap two objects in ParaView: the coodinates are shifted. Has to set bounding boxes under Transformation: scale: (-1, 1, 1) and orientation: (0, 90, 0)
    plot_filtered_bounding_box_paraview(bbox_h5_path, 42, 43, 30, 50*50*50, 
                                       "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/bounding_boxes/filter_test")

    filtered_organoids_paraview(bbox_h5_path, 
                                "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_91_tif_video/original_tif_data/",
                                42, 43, 30, 50*50*50,
                                "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/bounding_boxes/org_crop_test")

    ## ATTENTION: vedo: bounding boxes + annotations (organoid_id) --> not suitable for large microscopy images.
    ##  ATTENTION: ParaView: bounding boxes + organoids image --> not suitable for annotation shown as texts. 


