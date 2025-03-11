import cv2
from matplotlib import pyplot as plt
import numpy as np
import zipfile
import os
from natsort import os_sorted
import pyvista as pv
import h5py

def generate_volume(data_dir):
    img_stack = []

    # Open the file where single .tif 2D images are. (extracted with ImageJ)
    items = [file for file in os_sorted(os.listdir(data_dir)) if file.lower().endswith(".tif")]
    for file in items:
        img = cv2.imread(data_dir + file, 0)
        # Normalisation with cv2, so that the minimal intensity alpha and maximal intensity beta.
        # img_scaled = cv2.normalize(img, dst=True, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        img_stack.append(np.array(img).astype(np.float32))
        print("ok" + file)

    # Volume as a 3D numpy array at the end.
    volume = np.stack(np.array(img_stack), axis=0)

    return volume

def generate_vti(data_dir):
    volume = generate_volume(data_dir)

    # # ------ pyvista version 0.42.3 -------------
    # # Convert the vloume in .vti file.
    # depth, height, width = volume.shape
    #
    # # Create a PyVista UniformGrid (3D volume)
    # grid = pv.UniformGrid()
    # grid.dimensions = [width, height, depth]  # Make sure this is (X, Y, Z) order
    # grid.spacing = [1, 1, 1]  # You can adjust the spacing as needed
    #
    # # Flatten the 3D volume and assign it to cell data (or point data)
    # grid.point_data["values"] = volume.flatten()

    # ----------- pv version 0.44.1 --------------- .vti file not working for the whole stack, only a part of.
    # Create the ImageData
    grid = pv.ImageData()
    # Set the dimension. Make sure this is (X, Y, Z) order so that slice along Z-axis.
    # depth, height, width = volume.shape
    grid.dimensions = np.array(volume.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    # Flatten the 3D volume and assign it to cell data (or point data)
    grid.cell_data["values"] = volume.flatten(order="F")
    # grid information.
    print(grid)

    # Save the volume as a .vti file
    grid.save("output.vti")
    print("Volume saved as output.vti! Open in ParaView!")

def save_h5(volume):
    with h5py.File("output_organoid.h5", "w") as file:
        file.create_dataset("data", data=volume)


if __name__ == "__main__":

    data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_image/20181113_HCC_d0-2_t42c1_ORG(crop)/"
    generate_vti(data_dir)
