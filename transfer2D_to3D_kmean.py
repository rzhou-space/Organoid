import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# K-mean clustering segementation.
data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_image/20181113_HCC_d0-2_t42c1_ORG/"
jpg_file = "20181113_HCC_d0-2_t42c1_ORG0100.jpg"
tif_file = "20181113_HCC_d0-2_t42c1_ORG0400.tif"

img = cv.imread(data_dir + tif_file)
# print(np.shape(img))

def kmean_seg(img, K, attempts):
    # Apply k-mean clustering segementation on image.
    imgcolor=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    vectorized = imgcolor.reshape((-1,3))
    vectorized_np = np.float32(vectorized)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv.kmeans(vectorized_np, K, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    gray_result = np.array(cv.cvtColor(result_image, cv.COLOR_BGR2GRAY))
    # Normalization such that the minimum is 0 and maximal is 1. Using min-max-normalization.
    normalized_result = (gray_result - np.min(gray_result)) / (np.max(gray_result) - np.min(gray_result))

    return normalized_result

plt.imshow(kmean_seg(img,2, 10))
plt.show()
print(np.shape(kmean_seg(img, 2, 10)))
# ---------------------------------
if __name__ == "__main__":

    import os
    from natsort import os_sorted
    import pyvista as pv

    data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_image/20181113_HCC_d0-2_t42c1_ORG/"

    kmean_stack = []

    # Open the file where single .tif 2D images are. (extracted with ImageJ)
    items = [file for file in os_sorted(os.listdir(data_dir)) if file.lower().endswith(".tif")]
    for file in items[200:600]:
        img = cv.imread(data_dir + file)
        # Normalisation with cv2, so that the minimal intensity 0 and maximal intensity 255.
        kmean_img = kmean_seg(img, 2, 10)
        kmean_stack.append(np.array(kmean_img))
        print("ok"+ file)

    # Volume as a 3D numpy array at the end.
    volume = np.stack(np.array(kmean_stack), axis=0)
    print(np.shape(volume))

    # Convert the vloume in .vti file.

    # # ----------- pv version 0.42.3 -------------
    # depth, height, width = volume.shape
    #
    # # Create a PyVista UniformGrid (3D volume)
    # grid = pv.UniformGrid()
    # grid.dimensions = [width, height, depth]  # Make sure this is (X, Y, Z) order
    # grid.spacing = [1, 1, 1]  # You can adjust the spacing as needed
    # # Flatten the 3D volume and assign it to cell data (or point data)
    # grid.point_data["values"] = volume.flatten()
    # print(grid)


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
    print("Volume saved as .vti!")

