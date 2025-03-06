import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# K-mean clustering segementation.
data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_jpg_image/20181113_HCC_d0-2_t42c1_ORG/"
jpg_file = "20181113_HCC_d0-2_t42c1_ORG0000.jpg"
tif_file = "20181113_HCC_d0-2_t42c1_ORG0400.tif"

img = cv.imread(data_dir + jpg_file)

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

    return result_image

#plt.imshow(kmean_seg(img, 2, 10))
#plt.show()

# ---------------------------------
if __name__ == "__main__":

    import os
    from natsort import os_sorted

    data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_jpg_image/20181113_HCC_d0-2_t42c1_ORG/"

    kmean_stack = []

    # Open the file where single .tif 2D images are. (extracted with ImageJ)
    items = [file for file in os_sorted(os.listdir(data_dir)) if file.lower().endswith(".jpg")]
    for file in items:
        img = cv.imread(data_dir + file)
        # Normalisation with cv2, so that the minimal intensity 0 and maximal intensity 255.
        kmean_img = kmean_seg(img, 2, 10)
        kmean_stack.append(np.array(kmean_img))
        print("ok"+ file)

    # Volume as a 3D numpy array at the end.
    volume = np.stack(np.array(kmean_stack), axis=0)
    np.shape(volume)

    # Convert the vloume in .vti file.
    depth, height, width = volume.shape

    # Create a PyVista UniformGrid (3D volume)
    grid = pv.UniformGrid()
    grid.dimensions = [width, height, depth]  # Make sure this is (X, Y, Z) order
    grid.spacing = [1, 1, 1]  # You can adjust the spacing as needed

    # Flatten the 3D volume and assign it to cell data (or point data)
    grid.point_data["values"] = volume.flatten()

    # Add the scalar values to the grid

    # Save the volume as a .vti file
    grid.save("output_kmean.vti")
    print("Volume saved as output_kmean.vti!")