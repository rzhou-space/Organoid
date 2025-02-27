#!/usr/bin/env python
# coding: utf-8

# In[6]:


import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# Convert .h5 files to .vti.

# open h5 files.
h5_file_path = "organoid_2D_single_time_test.h5"
with h5py.File(h5_file_path, "r") as h5f:
    images = h5f["img"][:]

print(images.shape)

volume = np.array(images)


# In[8]:


plt.imshow(images[140])


# In[9]:


# Get the dimensions (n, height, width) where:
# n = number of images along the z-axis
# height = number of rows (Y dimension)
# width = number of columns (X dimension)
depth, height, width = volume.shape

# Create a PyVista UniformGrid (3D volume)
grid = pv.UniformGrid()
grid.dimensions = [width, height, depth]  # Make sure this is (X, Y, Z) order
grid.spacing = [1, 1, 1]  # You can adjust the spacing as needed

# Flatten the 3D volume and assign it to cell data (or point data)
grid.point_data["values"] = reduced_volume.flatten()  

# Add the scalar values to the grid

# Save the volume as a .vti file
grid.save("output_reduced_volume.vti")
print("Volume saved as .vti!")


# # 3D Model with OpenCV, Converting to numpy stack and then vtk into Paraview 
# 
# Single time visualisation of stack of 2D images along z-axis in Paraview. 

# In[1]:


import cv2
from matplotlib import pyplot as plt 
import numpy as np
import zipfile
import os
from natsort import os_sorted
import pyvista as pv


# In[2]:


data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_video/imagestacks/20181113_HCC_d0-2_t42c1_ORG/"

img_stack = []

# Open the file where single .tif 2D images are. (extracted with ImageJ)
items = [file for file in os_sorted(os.listdir(data_dir)) if file.lower().endswith(".tif")]
for file in items:
    img = cv2.imread(data_dir + file, 0) 
    # Normalisation with cv2, so that the minimal intensity 0 and maximal intensity 255. 
    img_scaled = cv2.normalize(img, dst=True, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_stack.append(np.array(img_scaled))
    print("ok"+ file)

# Volume as a 3D numpy array at the end. 
volume = np.stack(np.array(img_stack), axis=0)


# In[3]:


np.shape(volume)


# In[4]:


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
grid.save("output_example.vti")
print("Volume saved as output_example.vti!")


# # 2D slice segmentation 

# In[5]:


import numpy as np
import cv2 
import matplotlib.pyplot as plt


# In[27]:


data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_video/imagestacks/20181113_HCC_d0-2_t42c1_ORG/"
img_name = "20181113_HCC_d0-2_t42c1_ORG0391.tif"

img = cv2.imread(data_dir + file)
# assert img is not None, "file could not be read, check with os.path.exists()"
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img,255,0,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
 
# # sure background area
# sure_bg = cv.dilate(opening,kernel,iterations=3)
 
# # Finding sure foreground area
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
 
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg,sure_fg)

# masked_img_stack = []

# # Open the file where single .tif 2D images are. (extracted with ImageJ)
# items = [file for file in os_sorted(os.listdir(data_dir)) if file.lower().endswith(".tif")]
# for file in items:
#     img = cv2.imread(data_dir + file, 0) 
#     # Normalisation with cv2, so that the minimal intensity 0 and maximal intensity 255. 
#     img_scaled = cv2.normalize(img, dst=True, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     img_stack.append(np.array(img_scaled))
#     print("ok"+ file)

# # Volume as a 3D numpy array at the end. 
# volume = np.stack(np.array(img_stack), axis=0)


# In[18]:


test_img = 


# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Example 2D numpy array (image)
image = cv2.imread(data_dir + file)

# Convert to a 3-channel (RGB) image by stacking the single channel
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Apply a binary threshold to get a binary image
_, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

# Noise removal (optional step)
binary_image = cv2.dilate(binary_image, None, iterations=2)
binary_image = cv2.erode(binary_image, None, iterations=1)

# Marker labeling (must label foreground and background)
dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
_, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Convert markers to 32-bit integers
markers = np.int32(markers)

# Apply watershed algorithm
cv2.watershed(image_rgb, markers)

# Mark boundaries in red color
image_rgb[markers == -1] = [255, 0, 0]

# Plot the segmented image
plt.imshow(image_rgb)
plt.show()


# In[2]:


# K-means clustering here is not really bad. 

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example 2D numpy array (image)
image = np.array([[0, 50, 200], [100, 150, 200], [0, 100, 255]])


# Reshape image into a 1D array of pixels
pixels = image.reshape(-1, 1)

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)

# Reshape the labels back into the original image shape
segmented_image = kmeans.labels_.reshape(image.shape)

# Plot the segmented image
plt.imshow(segmented_image, cmap='viridis')
plt.show()


# In[8]:


pip install scikit-learn


# In[ ]:




