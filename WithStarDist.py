from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import StarDist3D

np.random.seed(6)
lbl_cmap = random_label_cmap()


tif_file = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_video/20181113_HCC_d0-2_t42c1_ORG(crop).tif"

# Showing the input data
img = imread(tif_file)
print(np.shape(img))

z = img.shape[0] // 2
y = img.shape[1] // 2

plt.figure()
plt.imshow(img[z],cmap='gray')
plt.title("XY slice")
plt.figure()
plt.imshow(img[:,y],cmap='gray')
plt.title("XZ slice")
plt.show()

# Loading pretrained model
model = StarDist3D.from_pretrained('3D_demo')

