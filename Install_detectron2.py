#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import h5py
import detectron2
import cv2
from matplotlib import pyplot as plt 

data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_video/"
file_name = "20181113_HCC_d0-2_t42c1_ORG.tif"


# In[ ]:


tif_path = data_dir + file_name

with tiff.TiffFile(tif_path) as tif:
    image_stack = tif.asarray()

print("image shape:", image_stack.shape)

# Store the image data into .h5 files.
h5_file_name = "organoid_2D_single_time_test.h5"
with h5py.File(h5_file_name, "w") as h5f:
    h5f.create_dataset("img", data = np.array(image_stack))

# open h5 files.
h5_file_path = "organoid_2D_single_time_test.h5"
with h5py.File(h5_file_path, "r") as h5f:
    images = h5f["img"][:]


# In[ ]:


im = plt.imshow(images[140], cmap="gray") 
plt.savefig("organoid_2D_single_time_test.jpg", format="jpg")


# In[2]:


get_ipython().system('pip install numpy==1.24.4')


# In[1]:


import cv2
from detectron2.detectron2.utils.visualizer import Visualizer, ColorMode
from matplotlib import pyplot as plt 

####################################################################

# the format of the predictions is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
outputs = predictor(im)

preds   = outputs["instances"].to("cpu"); 
v       = Visualizer(im[:, :, ::-1], scale=0.5 )  
out     = v.draw_instance_predictions(preds)
pltimg  = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(20,10))
ax.imshow(pltimg)


# In[8]:


pip install pycocotools


# In[ ]:




