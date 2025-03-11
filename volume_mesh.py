from stack3DVTK import generate_volume
import numpy as np
import pymeshlab
from skimage import measure

data_dir = "/Users/rzhoufias.uni-frankfurt.de/Documents/PhD_Franziska/Organoid/organoid_3Dtrack/data/HCC_42_65_tif_image/20181113_HCC_d0-2_t42c1_ORG(crop)/"
volume = generate_volume(data_dir)

# Extract the surface mesh using Marching Cubes
verts, faces, _, _ = measure.marching_cubes(volume.astype(float), level=1)

# Convert vertices and faces into PyMeshLab format
ms = pymeshlab.MeshSet()
mesh = pymeshlab.Mesh(verts, faces)
ms.add_mesh(mesh)

# Save the mesh to an .obj file
ms.save_current_mesh("mesh.obj")
