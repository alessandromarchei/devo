from devo.data_readers.utils import h5_to_voxels
import numpy as np




scene_root = "/scratch/amarchei/TartanEvent/"
scene = "ocean/Easy/P000/"

data = h5_to_voxels(scene_root + scene, nbins=5, H=640, W=480, use_event_stack=False)
print("Data shape: ", data.shape)
print("Data type: ", data.dtype)
print("Data: ", data)