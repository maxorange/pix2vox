import numpy as np
import binvox
import os
import cv2

def read_binvox(fname):
    with open(fname, 'rb') as f:
        model = binvox.read_as_3d_array(f)
        data = vox2tanh(model.data.astype(np.float32))
        return data

def save_binvox(data, fname):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(fname, 'wb') as f:
        model.write(f)

def read_image(filename, color=True):
    img = cv2.imread(filename, color)
    return rgb2tanh(img.astype(np.float32))

def tanh2rgb(data):
    return (data + 1) * 127.5

def rgb2tanh(data):
    return (data - 127.5) / 127.5

def vox2tanh(data):
    return (data - 0.5) / 0.5

def get_name(fname):
    head, tail = os.path.split(fname)
    name, ext = os.path.splitext(tail)
    return name
