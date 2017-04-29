import numpy as np
import time
import util
from scipy import ndimage as ndi
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class ConstrainedOpt(QThread):

    def __init__(self, model):
        QThread.__init__(self)
        self.model = model
        self.max_iters = 1
        self.iter_count = 0
        self.constraints = None
        self.constraints_t = None
        self.label = np.zeros([1, model.n_cls], dtype=np.float32)
        self.sample_z()

    def run(self):
        while True:
            if self.constraints is not None and self.iter_count < self.max_iters:
                self.update_voxel_model()
                self.iter_count += 1
            self.msleep(100)

    def update(self):
        self.iter_count = 0

    def sample_z(self):
        self.z = np.random.uniform(-1, 1, [1, self.model.nz]).astype(np.float32)
        self.update()

    def preprocess_constraints(self):
        color_old, edge_old = self.constraints
        color = util.rgb2tanh(color_old)[::-1, ::-1]
        color = np.expand_dims(color, 0).astype(np.float32)
        edge = np.expand_dims(edge_old, 0).astype(np.float32)
        self.constraints_t = [color, edge, self.z, self.label]

    def update_voxel_model(self):
        self.preprocess_constraints()
        self.model.update(*self.constraints_t)
        self.emit(SIGNAL('update_voxels'))

    def dilation(self):
        self.model.current_shapes[:, :, :, 0] = ndi.binary_dilation(self.model.current_shapes[:, :, :, 0])
        self.emit(SIGNAL('update_voxels'))

    def erosion(self):
        self.model.current_shapes[:, :, :, 0] = ndi.binary_erosion(self.model.current_shapes[:, :, :, 0])
        self.emit(SIGNAL('update_voxels'))

    def set_constraints(self, constraints):
        self.constraints = constraints

    def set_label(self, value, index):
        self.label[:, index] = value
        self.update_voxel_model()

    def get_3d_model(self):
        return self.model.current_shapes
