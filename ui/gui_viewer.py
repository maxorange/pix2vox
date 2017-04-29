import numpy as np
import vtk
import time
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from multiprocessing import Pool
from multiprocessing import Process

class GUIViewer(QVTKRenderWindowInteractor):

    def __init__(self, parent, opt_engine):
        QVTKRenderWindowInteractor.__init__(self, parent)
        self.opt_engine = opt_engine

        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.GetRenderWindow().GetInteractor()

        self.create_actor()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0, 0, 0)

    def create_voxel(self):
        numberOfVertices = 8

        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(1, 0, 0)
        points.InsertNextPoint(0, 1, 0)
        points.InsertNextPoint(1, 1, 0)
        points.InsertNextPoint(0, 0, 1)
        points.InsertNextPoint(1, 0, 1)
        points.InsertNextPoint(0, 1, 1)
        points.InsertNextPoint(1, 1, 1)

        voxel = vtk.vtkVoxel()
        for i in range(0, numberOfVertices):
            voxel.GetPointIds().SetId(i, i)

        ugrid = vtk.vtkUnstructuredGrid()
        ugrid.SetPoints(points)
        ugrid.InsertNextCell(voxel.GetCellType(), voxel.GetPointIds())

        gfilter = vtk.vtkGeometryFilter()
        gfilter.SetInput(ugrid)
        gfilter.Update()
        return gfilter

    def create_actor(self):
        self.points = vtk.vtkPoints()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetName("colors")
        self.colors.SetNumberOfComponents(4)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(self.points)
        polydata.GetPointData().SetScalars(self.colors)

        # create cell
        voxel = self.create_voxel()

        self.glyph3D = vtk.vtkGlyph3D()
        self.glyph3D.SetColorModeToColorByScalar()
        self.glyph3D.SetSource(voxel.GetOutput())
        self.glyph3D.SetInput(polydata)
        self.glyph3D.ScalingOff()
        self.glyph3D.Update()

        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(self.glyph3D.GetOutput())

        # actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetAmbient(0.15)

    def update_actor(self):
        data = self.opt_engine.get_3d_model()
        indices = np.where(data[:, :, :, 0] > 0)
        n_voxels = indices[0].shape[0]

        self.points.Reset()
        self.colors.Reset()

        for i in xrange(n_voxels):
            x = indices[0][i]
            y = indices[1][i]
            z = indices[2][i]
            r = data[x, y, z, 3]
            g = data[x, y, z, 2]
            b = data[x, y, z, 1]
            self.colors.InsertTuple4(i, r, g, b, 128)
            self.points.InsertNextPoint(x, y, z)

        self.glyph3D.Modified()
        self.renderer.ResetCamera()
        self.update()
        QApplication.processEvents()
        # print "update model"
