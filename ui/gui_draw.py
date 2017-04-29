import numpy as np
import vtk
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui_color import UIColor
from ui_sketch import UISketch
from ui_recorder import UIRecorder

class GUIDraw(QWidget):

    def __init__(self, opt_engine, win_size):
        QWidget.__init__(self)
        self.opt_engine = opt_engine
        self.win_size = win_size
        self.img_size = opt_engine.model.npx
        self.scale = win_size / float(self.img_size)
        self.points = []
        self.pos = None
        self.type = 'edge'
        self.brushWidth = int(self.scale)
        self.color = QColor().fromHsl(0, 255, 128)
        self.uiRecorder = UIRecorder()
        self.uiColor = UIColor(self.img_size, self.scale)
        self.uiSketch = UISketch(self.img_size, self.scale)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), Qt.white)
        painter.setRenderHint(QPainter.Antialiasing)

        # drawn paths
        self.uiRecorder.draw(painter)

        # new path
        if self.type is 'edge':
            painter.setPen(QPen(Qt.gray, self.brushWidth, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
        elif self.type is 'eraser':
            painter.setPen(QPen(Qt.white, self.brushWidth, cap=Qt.RoundCap, join=Qt.RoundJoin))
        else:
            painter.setPen(QPen(self.color, self.brushWidth, cap=Qt.RoundCap, join=Qt.RoundJoin))

        n_pnts = len(self.points)
        for i in xrange(0, n_pnts-5, 5):
            painter.drawLine(self.points[i], self.points[i+5])

        # cursor
        if self.pos is not None:
            fillWidth = self.brushWidth - 1
            pnt = QPoint(self.pos.x(), self.pos.y())

            # draw outline
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(Qt.black)
            painter.drawEllipse(pnt, self.brushWidth, self.brushWidth)

            # draw fill color
            if self.type is 'edge':
                painter.setPen(QPen(Qt.gray, 1))
                painter.setBrush(Qt.gray)
            elif self.type is 'eraser':
                painter.setPen(QPen(Qt.white, 1))
                painter.setBrush(Qt.white)
            else:
                painter.setPen(QPen(self.color, 1))
                painter.setBrush(self.color)
            painter.drawEllipse(pnt, fillWidth, fillWidth)

        painter.end()

    def wheelEvent(self, event):
        step = np.sign(event.delta())
        self.brushWidth = min(20, max(2, self.brushWidth + step))
        self.uiColor.update_width(self.brushWidth)
        self.uiSketch.update_width(self.brushWidth)
        self.update()

    def mousePressEvent(self, event):
        self.pos = self.round_point(event.pos())
        if event.button() == Qt.LeftButton:
            self.update_opt_engine()
            self.update_ui()
            self.update()
        elif event.button() == Qt.RightButton:
            color = QColorDialog.getColor(parent=self)
            self.color = color
            self.update()

    def mouseMoveEvent(self, event):
        self.pos = self.round_point(event.pos())
        self.points.append(self.pos)
        self.update_opt_engine()
        self.update_ui()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.update()
            self.uiRecorder.save(self.points, self.color.toHsl(), self.brushWidth, self.type)
            del self.points[:]

    def round_point(self, pnt):
        x = int(np.round(pnt.x()))
        y = int(np.round(pnt.y()))
        return QPoint(x, y)

    def update_opt_engine(self):
        color = self.uiColor.get_constraints()
        edge = self.uiSketch.get_constraints()
        self.opt_engine.set_constraints([color, edge])
        self.opt_engine.update()

    def update_ui(self):
        if self.type is 'color':
            self.uiColor.update(self.points, self.color)
        if self.type is 'edge':
            self.uiSketch.update(self.points)
        if self.type is 'eraser':
            self.uiColor.update(self.points, QColor(Qt.white))
            self.uiSketch.update(self.points, 0)

    def use_color(self):
        self.type = 'color'

    def use_edge(self):
        self.type = 'edge'

    def use_eraser(self):
        self.type = 'eraser'
