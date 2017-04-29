import copy
from PyQt4.QtCore import *
from PyQt4.QtGui import *

class UIRecorder(object):

    def __init__(self):
        self.strokes = []
        self.colors = []
        self.widths = []
        self.types = []

    def save(self, stroke, color, width, type):
        self.strokes.append(copy.deepcopy(stroke))
        self.colors.append(color)
        self.widths.append(width)
        self.types.append(type)

    def draw(self, painter):
        for points, color, width, type in zip(self.strokes, self.colors, self.widths, self.types):
            if type is 'edge':
                painter.setPen(QPen(Qt.gray, width, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
            elif type is 'eraser':
                painter.setPen(QPen(Qt.white, width, cap=Qt.RoundCap, join=Qt.RoundJoin))
            else:
                painter.setPen(QPen(color, width, cap=Qt.RoundCap, join=Qt.RoundJoin))
            n_pnts = len(points)
            for i in xrange(0, n_pnts-5, 5):
                painter.drawLine(points[i], points[i+5])

    def reset(self):
        del self.strokes[:]
        del self.colors[:]
        del self.widths[:]
