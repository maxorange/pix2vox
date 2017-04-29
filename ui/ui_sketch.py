import numpy as np
import cv2

class UISketch:
    def __init__(self, img_size, scale, nc=1):
        self.img_size = img_size
        self.scale = scale
        self.nc = nc
        self.img = np.zeros((img_size, img_size, self.nc), np.uint8)
        self.width = int(scale)

    def update(self, points, color=1):
        num_pnts = len(points)
        w = int(max(1, self.width / self.scale))
        for i in range(0, num_pnts - 1):
            pnt1 = (int(points[i].x()/self.scale), int(points[i].y()/self.scale))
            pnt2 = (int(points[i + 1].x()/self.scale), int(points[i + 1].y()/self.scale))
            cv2.line(self.img, pnt1, pnt2, color, w)

    def update_width(self, brushWidth):
        self.width = brushWidth

    def get_constraints(self):
        return self.img

    def reset(self):
        self.img = np.zeros((self.img_size, self.img_size, self.nc), np.uint8)

    def save(self):
        cv2.imwrite("out/edge.png", self.img*255)
