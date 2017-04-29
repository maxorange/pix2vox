import numpy as np
import cv2

class UIColor:

    def __init__(self, npx, scale):
        self.npx = npx
        self.scale = scale
        self.img = np.ones([npx, npx, 3], np.uint8)*255
        self.width = int(scale)

    def update(self, points, color):
        n_pnts = len(points)
        w = int(max(1, self.width / self.scale))
        c = (color.blue(), color.green(), color.red())
        for i in range(0, n_pnts - 1):
            pnt1 = (int(points[i].x() / self.scale), int(points[i].y() / self.scale))
            pnt2 = (int(points[i + 1].x() / self.scale), int(points[i + 1].y() / self.scale))
            cv2.line(self.img, pnt1, pnt2, c, w)

    def get_constraints(self):
        return self.img

    def update_width(self, brushWidth):
        self.width = brushWidth

    def reset(self):
        self.img = np.zeros([self.npx, self.npx, 3], np.uint8)

    def save(self):
        cv2.imwrite("out/color.png", self.img)
