import math
import numpy as np
from PIL import Image

from models.classes import ImageClass, OBJ3DModel, Point, Color


def drawTriangle(image):
    points = [Point(423, 100, 0), Point(345, 45, 0), Point(200, 600, 0)]
    image.drawTriangle(points, Color(255, 255, 0, 0))
    image.saveImageToFilePNG('ready.png')
    image.initializeMatrixAsWhiteImage()


arr = np.zeros((1000, 1000, 3), dtype=np.uint8)
image = ImageClass(arr)
image.initializeMatrixAsWhiteImage()
# drawTriangle(image)

obj = OBJ3DModel('objects/fox.obj')
image.transformPoints(obj.getPoints())
image.drawColorPolygons(obj.getPoints(), obj.getPolygons())
image.saveImageToFilePNG('fox.png')
