import math
import numpy as np
from PIL import Image

from models.classes import ImageClass, OBJ3DModel, Point, Color

arr = np.zeros((800, 800, 3), dtype=np.uint8)
image = ImageClass(arr)
image.initializeMatrixAsWhiteImage()
# points = [Point(423, 100, 0), Point(345, 45, 0), Point(200, 600, 0)]
# image.drawTriangle(points, Color(0, 0, 0, 0))
# image.saveImageToFilePNG('./lab2/ready.png')
# image.initializeMatrixAsWhiteImage()
obj = OBJ3DModel('./lab2/objects/fox.obj')
image.transformPoints(obj.getPoints())
image.drawColorPolygons(obj.getPoints(), obj.getPolygons())
image.saveImageToFilePNG('fox.png')
