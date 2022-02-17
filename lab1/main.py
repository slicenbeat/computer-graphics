import math
import sys
sys.path.insert(0, 'models/')
import numpy as np
from PIL import Image

from models.imageclass import ImageClass
from models.obj3dmodel import OBJ3DModel
from models.point import Point

arr = np.zeros((3000, 1000, 3), dtype=np.uint8)
image = ImageClass(arr)
image.initializeMatrixAsWhiteImage()
obj = OBJ3DModel('objects/fox.obj')
image.transformPoints(obj.getPoints())
image.drawPolygons(obj.getPoints(), obj.getPolygons())
image.saveImageToFilePNG('fox.png')

image.initializeMatrixAsWhiteImage()
for i in range(0, 13):
    image.drawLineFirstMethod(100, 100, int(100 + 95 * np.cos(2 * np.pi * i / 13)),
                            int(100 + 95 * np.sin(2 * np.pi * i / 13)))
image.showImage()

image.initializeMatrixAsWhiteImage()
for i in range(0, 13):
    image.drawLineSecondMethod(100, 100, int(100 + 95 * np.cos(2 * np.pi * i / 13)),
                            int(100 + 95 * np.sin(2 * np.pi * i / 13)))
image.showImage()

image.initializeMatrixAsWhiteImage()
for i in range(0, 13):
    image.drawLineThirdMethod(100, 100, int(100 + 95 * np.cos(2 * np.pi * i / 13)),
                            int(100 + 95 * np.sin(2 * np.pi * i / 13)))
image.showImage()

image.initializeMatrixAsWhiteImage()
for i in range(0, 13):
    image.drawLineFourthMethod(100, 100, int(100 + 95 * np.cos(2 * np.pi * i / 13)),
                            int(100 + 95 * np.sin(2 * np.pi * i / 13)))
image.showImage()
