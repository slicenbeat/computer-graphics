import math
import numpy as np
from PIL import Image
from models.classes import ImageClass, OBJ3DModel, Point, Color
# Задаем параметры изображения 
# высота, ширина, масштаб, центр, углы поворота
height = 4000
weight = 1000
a_x = 10
a_y = 10
u_0 = 10
v_0 = 0
angles = [0, 0, 0]
# вектор-столбец t, матрица внутренних параметров
t = np.array([[0.005], [-0.045], [15]])
intrinsic = np.array([[a_x, 0, u_0], [0, a_y, v_0], [0, 0, 1]])
# intrinsic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# scale = 1 / (maxs-mins) * [width , height] ax ay
obj = OBJ3DModel('objects/fox.obj')
arr = np.zeros((height, weight, 3), dtype=np.uint8)
image = ImageClass(arr)
image.initializeMatrixAsWhiteImage()

# Осуществляем сдвиг координат для корректного отображения на изображении
# Поворачиваем координаты лисы
modelRotationPoints = image.getModelRotationTransformationPoints(obj.getPoints(), t,intrinsic,angles)
image.drawColorPolygons(image.shiftPoints(modelRotationPoints), obj.getPolygons())
image.saveImageToFilePNG('fox.png')
