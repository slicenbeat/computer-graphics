import math
import numpy as np
from PIL import Image
from models.classes import ImageClass, OBJ3DModel, Point, Color
# Задаем параметры изображения 
# высота, ширина, масштаб, центр, углы поворота
height = 1000
weight = 1000
a_x = 1
a_y = 1
u_0 = height/2
v_0 = weight/2
angles = [0, 0, 0]
# вектор-столбец t, матрица внутренних параметров
t = np.array([[0.005], [-0.045], [15]])
intrinsic = np.array([[a_x, 0, u_0], [0, a_y, v_0], [0, 0, 1]])


obj = OBJ3DModel('objects/fox.obj')
arr = np.zeros((height, weight, 3), dtype=np.uint8)
image = ImageClass(arr)
image.initializeMatrixAsWhiteImage()

# Осуществляем сдвиг координат для корректного отображения на изображении
shiftedPoints = image.shiftPoints(obj.getPoints())

# Получаем координаты экранные (u, v, 1)
projectiveTransformationPoints = image.getProjectiveTransformationPoints(shiftedPoints, t, intrinsic) 

# Поворачиваем координаты лисы
modelRotationPoints = image.getModelRotationPoints(projectiveTransformationPoints, angles) 

image.drawColorPolygons(modelRotationPoints, obj.getPolygons())
image.saveImageToFilePNG('fox.png')
