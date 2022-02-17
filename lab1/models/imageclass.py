import math

import numpy as np
from PIL import Image
from random import randint
from typing import List

from point import Point


class ImageClass:

    def __init__(self, matrix):
        try:
            self.H = len(matrix)
            self.W = len(matrix[0])
            self.matrixImage = matrix
            for i in range(0, self.H):
                if len(self.matrixImage[i]) != self.W:
                    raise Exception("Не соответствующие размеры матрицы.")
            for i in range(0, self.H):
                for j in range(0, self.W):
                    if len(self.matrixImage[i][j]) != 3:
                        raise Exception("Не соответствующие размеры матрицы.")
        except Exception as e:
            print("Ошибка при инициализации матрицы. Код ошибки:")
            print(e.args)

    def saveArrayImageToFileNPY(self, fileName):
        try:
            np.save(fileName, self.matrixImage)
        except Exception as e:
            print("Ошибка при сохранении матрицы в файл. Код ошибки:")
            print(e.args)

    def loadArrayImageFromFileNPY(self, fileName):
        try:
            self.matrixImage = np.load(fileName)
        except Exception as e:
            print("Ошибка при загрузке матрицы из файла. Код ошибки:")
            print(e.args)

    def showImage(self):
        try:
            print(self.matrixImage)
            img = Image.fromarray(self.matrixImage)
            img.show()
        except Exception as e:
            print("Ошибка при отрисовке изображения. Код ошибки:")
            print(e.args)

    def saveImageToFilePNG(self, fileName):
        try:
            img = Image.fromarray(self.matrixImage)
            img.save(fileName)
        except Exception as e:
            print("Ошибка при сохранении изображения. Код ошибки:")
            print(e.args)

    def initializeMatrixAsWhiteImage(self):
        for i in range(self.H):
            for j in range(self.W):
                for k in range(3):
                    self.matrixImage[i][j][k] = 255

    def initializeMatrixAsBlackImage(self):
        self.matrixImage = np.zeros((self.H, self.W, 3), dtype=np.uint8)

    def initializeMatrixAsRedImage(self):
        for i in range(self.H):
            for j in range(self.W):
                self.matrixImage[i][j][0] = 255
                self.matrixImage[i][j][1] = 0
                self.matrixImage[i][j][2] = 0

    def initializeMatrixAsGradientImage(self):
        for i in range(self.H):
            for j in range(self.W):
                self.matrixImage[i][j][0] = (i + j) % 256
                self.matrixImage[i][j][1] = 0
                self.matrixImage[i][j][2] = 0

    def initializeMatrixAsRandColorImage(self):
        for i in range(self.H):
            for j in range(self.W):
                for k in range(3):
                    self.matrixImage[i][j][k] = randint(0, 255)

    def setPixel(self, x, y, color1, color2, color3):
        try:
            self.matrixImage[x][y] = [color1, color2, color3]
        except Exception as e:
            print("Ошибка при отрисовке пикселя. Код ошибки:")
            print(e.args)

    def drawLineFirstMethod(self, x0, y0, x1, y1):
        try:
            for t in np.arange(0, 1, 0.01):
                x = int(x0 * (1.0 - t) + x1 * t)
                y = int(y0 * (1.0 - t) + y1 * t)
                self.setPixel(x, y, 0, 0, 0)
        except Exception as e:
            print("Ошибка при отрисовке линии. Код ошибки:")
            print(e.args)

    def drawLineSecondMethod(self, x0, y0, x1, y1):
        try:
            for x in np.arange(x0, x1 + 1):
                t = (x - x0) / (x1 - x0)
                y = int(y0 * (1 - t) + y1 * t)
                self.setPixel(x, y, 0, 0, 0)
        except Exception as e:
            print("Ошибка при отрисовке линии. Код ошибки:")
            print(e.args)

    def drawLineThirdMethod(self, x0, y0, x1, y1):
        try:
            steep = False
            if np.abs(x0 - x1) < np.abs(y0 - y1):
                x0, y0 = y0, x0
                x1, y1 = y1, x1
                steep = True
            if x0 > x1:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            for x in np.arange(x0, x1 + 1):
                t = (x - x0) / (x1 - x0)
                y = int(y0 * (1 - t) + y1 * t)
                if steep:
                    self.setPixel(y, x, 0, 0, 0)
                else:
                    self.setPixel(x, y, 0, 0, 0)
        except Exception as e:
            print("Ошибка при отрисовке линии. Код ошибки:")
            print(e.args)

    def drawLineFourthMethod(self, x0, y0, x1, y1):
        print(x0, y0, x1, y1)
        try:
            steep = False
            if np.abs(x0 - x1) < np.abs(y0 - y1):
                x0, y0 = y0, x0
                x1, y1 = y1, x1
                steep = True
            if x0 > x1:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            dx = x1 - x0
            dy = y1 - y0
            derror = np.abs(dy / dx)
            error = 0
            y = y0
            for x in range(x0, x1 + 1):
                if steep:
                    self.setPixel(y, x, 0, 0, 0)
                else:
                    self.setPixel(x, y, 0, 0, 0)
                error += derror
                if error > 0.5:
                    if y1 > y0:
                        y += 1
                    else:
                        y -= 1
                    error -= 1
        except Exception as e:
            print("Ошибка при отрисовке линии. Код ошибки:")
            print(e.args)

    def drawPoints(self, points: List[Point]):
        for i in range(len(points)):
            self.setPixel(points[i].getY(), points[i].getZ(), 0, 0, 0)

    def drawPolygons(self, points: List[Point], polygons):
        try:
            for i in range(len(polygons)):
                if len(polygons[i]) != 3:
                    raise Exception("Invalid polygon")
        except Exception as e:
            print("Ошибка при отрисовке полигона. Код ошибки:")
            print(e.args)
        for i in range(len(polygons)):
            self.drawLineFourthMethod(points[polygons[i][0] - 1].getY(),
                                      points[polygons[i][0] - 1].getZ(),
                                      points[polygons[i][1] - 1].getY(),
                                      points[polygons[i][1] - 1].getZ())
            self.drawLineFourthMethod(points[polygons[i][0] - 1].getY(),
                                      points[polygons[i][0] - 1].getZ(),
                                      points[polygons[i][2] - 1].getY(),
                                      points[polygons[i][2] - 1].getZ())
            self.drawLineFourthMethod(points[polygons[i][1] - 1].getY(),
                                      points[polygons[i][1] - 1].getZ(),
                                      points[polygons[i][2] - 1].getY(),
                                      points[polygons[i][2] - 1].getZ())

    def transformPoints(self, points: List[Point]):
        minValueY = 0
        minValueZ = 0
        for i in range(len(points)):
            if points[i].getY() < minValueY:
                minValueY = points[i].getY()
            if points[i].getZ() < minValueZ:
                minValueZ = points[i].getZ()
        for i in range(len(points)):
            points[i].y -= minValueY
            points[i].z -= minValueZ

        for i in range(len(points)):
            points[i].y = int(points[i].y * 5)
            points[i].z = int(points[i].z * 5)