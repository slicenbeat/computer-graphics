import math

import numpy as np
from PIL import Image
from random import randint
from typing import List

from pexpect import ExceptionPexpect


class Color:
    def __init__(self, r=0, g=0, b=0, a=0):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def getR(self):
        return self.r

    def getG(self):
        return self.g

    def getB(self):
        return self.b

    def getA(self):
        return self.a


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z


numberOfNormals = []
normals = []


class OBJ3DModel:
    def __init__(self, path: str):
        f = open(path, mode='r')
        self.points = []
        self.polygons = []
        for line in f:
            if line[:3] == 'vn ':
                normals.append([float(i)
                                for i in line[3:].strip('\n').split(' ')])
            if line[:2] == 'v ':
                line_of_numbers = [float(i)
                                   for i in line[2:].strip('\n').split(' ')]
                self.points.append(
                    # Point(900 - line_of_numbers[1] * 10, line_of_numbers[0]*10 + 500, line_of_numbers[2] * 10))
                    Point(line_of_numbers[0], line_of_numbers[1], line_of_numbers[2]))
            elif line[:2] == 'f ':
                line_of_numbers = line[2:].strip('\n').split(' ')
                temp_polygons = []
                numberOfNormals.append(line_of_numbers[0].split('/')[2])
                for i in range(3):
                    line_to_numbers = line_of_numbers[i].split('/')
                    temp_polygons.append(int(line_to_numbers[0]))
                self.polygons.append(temp_polygons)

    def getPoints(self):
        return self.points

    def getPolygons(self):
        return self.polygons


class ImageClass:
    def __init__(self, matrix):
        try:
            self.H = len(matrix)
            self.W = len(matrix[0])
            self.matrixImage = matrix
            for i in range(0, self.H):
                if len(self.matrixImage[i]) != self.W:
                    raise Exception("???? ?????????????????????????????? ?????????????? ??????????????.")
            for i in range(0, self.H):
                for j in range(0, self.W):
                    if len(self.matrixImage[i][j]) != 3:
                        raise Exception("???? ?????????????????????????????? ?????????????? ??????????????.")
            self.matrixZ = np.zeros((self.H, self.W))
        except Exception as e:
            print("???????????? ?????? ?????????????????????????? ??????????????. ?????? ????????????:")
            print(e.args)

    def saveArrayImageToFileNPY(self, fileName):
        try:
            np.save(fileName, self.matrixImage)
        except Exception as e:
            print("???????????? ?????? ???????????????????? ?????????????? ?? ????????. ?????? ????????????:")
            print(e.args)

    def loadArrayImageFromFileNPY(self, fileName):
        try:
            self.matrixImage = np.load(fileName)
        except Exception as e:
            print("???????????? ?????? ???????????????? ?????????????? ???? ??????????. ?????? ????????????:")
            print(e.args)

    def showImage(self):
        try:
            print(self.matrixImage)
            img = Image.fromarray(self.matrixImage)
            img.show()
        except Exception as e:
            print("???????????? ?????? ?????????????????? ??????????????????????. ?????? ????????????:")
            print(e.args)

    def saveImageToFilePNG(self, fileName):
        try:
            img = Image.fromarray(self.matrixImage)
            img.save(fileName)
        except Exception as e:
            print("???????????? ?????? ???????????????????? ??????????????????????. ?????? ????????????:")
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
            print("???????????? ?????? ?????????????????? ??????????????. ?????? ????????????:")
            print(e.args)

    def drawLineFirstMethod(self, x0, y0, x1, y1):
        try:
            for t in np.arange(0, 1, 0.01):
                x = int(x0 * (1.0 - t) + x1 * t)
                y = int(y0 * (1.0 - t) + y1 * t)
                self.setPixel(x, y, 0, 0, 0)
        except Exception as e:
            print("???????????? ?????? ?????????????????? ??????????. ?????? ????????????:")
            print(e.args)

    def drawLineSecondMethod(self, x0, y0, x1, y1):
        try:
            for x in np.arange(x0, x1 + 1):
                t = (x - x0) / (x1 - x0)
                y = int(y0 * (1 - t) + y1 * t)
                self.setPixel(x, y, 0, 0, 0)
        except Exception as e:
            print("???????????? ?????? ?????????????????? ??????????. ?????? ????????????:")
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
            print("???????????? ?????? ?????????????????? ??????????. ?????? ????????????:")
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
            print("???????????? ?????? ?????????????????? ??????????. ?????? ????????????:")
            print(e.args)

    def drawPoints(self, points: List[Point]):
        for i in range(len(points)):
            self.setPixel(points[i].getY(), points[i].getZ(), 0, 0, 0)

    def drawPolygons(self, points: List[Point], polygons):
        try:
            for i in range(len(polygons)):
                if len(polygons[i]) != 3:
                    raise Exception("Invalid polygon")

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
        except Exception as e:
            print("???????????? ?????? ?????????????????? ????????????????. ?????? ????????????:")
            print(e.args)

    def drawColorPolygons(self, points: List[Point], polygons):
        try:
            for i in range(len(polygons)):
                if len(polygons[i]) != 3:
                    raise Exception("Invalid polygon")

            for i in range(len(polygons)):
                points_for_triangle = [Point(points[polygons[i][0] - 1].getX(), points[polygons[i][0] - 1].getY(),
                                             points[polygons[i][0] - 1].getZ()),
                                       Point(
                                           points[polygons[i][1] - 1].getX(), points[polygons[i][1] - 1].getY(),
                                           points[polygons[i][1] - 1].getZ()),
                                       Point(points[polygons[i][2] - 1].getX(), points[polygons[i][2] - 1].getY(),
                                             points[polygons[i][2] - 1].getZ())]
                self.drawTriangle(points_for_triangle, polygons[i][0], polygons[i][1], polygons[i][2])
        except Exception as e:
            print("???????????? ?????? ?????????????????? ????????????????. ?????? ????????????:")
            print(e.args)

    def calculateBarycentricCoordinates(self, x, y, x0, y0, x1, y1, x2, y2):

        lambda0 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)) / \
            ((x1 - x2)*(y0 - y2) - (y1 - y2)*(x0 - x2))

        lambda1 = ((x2 - x0)*(y - y0) - (y2 - y0)*(x - x0)) / \
            ((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0))

        lambda2 = ((x0 - x1)*(y - y1) - (y0 - y1)*(x - x1)) / \
            ((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))

        lambdas = [lambda0, lambda1, lambda2]
        return lambdas

    def shiftPoints(self, points: List[Point]):
        changePoints = points
        minValueY = 0
        minValueZ = 0
        minValueX = 0
        for i in range(len(points)):
            if changePoints[i].getY() < minValueY:
                minValueY = changePoints[i].getY()
            if changePoints[i].getZ() < minValueZ:
                minValueZ = changePoints[i].getZ()
            if changePoints[i].getX() < minValueX:
                minValueX = changePoints[i].getX()
        for i in range(len(points)):
            changePoints[i].y -= minValueY
            changePoints[i].z -= minValueZ
            changePoints[i].x -= minValueX

        for i in range(len(points)):
            changePoints[i].y = int(changePoints[i].y)
            changePoints[i].z = int(changePoints[i].z)
            changePoints[i].x = int(changePoints[i].x)
        return changePoints

    def drawTriangle(self, points: List[Point], numberPointOne, numberPointTwo, numberPointThree):
        x_min = min(points[0].getX(), points[1].getX(), points[2].getX())
        y_min = min(points[0].getY(), points[1].getY(), points[2].getY())
        x_max = max(points[0].getX(), points[1].getX(), points[2].getX())
        y_max = max(points[0].getY(), points[1].getY(), points[2].getY())
        try:
            if x_min < 0 or x_max > self.H or y_min < 0 or y_max > self.W:
                raise Exception('?????????????????????? ???? ???????????????????? ???? ????????????????')

            n = self.calculatingNormal(points[0].getX(),
                                       points[1].getX(),
                                       points[2].getX(),
                                       points[0].getY(),
                                       points[1].getY(),
                                       points[2].getY(),
                                       points[0].getZ(),
                                       points[1].getZ(),
                                       points[2].getZ())
            n_l = np.dot(n, [0, 0, 1])  # [0, 0, 1] ???????????????? ?????? ?????????? ???????????????????????????? ????????????????, ?????????????? ?????????? ?? ?????????????? ?????????????? ????????
            n_l_norm = n_l / (np.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))
            if n_l_norm > 0:
                return
            l = [0, 1, 0]   # ?????????????????????? ??????????
            n0 = normals[int(numberOfNormals[numberPointOne - 1])]
            n1 = normals[int(numberOfNormals[numberPointTwo - 1])]
            n2 = normals[int(numberOfNormals[numberPointThree - 1])]
            l0 = np.dot(n0, l) / (np.sqrt(n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2]) * np.sqrt(
                l[0] * l[0] + l[1] * l[1] + l[2] * l[2]))
            l1 = np.dot(n1, l) / (np.sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]) * np.sqrt(
                l[0] * l[0] + l[1] * l[1] + l[2] * l[2]))
            l2 = np.dot(n2, l) / (np.sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]) * np.sqrt(
                l[0] * l[0] + l[1] * l[1] + l[2] * l[2]))

            for x_i in range(round(x_min), round(x_max)):
                for y_j in range(round(y_min), round(y_max)):
                    ls_1 = self.calculateBarycentricCoordinates(x_i, y_j,
                                                                points[0].getX(),
                                                                points[0].getY(),
                                                                points[1].getX(),
                                                                points[1].getY(),
                                                                points[2].getX(),
                                                                points[2].getY())
                    peep = np.all(np.array(ls_1) >= 0)
                    if peep:
                        z_ = ls_1[0] * points[0].getZ() + ls_1[1] * points[1].getZ() + ls_1[2] * points[2].getZ()
                        if z_ > self.matrixZ[x_i][y_j]:
                            self.matrixZ[x_i][y_j] = z_
                            self.setPixel(x_i, y_j, 255 * (ls_1[0] * abs(l0) + ls_1[1] * abs(l1) + ls_1[2] * abs(l2)), 0, 0)
        except Exception as e:
            print("???????????? ?????? ?????????????????? ????????????????????????. ?????? ????????????:")
            print(e.args)

    def calculatingNormal(self, x0, x1, x2, y0, y1, y2, z0, z1, z2):
        vect1 = np.array([x1 - x0, y1 - y0, z1 - z0])
        vect2 = np.array([x1 - x2, y1 - y2, z1 - z2])
        n = np.cross(vect1, vect2)
        return n
            
    def getProjectiveTransformationPoints(self, points, t, intrinsic):
        points_np = np.array(points).T
        # ???????????????? ???????????????? ???????????????????? ???????????????????? ???? (?????????????? ?????????????????? + t)
        projectiveTransformationPoints = np.matmul(intrinsic, (points_np.reshape(3,1) + t))

        # ???????????????????????? ?????????????? ???? Z 
        # projectiveTransformationPoints /= projectiveTransformationPoints[2]  

        return projectiveTransformationPoints

    def getModelRotationPoints(self, projectiveTransformationPoints, angles):
        # ???????????????? ???????? ?? ????????????????
        alpha, beta, gamma = [180 * k/np.pi for k in angles]
        
        # ?????????????????? ????????
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        cosg = np.cos(gamma)
        sing = np.sin(gamma)

        #???????????? ?????????????? ????????????????
        matRotateX = np.array([[1, 0, 0], [0, cosa, sina], [0, -sina, cosa]]).reshape(3,3)
        matRotateY = np.array([[cosb, 0, sinb], [0, 1, 0], [-sinb, 0, cosb]]).reshape(3,3)
        matRotateZ = np.array([[cosg, sing, 0], [-sing, cosg, 0], [0, 0, 1]]).reshape(3,3)
        
        # ?????????????????? ?????????????? ???????????????? ???? ???????? ????????
        XY = np.matmul(matRotateX,matRotateY)
        R = np.matmul(XY, matRotateZ)
        
        # ?????????????????? ?????????????? ????????????
        turnPoints = np.matmul(R, projectiveTransformationPoints)
        turnPoints = turnPoints.T.tolist()

        return turnPoints

    def getModelRotationTransformationPoints(self, points, t, intrinsic, angles):
        points_list = []
        for point in points: 
            pointL = [point.getX(),point.getY(), point.getZ()]
            gptp = self.getProjectiveTransformationPoints(pointL, t, intrinsic)
            mrp = self.getModelRotationPoints(gptp, angles)
            points_list.append(Point(mrp[0][0], mrp[0][1], mrp[0][2]))
        return points_list