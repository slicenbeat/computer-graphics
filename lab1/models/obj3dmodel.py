from point import Point

class OBJ3DModel:
    def __init__(self, path: str):
        f = open(path, mode='r')
        self.points = []
        self.polygons = []
        for line in f:
            if line[:2] == 'v ':
                line_of_numbers = [float(i) for i in line[2:].strip('\n').split(' ')]
                self.points.append(Point(line_of_numbers[0], line_of_numbers[1], line_of_numbers[2]))
            elif line[:2] == 'f ':
                line_of_numbers = line[2:].strip('\n').split(' ')
                temp_polygons = []
                for i in range(3):
                    line_to_numbers = line_of_numbers[i].split('/')
                    temp_polygons.append(int(line_to_numbers[0]))
                self.polygons.append(temp_polygons)

    def getPoints(self):
        return self.points

    def getPolygons(self):
        return self.polygons
