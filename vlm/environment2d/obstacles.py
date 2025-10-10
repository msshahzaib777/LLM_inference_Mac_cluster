import matplotlib.patches as patches

class Obstacle2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, ax):
        pass

    def translate(self, dx, dy):
        self.x += dx
        self.y += dy

    def rotate(self, angle_degrees):
        pass

class RectangleObstacle(Obstacle2D):
    def __init__(self, x, y, width, height, angle=0):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.angle = angle

    def draw(self, ax):
        rect = patches.Rectangle((self.x, self.y), self.width, self.height, angle=self.angle, color='black')
        ax.add_patch(rect)

    def rotate(self, angle_degrees):
        self.angle += angle_degrees


class CircleObstacle(Obstacle2D):
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self.radius = radius

    def draw(self, ax):
        circ = patches.Circle((self.x, self.y), radius=self.radius, color='black')
        ax.add_patch(circ)


class WallObstacle(Obstacle2D):
    def __init__(self, x1, y1, x2, y2):
        super().__init__(x1, y1)
        self.x2 = x2
        self.y2 = y2

    def draw(self, ax):
        ax.plot([self.x, self.x2], [self.y, self.y2], color='black', linewidth=2)

    def as_line_segment(self):
        return (self.x, self.y, self.x2, self.y2)

    def as_polygon(self, thickness=1):
        from shapely.geometry import LineString
        return LineString([(self.x, self.y), (self.x2, self.y2)]).buffer(thickness / 2, cap_style=2)