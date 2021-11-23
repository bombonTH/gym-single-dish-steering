import math


class Point(object):
    def __init__(self, name: str, layer: int, location: (float, float, float), radius: int = 10):
        self.name = name
        self.layer = layer
        self.location = location
        self.ns = 0
        self.ew = 0
        self.radius = radius

    def set_position(self, ns: float, ew: float):
        self.ns = ns
        self.ew = ew

    def get_position(self):
        return self.ns, self.ew

    def draw(self, canvas):
        canvas = canvas[:, :, self.layer]
        h = canvas.shape[0]
        w = canvas.shape[1]
        y = (math.pi / 2 - self.ns) / math.pi * h - 0.5
        x = (self.ew + math.pi / 2) / math.pi * w - 0.5
        if (-self.radius < y < h + self.radius) and (-self.radius < x < w + self.radius):
            for k in range(max(0, math.floor(y - self.radius)), min(h, math.ceil(y + self.radius))):
                for j in range(max(0, math.floor(x - self.radius)), min(w, math.ceil(x + self.radius))):
                    distance = math.dist((k, j), (y, x))
                    canvas[k, j] = 1 - math.dist((k, j), (y, x)) / self.radius if distance < self.radius else 0

    def get_status(self) -> str:
        return f'NS: {self.ns:.2f} rad - EW: {self.ew:.2f} rad'
