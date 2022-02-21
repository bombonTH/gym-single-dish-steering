import math


class Point(object):
    def __init__(self, name: str, layer: int, location: (float, float, float), radius: int = 10):
        self.name = name
        self.layer = layer
        self.location = location
        self.ns = 0
        self.ew = 0
        self.radius = radius
        self.graphic = []
        self.hit = 0
        self.done = False
        self.out_of_bound = False


    def set_position(self, ns: float, ew: float):
        self.ns = ns
        self.ew = ew

    def get_position(self):
        return self.ns, self.ew

    def set_radius(self, radius: int):
        self.radius = radius

    def reset(self, random=False):
        pass

    def draw(self, canvas):
        if self.done:
            return
        canvas = canvas[:, :, self.layer]
        h = canvas.shape[0]
        w = canvas.shape[1]
        y = (math.pi / 2 - self.ns) / math.pi * h - 0.5
        x = (self.ew + math.pi / 2) / math.pi * w - 0.5
        if (-self.radius < y < h + self.radius) and (-self.radius < x < w + self.radius):
            for k in range(max(0, math.floor(y - self.radius)), min(h, math.ceil(y + self.radius))):
                for j in range(max(0, math.floor(x - self.radius)), min(w, math.ceil(x + self.radius))):
                    distance = math.dist((k, j), (y, x))
                    pixel = int(255 * (1 - distance / self.radius)) if distance < self.radius else canvas[k, j]
                    canvas[k, j] = min(pixel+canvas[k, j], 255)

    def get_status(self) -> str:
        return f'NS: {self.ns:.2f} rad - EW: {self.ew:.2f} rad'
