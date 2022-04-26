import math
import coordinate


class Point(object):
    def __init__(self, name: str, layer: int, location: (float, float, float), radius: int = 2):
        self.name = name
        self.layer = layer
        self.radius = radius
        self.location = location
        
        self.az = 0.0
        self.el = 0.0
        self.az_rate = 0.0
        self.el_rate = 0.0
        self.y = 0.0
        self.x = 0.0
        self.y_rate = 0.0
        self.x_rate = 0.0
        self.xy = coordinate.XY()
        self.elaz = coordinate.ElAz()
        
        self.min_x = math.radians(-50)
        self.max_x = math.radians(50)
        self.min_y = math.radians(-80)
        self.max_y = math.radians(80)
        
        self.hit = 0
        self.done = False
        self.out_of_bound = False
        self.obstructed = False

    def set_location(self, location: (float, float, float)):
        self.location = location
        return self

    def set_radius(self, radius: int):
        self.radius = radius

    def equal(self, other, precision=4):
        threshold = math.pow(0.1, precision)
        return self.elaz.distance_to(other.elaz) < threshold
    
    def same_pace(self, other, precision=8):
        threshold = math.pow(0.1, precision)
        dif_y_rate = abs(self.y_rate - other.y_rate)
        dif_x_rate = abs(self.x_rate - other.x_rate)
        
        return dif_y_rate < threshold and dif_x_rate < threshold
        
    
    def update(self, time=0):
        self.elaz, self.xy = self.step(time)
        self.az_rate = self.elaz.az - self.az
        self.el_rate = self.elaz.el - self.el
        self.az = self.elaz.az
        self.el = self.elaz.el
        
        self.x_rate = self.xy.x - self.x
        self.y_rate = self.xy.y - self.y
        self.x = self.xy.x
        self.y = self.xy.y
        
        self.out_of_bound = not (self.min_x < self.x < self.max_x and self.min_y < self.y < self.max_y)
        
    def step(self, time):
        return coordinate.ElAz(az=self.az, el=self.el), coordinate.XY(x=self.x, y=self.y)

    def reset(self, random=False):
        pass

    def draw(self, canvas):
        if self.done:
            return

        canvas = canvas[:, :, self.layer]
        zenith_angle = (math.pi / 2) - self.el

        h = canvas.shape[0]
        w = canvas.shape[1]
        hor_h = int(0.95 * h)
        hor_w = int(0.95 * w)
        y = h/2 - (zenith_angle * math.cos(self.az) / math.pi * hor_h)
        x = (zenith_angle * math.sin(self.az) / math.pi * hor_w) + w/2
        if (-self.radius < y < h + self.radius) and (-self.radius < x < w + self.radius):
            for k in range(max(1, math.floor(y - self.radius)), min(h - 2, math.ceil(y + self.radius))):
                for j in range(max(1, math.floor(x - self.radius)), min(w - 2, math.ceil(x + self.radius))):
                    distance = math.dist((k, j), (y, x))
                    pixel = int(192 * (1 - distance / self.radius)) if distance < self.radius else 0
                    pixel = min(pixel, 255 - canvas[k, j])
                    canvas[k, j] = pixel + canvas[k, j]

    def get_status(self) -> str:
        return f'NS: {self.ns:.2f} rad - EW: {self.ew:.2f} rad'
