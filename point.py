import math
import coordinate


class Point(object):
    def __init__(self, name: str, layer: int, location: (float, float, float), radius: int = 2, dwell: int = 10):
        self.name = name
        self.layer = layer
        self.radius = radius
        self.location = location
        
        self.az = 0.0
        self.el = math.pi/2
        self.az_rate = 0.0
        self.el_rate = 0.0
        self.y = 0.0
        self.x = 0.0
        self.azs = [0.0, 0.0, 0.0, 0.0]
        self.els = [0.0, 0.0, 0.0, 0.0]
        self.xs = [0.0, 0.0, 0.0, 0.0]
        self.ys = [0.0, 0.0, 0.0, 0.0]
        self.y_rate = 0.0
        self.x_rate = 0.0
        self.dwell = dwell
        self.xy = coordinate.XY(x=math.degrees(self.x), y=math.degrees(self.y))
        self.elaz = coordinate.ElAz(az=math.degrees(self.az), el=math.degrees(self.el))
        self.pixel = [0, 0]
        self.distance = 0
        
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
        
    def distance_to(self, other):
        return self.elaz.distance_to(other.elaz)

    def equal(self, other, precision=4):
        threshold = math.pow(0.1, precision)
        return self.distance_to(other) < threshold
    
    def same_pace(self, other, precision=8):
        threshold = math.pow(0.1, precision)
        dif_y_rate = self.az_rate - other.az_rate
        dif_x_rate = self.el_rate - other.el_rate
        rate = math.pow(dif_y_rate,2) + math.pow(dif_x_rate,2)
        return math.sqrt(rate) < threshold
        
    
    def update(self, time=0):
        self.elaz, self.xy = self.step(time)
        self.az_rate = self.elaz.az - self.az
        self.el_rate = self.elaz.el - self.el
        self.az = self.elaz.az
        self.el = self.elaz.el
        self.azs[3] = self.azs[2]
        self.azs[2] = self.azs[1]
        self.azs[1] = self.azs[0]
        self.azs[0] = self.az
        self.els[3] = self.els[2]
        self.els[2] = self.els[1]
        self.els[1] = self.els[0]
        self.els[0] = self.el
        
        self.x_rate = self.xy.x - self.x
        self.y_rate = self.xy.y - self.y
        self.x = self.xy.x
        self.y = self.xy.y
        self.xs[3] = self.xs[2]
        self.xs[2] = self.xs[1]
        self.xs[1] = self.xs[0]
        self.xs[0] = self.x
        self.ys[3] = self.ys[2]
        self.ys[2] = self.ys[1]
        self.ys[1] = self.ys[0]
        self.ys[0] = self.y
        
        self.out_of_bound = not (self.min_x < self.x < self.max_x and self.min_y < self.y < self.max_y)
        return(self)
        
    def step(self, time):
        return coordinate.ElAz(az=math.degrees(self.az), el=math.degrees(self.el)), coordinate.XY(x=math.degrees(self.x), y=math.degrees(self.y))

    def reset(self, random=False):
        self.distance = 0.0
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
        self.pixel = [int(x), int(y)]
        if (-self.radius < y < h + self.radius) and (-self.radius < x < w + self.radius):
            for k in range(max(1, math.floor(y - self.radius)), min(h - 2, math.ceil(y + self.radius))):
                for j in range(max(1, math.floor(x - self.radius)), min(w - 2, math.ceil(x + self.radius))):
                    distance = math.dist((k, j), (y, x))
                    pixel = int(192 * (1 - distance / self.radius)) if distance < self.radius else 0
                    pixel = min(pixel, 255 - canvas[k, j])
                    canvas[k, j] = pixel + canvas[k, j]
        return self

    def get_status(self) -> str:
        return f'NS: {self.ns:.2f} rad - EW: {self.ew:.2f} rad'
