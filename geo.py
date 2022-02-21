import math
import numpy as np
from point import Point


class Geo(Point):
    def __init__(self, name: str = 'geo', ns: float = 0.0, ew: float = 0.0, radius=2, layer=2):
        super(Geo, self).__init__(name=name, layer=layer, location=(0, 0, 0), radius=radius)
        self.draw_boundary = math.pi / 2
        self.ns = ns
        self.ew = ew
        self.hit = 0

    def set_position(self, ns, ew):
        self.ns = ns
        self.ew = ew

    def reset(self, random=False):
        self.hit = 0
        self.done = False
        if random:
            self.set_position(np.random.uniform(-math.pi / 2, math.pi / 2), np.random.uniform(-math.pi / 2, math.pi / 2))
        

    def update(self, time):
        pass

    def get_status(self) -> str:
        return f'GEO - NS: {self.ns:03.2f} - EW: {self.ew:03.2f}'
