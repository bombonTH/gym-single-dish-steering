import math
import numpy as np
import coordinate
from point import Point


class Geo(Point):
    def __init__(self, name: str = 'geo', el: float = 0.0, az: float = 0.0, radius=2, layer=2):
        super(Geo, self).__init__(name=name, layer=layer, location=(0, 0, 0), radius=radius)
        self.elaz = coordinate.ElAz(el=el, az=az)
        self.az = self.elaz.az
        self.el = self.elaz.el
        self.xy = self.elaz.to_xy()
        self.y = self.xy.y
        self.x = self.xy.x

    def reset(self, random=False):
        self.hit = 0
        self.done = False
        if random and False:
            el = np.random.uniform(0, 45)
            az = np.random.uniform(0, 360)
            self.elaz = coordinate.ElAz(el=el, az=az)
            self.az = self.elaz.az
            self.el = self.elaz.el
            self.xy = self.elaz.to_xy()
            self.ns = self.xy.y
            self.ew = self.xy.x

    def step(self, time):
        return self.elaz, self.xy

    def get_status(self) -> str:
        return f'GEO - NS: {self.ns:03.2f} - EW: {self.ew:03.2f}'
