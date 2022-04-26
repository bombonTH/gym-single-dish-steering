import math
import sqlite3

from point import Point
import numpy as np
import coordinate as cd
import math
from datetime import datetime, timedelta


class Airplane(Point):
    def __init__(self, name: str = 'air', location=(0, 0, 0), antenna=(1, 1, 1), time=None, radius=3, layer=2):
        super(Airplane, self).__init__(name=name, layer=layer, location=location, radius=radius)
        self.connection = sqlite3.connect("db/aircraft.db")
        self.cursor = self.connection.cursor()

        self.location = cd.Ellipsoidal(*location)
        self.origin = cd.Ellipsoidal(*antenna)
        self.init_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        self.time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        self.draw_boundary = math.pi / 2
        self.azimuth = 0
        self.elevation = 0
        self.distance = 0
        self.hit = 0
        self.track = 0
        self.speed = 0
        self.elaz = cd.ElAz(el=self.elevation, az=self.azimuth)
        self.xy = self.elaz.to_xy()
        self.update(0)

    def reset(self, random=False):
        self.time = self.init_time
        self.hit = 0
        self.done = False
        if random:
            _lat = self.origin.lat
            _lng = self.origin.lng + np.random.uniform(-2, 2)
            _alt = np.random.uniform(3000, 10000)
            _speed = np.random.uniform(300, 500)
            self.set_location(location=(_lat, _lng, _alt), track=0, speed=_speed)
            self.update(0)
            self.set_course((math.degrees(self.azimuth) + 180 + np.random.uniform(-2, 2)) % 360)

    def set_time(self, time):
        self.init_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        self.time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

    def set_location(self, location=(0, 0, 0), track=0.0, speed=0.0):
        self.location = cd.Ellipsoidal(*location)
        self.update(0)
        self.track = track
        self.speed = speed
        return self

    def set_course(self, track=0):
        self.track = track

    def dead_reckon(self):
        dif_time = (self.time - self.init_time).total_seconds()
        dif_lat = math.cos(math.radians(self.track)) * self.speed * dif_time / 216000
        dif_lng = math.sin(math.radians(self.track)) * self.speed * dif_time / 216000

        _lat = self.location.lat + dif_lat
        _lng = self.location.lng + dif_lng
        _alt = self.location.alt
        _location = cd.Ellipsoidal(_lat, _lng, _alt)
        az = self.origin.get_azimuth(_location)
        el = self.origin.get_elevation(_location)
        return el, az

    def update(self, time):
        self.time += timedelta(seconds=time)
        self.elaz = cd.ElAz(*self.dead_reckon())
        self.speedx = self.elaz.az - self.azimuth
        self.speedy = self.elaz.el - self.elevation
        self.azimuth = self.elaz.az
        self.elevation = self.elaz.el
        self.xy = self.elaz.to_xy()
        self.ns = self.xy.y
        self.ew = self.xy.x


air = Airplane(time='2021-08-16 05:00:00')
