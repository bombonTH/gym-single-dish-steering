import math
import sqlite3

import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time

from point import Point


class Sun(Point):
    def __init__(self, name: str = 'sun', location=(13.7309711, 100.7873937, 15.0), time=0, radius=10, layer=1):
        super(Sun, self).__init__(name=name, layer=layer, location=location, radius=radius)

        self.connection = sqlite3.connect("db/sun.db")
        self.cursor = self.connection.cursor()

        self.location = EarthLocation(lat=location[0] * u.deg, lon=location[1] * u.deg, height=location[2] * u.m)
        self.time = Time(time)
        self.draw_boundary = math.pi / 2
        self.az = 0
        self.elv = 0
        self.frame = AltAz(obstime=self.time, location=self.location)

        self.update(0)

    def get_sun_position(self):
        self.cursor = self.connection.cursor()
        sun = get_sun(self.time).transform_to(self.frame)
        self.az = sun.az.hour * 15
        self.elv = sun.alt.hour * 15
        elevation = math.pi / 2 - self.elv / 180 * math.pi
        azimuth = self.az / 180 * math.pi
        self.ns = elevation * math.cos(azimuth)
        self.ew = elevation * math.sin(azimuth)
        print(str(self.time), self.az, self.elv, self.ns, self.ew)
        self.cursor.execute(f"INSERT INTO sun VALUES (?, ?, ?, ?, ?)", (str(self.time), self.az, self.elv, self.ns, self.ew))
        self.connection.commit()
        self.cursor.close()

    def update(self, time):
        self.time += time * u.second
        self.frame = AltAz(obstime=self.time, location=self.location)
        self.get_sun_position()

    def get_status(self) -> str:
        return f'SUN - Az: {self.az:06.2f} - Elv: {self.elv:05.2f} - Time: {self.time.to_datetime()} UT'
