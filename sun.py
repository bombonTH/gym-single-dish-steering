import math
import sqlite3

import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time, TimeDelta
from point import Point
import numpy as np
import coordinate


class Sun(Point):
    def __init__(self, name: str = 'sun', location=(13.7309711, 100.7873937, 15.0), time=0, radius=2, layer=1):
        super(Sun, self).__init__(name=name, layer=layer, location=location, radius=radius)

        self.connection = sqlite3.connect("db/sun.db")
        self.cursor = self.connection.cursor()

        self.location = EarthLocation(lat=location[0] * u.deg, lon=location[1] * u.deg, height=location[2] * u.m)
        self.init_time = time
        self.time = Time(time)

        self.frame = AltAz(obstime=self.time, location=self.location)

        self.update(0)
    
    def step(self, time):
        self.time += time * u.second
        self.frame = AltAz(obstime=self.time, location=self.location)
        return self.get_sun_position()
        

    def reset(self, random=False):
        self.set_time(self.init_time)
        if random:
            self.time = self.time + int(np.random.uniform(-21600, 21600)) * u.second
        self.hit = 0
        self.done = False

    def set_time(self, time):
        self.time = Time(time)

    def get_sun_position(self):
        self.cursor = self.connection.cursor()
        time = str(self.time)
        sun_position = self.read_sun(time)

        if sun_position is None:
            print('Not found - calculating position')
            az, alt = self.cal_sun(self.time)
            self.elaz = coordinate.ElAz(el=alt, az=az)
            self.xy = self.elaz.to_xy()
            y = self.xy.y
            x = self.xy.x
            az = self.elaz.az
            elv = self.elaz.el            
            self.write_sun(time, az, elv, y, x)
            sun_position = (time, az, elv, y, x)
        else:
            time, az, elv, y, x = sun_position 
            self.elaz = coordinate.ElAz(el=math.degrees(elv), az = math.degrees(az))
            self.xy = self.elaz.to_xy()

        self.cursor.close()
        return self.elaz, self.xy

    def read_sun(self, time):
        self.cursor.execute("SELECT * FROM sun WHERE time = ?", (time,))
        return self.cursor.fetchone()

    def cal_sun(self, time):
        sun = get_sun(time).transform_to(self.frame)
        az = sun.az.hour * 15
        alt = sun.alt.hour * 15
        return az, alt

    def write_sun(self, time, az, elv, ns, ew):
        self.cursor.execute("INSERT INTO sun VALUES (?, ?, ?, ?, ?)",
                            (time, az, elv, ns, ew))
        self.connection.commit()

    def get_status(self) -> str:
        return f'SUN - Az: {self.azimuth:06.2f} - Elv: {self.elevation:05.2f} - Time: {self.time.to_datetime()} UT'
