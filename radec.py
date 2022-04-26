import math
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimeDelta
from point import Point
import coordinate


class RaDec(Point):
    def __init__(self, name='radec', location=(13.7309711, 100.7873937, 15), time=0, radius=2, layer=1, ra=0, dec=0):
        super(RaDec, self).__init__(name=name, layer=layer, location=location, radius=radius)

        self.location = EarthLocation(lat=location[0] * u.deg, lon=location[1] * u.deg, height=location[2] * u.m)
        self.skycoord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        self.init_time = time
        self.time = Time(time)
        
        self.draw_boundary = math.pi / 2

        self.frame = AltAz(obstime=self.time, location=self.location)

        self.update(0)

    def step(self, time):
        self.time += time * u.second
        self.frame = AltAz(obstime=self.time, location=self.location)
        altaz = self.skycoord.transform_to(self.frame)
        elaz = coordinate.ElAz(el=altaz.alt.hour * 15, az=altaz.az.hour * 15)
        xy = elaz.to_xy()
        
        return elaz, xy
        
    def reset(self, random=False):
        self.time = self.init_time
        self.hit = 0
        self.done = False
        self.update()
       
    def set_time(self, time):
        self.init_time = time
        self.time = Time(time)
        self.update(0)

    def get_status(self) -> str:
        return f'RADEC - Az: {self.azimuth:06.2f} - Elv: {self.elevation:05.2f} - Time: {self.time.to_datetime()} UT'
