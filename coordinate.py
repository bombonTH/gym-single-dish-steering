import math
from dataclasses import dataclass

import constant


@dataclass
class Coordinate:
    lat: float = 0.0
    lng: float = 0.0
    alt: float = 0.0

    def __post_init__(self):
        self.lat_dd = int(self.lat)
        self.lat_m = self.lat % 1 * 60
        self.lat_mm = int(self.lat_m)
        self.lat_ss = math.floor(self.lat_m % 1 * 60 * 10000) / 10000
        self.lng_dd = int(self.lng)
        self.lng_m = self.lng % 1 * 60
        self.lng_mm = int(self.lng_m)
        self.lng_ss = math.floor(self.lng_m % 1 * 60 * 10000) / 10000
        self.radius = get_earth_radius(self.lat)
        self.horizon = get_horizon(self.alt, self.radius)
        self.km_per_deg = 2 * math.pi * self.radius / 360


def get_earth_radius(_lat):
    lat = math.radians(_lat)
    a = pow(pow(constant.RADIUS_EQUATOR, 2) * math.cos(lat), 2) + \
        pow(pow(constant.RADIUS_POLAR, 2) * math.sin(lat), 2)
    b = pow(constant.RADIUS_EQUATOR * math.cos(lat), 2) + \
        pow(constant.RADIUS_POLAR * math.sin(lat), 2)
    return math.sqrt(a / b)


def get_horizon(_alt, _radius):
    return math.sqrt(pow(_radius + _alt, 2) - pow(_radius, 2))


def get_slant_range(_origin, _target):
    earth_radius = (_origin.radius + _target.radius) / 2
    km_per_degree = (_origin.km_per_deg + _target.km_per_deg) / 2
    dist_degree = math.radians(get_straight_range(_origin, _target) / km_per_degree)
    a = pow(earth_radius + _origin.alt, 2) + pow(earth_radius + _target.alt, 2)
    b = 2 * (earth_radius + _origin.alt) * (earth_radius + _target.alt) * math.cos(dist_degree)
    return math.sqrt(a - b)


def get_straight_range(_origin, _target):
    earth_radius = (_origin.radius + _target.radius) / 2
    lat_origin = math.radians(_origin.lat)
    lat_target = math.radians(_target.lat)
    delta_lng = math.radians(_target.lng - _origin.lng)
    a = math.sin(lat_origin) * math.sin(lat_target) + math.cos(lat_origin) * math.cos(lat_target) * math.cos(delta_lng)
    return earth_radius * math.acos(min(a, 1))


def get_elevation(_origin, _target):
    try:
        earth_radius = (_origin.radius + _target.radius) / 2
        km_per_degree = (_origin.km_per_deg + _target.km_per_deg) / 2
        straight_range = get_straight_range(_origin, _target)
        slant_range = get_slant_range(_origin, _target)
        d = math.radians(straight_range / km_per_degree)
        sin_d_by_slant = math.sin(d) / slant_range
        horizon = _origin.horizon + _target.horizon
        depression = math.asin(sin_d_by_slant * (earth_radius + _origin.alt))
        elevation = math.acos(sin_d_by_slant * (earth_radius + _target.alt))
    except ValueError:
        elevation = math.pi / 2
    return math.degrees(elevation if slant_range < horizon else -elevation)


def get_azimuth(_origin, _target):
    lat_origin = math.radians(_origin.lat)
    lat_target = math.radians(_target.lat)
    delta_lng = math.radians(_target.lng - _origin.lng)
    y = math.sin(delta_lng) * math.cos(lat_target)
    x = math.cos(lat_origin) * math.sin(lat_target) - math.sin(lat_origin) * math.cos(lat_target) * math.cos(delta_lng)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


@dataclass
class Ellipsoidal:
    lat: float = 0.0
    lng: float = 0.0
    alt: float = 0.0

    def __post_init__(self):
        self.lat_dd = int(self.lat)
        self.lat_m = self.lat % 1 * 60
        self.lat_mm = int(self.lat_m)
        self.lat_ss = math.floor(self.lat_m % 1 * 60 * 10000) / 10000
        self.lng_dd = int(self.lng)
        self.lng_m = self.lng % 1 * 60
        self.lng_mm = int(self.lng_m)
        self.lng_ss = math.floor(self.lng_m % 1 * 60 * 10000) / 10000

    def get_azimuth(self, other):
        lat_origin = math.radians(self.lat)
        lat_target = math.radians(other.lat)
        delta_lng = math.radians(other.lng - self.lng)
        y = math.sin(delta_lng) * math.cos(lat_target)
        a = math.cos(lat_origin) * math.sin(lat_target)
        b = math.sin(lat_origin) * math.cos(lat_target) * math.cos(delta_lng)
        x = a - b
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def get_elevation(self, other):
        return self.to_cartesian().get_elevation(other.to_cartesian())

    def to_cartesian(self):
        a = 6378137
        ecc = 0.0818191908426215
        e2 = math.pow(ecc, 2)

        phi_e = math.radians(self.lat)
        lam_e = math.radians(self.lng)
        h = self.alt

        n = a / math.sqrt(1 - e2 * math.pow(math.sin(phi_e), 2))  # radius of curvature in the prime vertical
        x = (n + h) * math.cos(phi_e) * math.cos(lam_e)
        y = (n + h) * math.cos(phi_e) * math.sin(lam_e)
        z = (n * (1 - e2) + h) * math.sin(phi_e)

        return Cartesian(x, y, z)


@dataclass
class Cartesian:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_ellipsoidal(self):
        a = 6378137
        f = 1 / 298.257223563
        e2 = 2 * f - math.pow(f, 2)
        b = a * (1 - f)

        lam = math.atan2(self.y, self.x)

        p = math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))
        mu = math.atan(a * self.z / (b * p))
        ea2 = e2 / (1 - e2)

        phi = math.atan2(self.z + ea2 * b * math.sin(mu), e2 * a * math.cos(mu))

    def get_zenith_angle(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        a = self.x * dx + self.y * dy + self.z * dz
        b = math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2)
        c = math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2)
        zenith_angle = math.degrees(math.acos(a / math.sqrt(b * c)))
        return zenith_angle

    def get_elevation(self, other):
        return 90 - self.get_zenith_angle(other)

    def get_azimuth(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        abs1 = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2))
        x2 = math.pow(self.x, 2)
        y2 = math.pow(self.y, 2)
        z2 = math.pow(self.z, 2)

        a = (-self.z * self.x * dx - self.z * self.y * dy) + (x2 + y2) * dz
        b = math.sqrt((x2 + y2) * (x2 + y2 + z2) * (math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2)))
        cos_az = math.acos(a / b)
        d = -self.y * dx + self.x * dy
        e = math.sqrt((x2 + y2) * (math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2)))
        sin_az = math.asin(d / e)

@dataclass
class XY:
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self):
        self.x = math.radians(self.x)
        self.y = math.radians(self.y)

    def to_altaz(self):
        el = math.asin(math.cos(self.y) * math.cos(self.x))
        az = math.atan2(math.tan(self.x), math.sin(self.y))
        el = math.degrees(el)
        az = math.degrees(az) % 360
        return ElAz(el=el, az=az)

    def print_degree(self):
        return f'X: {math.degrees(self.x):.6f} Y: {math.degrees(self.y):.6f}'


@dataclass
class ElAz:
    el: float = 0.0
    az: float = 0.0

    def __post_init__(self):
        self.el = math.radians(self.el)
        self.az = math.radians(self.az)

    def to_xy(self):
        x = math.asin(math.sin(self.az) * math.cos(self.el))
        y = math.atan2(math.cos(self.az) * math.cos(self.el), math.sin(self.el))
        x = math.degrees(x)
        y = math.degrees(y)
        return XY(x=x, y=y)
    
    def distance_to(self, other):
        a = (math.pi / 2) - self.el
        b = (math.pi / 2) - other.el
        g = abs(self.az - other.az)
        return math.acos(math.cos(a) * math.cos(b) + math.sin(a) * math.sin(b) * math.cos(g))

    def print_degree(self):
        return f'El: {math.degrees(self.el):.6f} Az: {math.degrees(self.az):.6f}'