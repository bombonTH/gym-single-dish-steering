import sqlite3
from geo import Geo

connection = sqlite3.connect("db/geo.db")
cursor = connection.cursor()


def load_geos(env):
    cur = cursor.execute("SELECT * FROM geo WHERE 1")
    for row in cur:
        env.add_obstacle(Geo(name=row[0], az=row[1], el=row[2], radius=row[3]))
