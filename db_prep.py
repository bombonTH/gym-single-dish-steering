import sqlite3

connection = sqlite3.connect("db/sun.db")
cursor = connection.cursor()
cursor.execute("CREATE TABLE sun (time TEXT, az FLOAT, elv FLOAT, ns FLOAT, ew FLOAT)")
