import sqlite3

connection = sqlite3.connect("db/sun.db")
cursor = connection.cursor()
cursor.execute("DROP TABLE IF EXISTS sun")
cursor.execute("CREATE TABLE sun (time DATETIME, az FLOAT, elv FLOAT, ns FLOAT, ew FLOAT)")
