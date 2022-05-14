import sqlite3
import pandas as pd

connection = sqlite3.connect("db/geo.db")
cursor = connection.cursor()
cursor.execute("DROP TABLE IF EXISTS geo")
cursor.execute("CREATE TABLE geo (name VARCHAR, az FLOAT, elevation FLOAT, radius INT)")



df = pd.read_csv('db/geo.csv', header=1)

for geo in df.values:
    print(geo)
    cursor.execute("INSERT INTO geo VALUES (?, ?, ?, ?)",
                   (geo[0], geo[1], geo[2], geo[3]))
connection.commit()