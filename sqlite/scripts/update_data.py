import sqlite3

sql = "UPDATE tasks SET priority = ?, status_id = ? WHERE id = ?"

try:
    with sqlite3.connect("../DBs/my.db") as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (3, 2, 1))
        conn.commit
except sqlite3.Error as e:
    print(e)

sql2 = "UPDATE tasks SET end_date = ?"

try:
    with sqlite3.connect("../DBs/my.db") as conn:
        cursor = conn.cursor()
        cursor.execute(sql2, ("2024-08-12",))
        conn.commit()
except sqlite3.Error as e:
    print(e)
