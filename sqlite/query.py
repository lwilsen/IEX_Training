import sqlite3

try:
    with sqlite3.connect("my.db") as conn:
        cur = conn.cursor()
        cur.execute("select id, name, priority from tasks")
        rows = cur.fetchall()
        for row in rows:
            print(row)

except sqlite3.Error as e:
    print(e)

try:
    with sqlite3.connect("my.db") as conn:
        cur = conn.cursor()
        cur.execute("select id, name, priority from tasks where id = ?", (1,))
        rows = cur.fetchall()
        for row in rows:
            print(row)

except sqlite3.Error as e:
    print(e)

"""
Use the fetchall() method of the cursor object to return all rows of a query.
Use the fetchone() method to return the next row returned by a query.
Use the fetchmany() method to return some rows from a query.
"""