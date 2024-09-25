import pandas as pd
import pickle
import sqlite3

train = pd.DataFrame(pickle.load(open("train.pkl", "rb")))
test = pd.DataFrame(pickle.load(open("test.pkl", "rb")))


def create_titanic(filename):
    conn = None
    try:
        conn = sqlite3.connect(filename)
        print("File created or connection opened")
        train.to_sql("train", conn, if_exists="replace")
        test.to_sql("test", conn, if_exists="replace")
        conn.execute(
            """
            create table my_table as 
            select * from my_data
            """
        )
        print("Tables created")
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
            print("connection closed")


if __name__ == "__main__":
    create_titanic("titanic.db")
