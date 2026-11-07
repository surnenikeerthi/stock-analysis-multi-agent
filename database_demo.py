import sqlite3
# helps in connecting to a database, if the database ,mentioned is not present then a new one is created
connection = sqlite3.connect("demo_data.db")
# cursor is something that acts as a connection with database
#through this sql commands are invoked
cursor = connection.cursor()
create_table_query = ("CREATE TABLE IF NOT EXISTS demo (stock_name char(100) PRIMARY KEY,"
                      " number_of_stocks float(10), "
                      "bought_stock_price float(10), "
                      "amount_spent_full float(100));")
cursor.execute(create_table_query)
delete_table_query = ("DROP TABLE IF EXISTS demo;")
#cursor.execute(delete_table_query)
#executed this just to confirm if the demo is dropped or not
select_command = ("SELECT * from demo;")
cursor.execute(select_command)
insert_command = ("INSERT INTO demo VALUES ('TCS', '10', '40', '400');")
cursor.execute(insert_command)
#tried inserting the same again ,and it throwed error
cursor.execute(select_command)
update_command = ("UPDATE demo ")

rows = cursor.fetchall()
print(rows)
print("yooo")
for row in rows:
    print(row)