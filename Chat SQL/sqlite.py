import sqlite3

## code to connect 

connect= sqlite3.connect("student.db")

## create cursor object to insert record

cursor=connect.cursor()

table_info="create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT)"

cursor.execute(table_info)

## INSERT some more records

cursor.execute('''Insert into STUDENT values('Krish','Data Science','A',90)''')
cursor.execute('''Insert into STUDENT values('Jhon','Data Science','A',80)''')
cursor.execute('''Insert into STUDENT values('Mukesh','Data Science','A',100)''')
cursor.execute('''Insert into STUDENT values('Jacob','Devops','A',50)''')
cursor.execute('''Insert into STUDENT values('Dipesh','Devops','A',35)''')

## Dislay all the records

data = cursor.execute('''select * from STUDENT''')
for row in data:
    print(row)


connect.commit()
connect.close()