import sqlite3

## Coonect to sqlite
connection = sqlite3.connect("student.db")

## Create a cursor object to insert record, create table
cursor = connection.cursor()

## Create table
table_info = """
    create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25),
    SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

## Insert more records
cursor.execute('''Insert Into STUDENT values('Rakshitha', 'AI', 'A', 90)''')
cursor.execute('''Insert Into STUDENT values('Sinshu', 'Gen-AI', 'A', 90)''')
cursor.execute('''Insert Into STUDENT values('Varshini', 'Javascript', 'A', 95)''')
cursor.execute('''Insert Into STUDENT values('Prakruthi', 'AI', 'A', 98)''')


## To display the records
print("Inserted records are")
data = cursor.execute('''Select * from STUDENT''')

for row in data:
    print(row)

## To commit changes in databases
connection.commit()
connection.close()