import pyodbc
 

server = 'twittertestdb.database.windows.net'
database = 'TwitterTestDB'
username = 'db_owner_peter'
password = 'Testpassword123!'

connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password
cnxn = pyodbc.connect(connection_string)
cursor = cnxn.cursor()
cursor.execute("SELECT * FROM dbo.LND_TWITTER_DATA")
rows = cursor.fetchall()
for row in rows:
    print(row)