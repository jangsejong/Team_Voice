import pymysql
host ='db-sd2022.cbmzxbmxkhi4.us-east-1.rds.amazonaws.com'
port = 3306
master_admin = 'yhjh' 
db='chat_bot_db'
pw = 'study'

aws_db = pymysql.connect(
    user=master_admin, 
    passwd=pw, 
    host=host, 
    db=db, 
    charset='utf8'
)

cursor = aws_db.cursor(pymysql.cursors.DictCursor)
sql = "Show Databases"
cursor.execute(sql)
result = cursor.fetchall()

print(result)