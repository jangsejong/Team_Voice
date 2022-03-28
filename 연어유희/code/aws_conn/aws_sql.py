from venv import create
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

def init_table():
    table_list = ["Answer, Question"]
    for table in table_list:
        delete_table = f"Drop Table if exist {table}"
        cursor.execute(delete_table)
        aws_db.commit()
        print(delete_table)
        
        create_table = f"Create table if not exist {table} (id not null auto_increment, txt Varchar(2000), primary key (id))"
        cursor.execute(create_table)
        aws_db.commit()
        print(create_table)
    aws_db.close()
    
def insert_table(table, txt):
    sql = f"Insert Into {table} (txt) Values ('{txt}');"
    cursor.execute(sql)
    print(sql)
    aws_db.commit()
    
    

