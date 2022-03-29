import pymysql
import pandas as pd

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
    table = "Question_and_Answer"
    delete_table = f"Drop Table if exists {table}"
    cursor.execute(delete_table)
    print(delete_table)
    create_table= f"create table if not exists {table} (id MEDIUMINT NOT NULL AUTO_INCREMENT, Question VARCHAR(2000),Answer VARCHAR(2000),PRIMARY KEY (id))"
    cursor.execute(create_table)
    aws_db.commit()
    print(create_table)
    print()
    aws_db.close()    
    
def insert_QnA(q,a):
    table = "Question_and_Answer"
    sql = f"INSERT INTO {table} (Question,Answer) VALUES ('{q}','{a}');"
    cursor.execute(sql)
    # print(sql)
    aws_db.commit()   

def select_Table():
    table = "Question_and_Answer"
    sql = f"Select Question, Answer from {table}"
    cursor.execute(sql)
    result = pd.read_sql_query(sql,aws_db)    
    return result

if __name__ == '__main__':    
    talk = pd.DataFrame(select_Table())    
    print(type(talk))
    print(talk)
    




    

