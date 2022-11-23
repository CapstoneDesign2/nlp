import os

host=os.environ['capstone_db_url']
user=os.environ['capstone_user_id'] # server 에서는 os.env 써야한다!
passwd=os.environ['capstone_user_passwd'] # server 에서는 os.env 써야한다!
database='cafeishorse'

#os


#host='localhost'
#user='root'
#passwd='root'
#database='test'
autocommit=False
# Default 값은 3306  (Port 번호는 변경될 수 있음)
port=int(3306)