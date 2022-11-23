import os
from dotenv import load_dotenv

# .env 사용하기 위한거임

load_dotenv()

host=os.environ.get('capstone_db_url')
user=os.environ.get('capstone_user_id')
passwd=os.environ.get('capstone_user_passwd')

#print(host, user, passwd)

database='cafeishorse'

#os
#host='localhost'
#user='root'
#passwd='root'
#database='test'
autocommit=False
# Default 값은 3306  (Port 번호는 변경될 수 있음)
port=int(3306)