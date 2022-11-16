import mysql.connector as sql
from bsproject import settings

def dbconnector():
    DBconf = settings.DATABASES.get("mysql")
    return sql.connect(host=DBconf.get("DB_HOST"), username=DBconf.get("DB_USERNAME"), passwd=DBconf.get("DB_PASSWORD"), database=DBconf.get("DB_NAME"))

