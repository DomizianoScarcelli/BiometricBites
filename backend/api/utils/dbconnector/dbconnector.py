import mysql.connector as sql
from bsproject import settings

def dbconnector():
    DBconf = settings.DATABASES.get("mysql")
    return sql.connect(host="", user="root", passwd="", database="bsproject")
