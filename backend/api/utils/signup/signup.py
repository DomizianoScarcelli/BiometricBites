'''
Note: this is a basic function to register a new user in the system. We assume that users can login using
an external service so the sign up operation is not needed in the real system.
This function takes a username (string), a password (string) and a role (enum('student', 'admin')).
Also, the password isn't hashed in the db (it is not safe but we don't need the system to be safe since
the system won't go live).
'''
import mysql.connector as sql
from bsproject import settings

def dbconnector():
    DBconf = settings.DATABASES.get("mysql")
    return sql.connect(host=DBconf.get("DB_HOST"), username=DBconf.get("DB_USERNAME"), passwd=DBconf.get("DB_PASSWORD"), database=DBconf.get("DB_NAME"))

def signup(username, password, role):
    if username == "" or password == "" or (role not in ["student", "admin"]):
        return ValueError("Unable to add a new user, please check the input values.")
    conn = dbconnector()
    cursor = conn.cursor()
    query = "INSERT INTO users (username, password, role) VALUES('{}', '{}', '{}')".format(username, password, role)
    cursor.execute(query)
    conn.commit()
    return "User added to DB!"

#Type the values here:
user = {
    "USERNAME": "alessio",
    "PASSWORD": "password",
    "ROLE": "admin" #admin or student
}
registration = signup(user.get("USERNAME"), user.get("PASSWORD"), user.get("ROLE"))
print(registration)