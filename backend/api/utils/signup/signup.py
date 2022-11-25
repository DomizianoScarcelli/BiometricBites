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

def signup(email, password, role, name, surname, cf, isee):
    if email == "" or password == "" or (role not in ["student", "admin"]) or name == "" or surname == "" or len(cf) != 16 or not isee.isnumeric():
        return ValueError("Unable to add a new user, please check the input values.")
    conn = dbconnector()
    cursor = conn.cursor()
    try:
        query_one = "INSERT INTO users (email, password, role) VALUES('{}', '{}', '{}')".format(email, password, role)
        cursor.execute(query_one)
        lastid = cursor.lastrowid
        query_two = "INSERT INTO users_info (id, name, surname, cf, isee) VALUES('{}', '{}', '{}', '{}', '{}')".format(lastid, name, surname, cf, isee)
        cursor.execute(query_two)
        conn.commit()
        return "User added to DB!"
    except SystemError as err:
        conn.rollback()
        return err

#Type the values here:
user = {
    "EMAIL": "prova@gmail.com", #Note this is a unique value (you can't add the same email for two different users)
    "PASSWORD": "password", #varchar
    "ROLE": "admin", #enum(admin, student)
    "NAME": "ale", #varchar
    "SURNAME": "luc", #varchar
    "CF": "ABCDEFGHILMNOPQR", #varchar - 16 characters
    "ISEE": "15000" #float
}
registration = signup(user.get("EMAIL"), user.get("PASSWORD"), user.get("ROLE"), user.get("NAME"), user.get("SURNAME"), user.get("CF"), user.get("ISEE"))
print(registration)