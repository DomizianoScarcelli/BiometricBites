from ..isee_to_cost_calculator import cost_calculator
from ..dbconnector import dbconnector as db

def get_user_info(user_id):
    output_data = {
        "ID": "",
        "NAME": "",
        "SURNAME": "",
        "CF": "",
        "COST": ""
    }

    if user_id is None:
        return ValueError("You must specify the ID of the user.")
    conn = db.dbconnector()
    cursor = conn.cursor()
    query = "SELECT id, name, surname, cf, isee FROM users_info WHERE id='{}'".format(user_id)
    cursor.execute(query)
    ret = tuple(cursor.fetchall())
    if not ret:
        return ValueError("User not found.")
    else:
        ret = ret[0]
        output_data["ID"] = ret[0]
        output_data["NAME"] = ret[1]
        output_data["SURNAME"] = ret[2]
        output_data["CF"] = ret[3]
        output_data["COST"] = cost_calculator.cost_calculator(ret[4])
        return output_data