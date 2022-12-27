from ..isee_to_cost_calculator import cost_calculator
from ..dbconnector import dbconnector as db
import os
from ..encoding.encoding import opencvimg_to_b64_str
import imghdr
from matplotlib.image import imread
from bsproject.settings import SAMPLES_ROOT

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

def get_profile_pic(id):
    supported_types = ['jpeg', 'jpg', 'png']
    sample_path = os.path.join(SAMPLES_ROOT, id)
    output_data = []
    if os.path.exists(sample_path):
        for file in os.listdir(sample_path):
            img_path = os.path.join(sample_path, file)
            if imghdr.what(img_path) in supported_types:
                img = opencvimg_to_b64_str(imread(img_path))
                output_data.append([file, img])
    output_data.sort(key=lambda x: x[0])
    return output_data[0][1]