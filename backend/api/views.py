#Define all the API here by creating a new function with the name of the API
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils.recognition import faces
from .utils.dbconnector import dbconnector as db
import json

def api(request, *args, **kwargs):
    return JsonResponse({'message': 'Test Api'})

@csrf_exempt 
def login(request, *args, **kwargs):
    input_data = {
        "EMAIL": "",
        "PASSWORD": ""
    }

    output_data = {
        "ID": "",
        "EMAIL": "",
        "ROLE": "",
        "NAME": "",
        "SURNAME": "",
        "CF": "",
        "ISEE": ""
    }
    
    if request.method == "POST":
        req_data = request.POST.dict()
        for key, value in req_data.items():
            if key == "email":
                input_data["EMAIL"] = value
            if key == "password":
                input_data["PASSWORD"] = value
    if input_data["EMAIL"] == "" or input_data["PASSWORD"] == "":
        return JsonResponse({"status": "400", "message": "Parameters not valid."})
    else:
        conn = db.dbconnector()
        cursor = conn.cursor()
        query = "SELECT users.id, email, role, name, surname, cf, isee FROM users, users_info WHERE email='{}' AND password='{}' AND users.id=users_info.id".format(input_data["EMAIL"], input_data["PASSWORD"])
        cursor.execute(query)
        ret = tuple(cursor.fetchall())
        if not ret:
            return JsonResponse({"status": "404", "message": "User not found."})
        else:
            ret = ret[0]
            output_data["ID"] = ret[0]
            output_data["EMAIL"] = ret[1]
            output_data["ROLE"] = ret[2]
            output_data["NAME"] = ret[3]
            output_data["SURNAME"] = ret[4]
            output_data["CF"] = ret[5]
            output_data["ISEE"] = ret[6]
            return JsonResponse({"status": "200", "message": "OK", "data": json.dumps(output_data)})