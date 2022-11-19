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
        "COST": ""
    }
    
    if request.method == "POST":
        req_data = request.POST
        for key, value in req_data.items():
            if key == "email":
                input_data["EMAIL"] = value
            if key == "password":
                input_data["PASSWORD"] = value
        if input_data["EMAIL"] == "" or input_data["PASSWORD"] == "":
            return JsonResponse({"message": "Parameters not valid."}, status=400)
        else:
            conn = db.dbconnector()
            cursor = conn.cursor()
            query = "SELECT users.id, email, role, name, surname, cf, isee FROM users, users_info WHERE email='{}' AND password='{}' AND users.id=users_info.id".format(input_data["EMAIL"], input_data["PASSWORD"])
            cursor.execute(query)
            ret = tuple(cursor.fetchall())
            if not ret:
                return JsonResponse({"message": "User not found."}, status=404)
            else:
                ret = ret[0]
                output_data["ID"] = ret[0]
                output_data["EMAIL"] = ret[1]
                output_data["ROLE"] = ret[2]
                output_data["NAME"] = ret[3]
                output_data["SURNAME"] = ret[4]
                output_data["CF"] = ret[5]
                output_data["COST"] = ret[6]
                return JsonResponse({"message": "OK", "data": json.dumps(output_data)}, status=200)
    return JsonResponse({"message": "Request not valid."}, status=400)

@csrf_exempt 
def get_user_info(request, *args, **kargs):
    input_data = {
        "ID": ""
    }

    output_data = {
        "ID": "",
        "NAME": "",
        "SURNAME": "",
        "CF": "",
        "COST": ""
    }

    if request.method == "GET":
        req_data = request.GET.get("id")
        if req_data is None or not req_data.isnumeric():
            return JsonResponse({"message": "ID not specified in the request."}, status=400)
        input_data["ID"] = req_data
        conn = db.dbconnector()
        cursor = conn.cursor()
        query = "SELECT id, name, surname, cf, isee FROM users_info WHERE id='{}'".format(input_data["ID"])
        cursor.execute(query)
        ret = tuple(cursor.fetchall())
        if not ret:
            return JsonResponse({"message": "User not found."}, status=404)
        else:
            ret = ret[0]
            output_data["ID"] = ret[0]
            output_data["NAME"] = ret[1]
            output_data["SURNAME"] = ret[2]
            output_data["CF"] = ret[3]
            output_data["COST"] = ret[4]
            return JsonResponse({"message": "OK", "data": json.dumps(output_data)}, status=200)