#Define all the API here by creating a new function with the name of the API
from django.conf import settings
from django.http import JsonResponse, QueryDict
from django.views.decorators.csrf import csrf_exempt
from .utils.dbconnector import dbconnector as db
from .utils.isee_to_cost_calculator import cost_calculator
import imghdr
import json
import os
from .utils.encoding.encoding import b64str_to_opencvimg
import cv2

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
                output_data["COST"] = cost_calculator.cost_calculator(ret[6])
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
            output_data["COST"] = cost_calculator.cost_calculator(ret[4])
            return JsonResponse({"message": "OK", "data": json.dumps(output_data)}, status=200)
    return JsonResponse({"message": "Request not valid."}, status=400)

'''
API get_attendance_list: Get the attendance list of the user
Type: GET Request
Inputs:
-id (integer): the id of the user;
Output:
-List(Attendance): all the attendances in JSON format where Attendance: {"user_id": integer, "attendance_id": integer, "paid": float, "date": timestamp}
'''
@csrf_exempt
def get_attendance_list(request, *args, **kargs):
    input_data = {
        "ID": ""
    }

    output_data = []

    if request.method == "GET":
        req_data = request.GET.get("id")
        if req_data is None or not req_data.isnumeric():
            return JsonResponse({"message": "ID not specified in the request."}, status=400)
        input_data["ID"] = req_data
        conn = db.dbconnector()
        cursor = conn.cursor()
        query = "SELECT * FROM users_attendance WHERE user_id='{}'".format(input_data["ID"])
        cursor.execute(query)
        ret = tuple(cursor.fetchall())
        if not ret:
            return JsonResponse({"message": "OK", "data": []}, status=200)
        else:
            for attendance in ret:
                attendance_entry = {
                    "user_id": attendance[0],
                    "attendance_id": attendance[1],
                    "paid": attendance[2],
                    "date": json.dumps(attendance[3].isoformat(), indent=1, sort_keys=True, default=str)
                }
                output_data.append(attendance_entry)
            return JsonResponse({"message": "OK", "data": json.dumps(output_data)}, status=200)
    return JsonResponse({"message": "Request not valid."}, status=400)

'''
API add_attendance: Add a new attendance to the canteen
Type: PUT Request
Inputs:
-user_id (integer): the id of the user;
-paid (float): how much he spent;
'''
@csrf_exempt
def add_attendance(request, *args, **kargs):
    input_data = {
        "USER_ID": "",
        "PAID": ""
    }

    if request.method == "PUT":
        req_data = QueryDict(request.body)
        for key, value in req_data.items():
            if key == "user_id":
                input_data["USER_ID"] = value
            if key == "paid":
                input_data["PAID"] = value
        if input_data["USER_ID"] == "" or input_data["PAID"] == "":
            return JsonResponse({"message": "Parameters not valid."}, status=400)
        else:
            print(input_data)
            conn = db.dbconnector()
            try:
                cursor = conn.cursor()
                query = "INSERT INTO users_attendance(attendance_id, user_id, paid, date) VALUES (DEFAULT, '{}', '{}', current_timestamp())".format(input_data["USER_ID"], input_data["PAID"])
                cursor.execute(query)
                conn.commit()
                return JsonResponse({"message": "OK"}, status=200) 
            except SystemError as err:
                conn.rollback()
                return JsonResponse({"message": err}, status=500)
    return JsonResponse({"message": "Request not valid."}, status=400)

@csrf_exempt
def get_photo_list(request, *args, **kargs):
    supported_types = ['jpeg', 'jpg', 'png']
    input_data = {
        "ID": ""
    }

    output_data = []

    if request.method == "GET":
        req_data = request.GET.get("id")
        if req_data is None or not req_data.isnumeric():
            return JsonResponse({"message": "ID not specified in the request."}, status=400)
        input_data["ID"] = req_data
        sample_path = os.path.join(settings.SAMPLES_ROOT, input_data["ID"])
        if os.path.exists(sample_path):
            for file in os.listdir(sample_path):
                img_path = os.path.join(sample_path, file)
                if imghdr.what(img_path) in supported_types:
                    output_data.append(file)
            return JsonResponse({"message": "OK", "data": json.dumps(output_data)}, status=200)
        else:
            return JsonResponse({"message": "The specified user has no photos."}, status=404)
    return JsonResponse({"message": "Request not valid."}, status=400)

@csrf_exempt
def delete_photo(request, *args, **kargs):
    input_data = {
        "ID": "",
        "NAME": ""
    }

    if request.method == "DELETE":
        req_data = request.GET
        if req_data.get("id") is None or not req_data.get("id").isnumeric():
            return JsonResponse({"message": "ID not specified in the request."}, status=400)
        if req_data.get("name") is None:
            return JsonResponse({"message": "Sample name to delete not specified in the request."}, status=400)
        input_data["ID"] = req_data.get("id")
        input_data["NAME"] = req_data.get("name")
        sample_path = os.path.join(settings.SAMPLES_ROOT, input_data["ID"], input_data["NAME"])
        if os.path.exists(sample_path):
            os.remove(sample_path)
            return JsonResponse({"message": "OK"}, status=200)
        else:
            return JsonResponse({"message": "The photo which has to be deleted, doesn't exist."}, status=404)
    return JsonResponse({"message": "Request not valid."}, status=400)


@csrf_exempt 
def upload_photo_enrollment(request, *args, **kargs):
    """
    Takes the list of photos that the user made in order to enroll himself and trains the model.
    """
    if request.method == "POST":
        req_data = json.loads(request.POST.get("photoList"))
        if req_data is None:
            return JsonResponse({"message": "Photo data not specified in the request in the field 'data'."}, status=400)
        for index, img in enumerate(req_data):
            opencv_img = b64str_to_opencvimg(img)
            # TODO: do training with the image here
            cv2.imwrite(f"image_{index}.jpeg", opencv_img) #TODO: remove this line since it's just for testing
        return JsonResponse({"message": "Photo uploaded correctly"}, status=200)
        