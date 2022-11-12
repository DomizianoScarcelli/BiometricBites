#Define all the API here by creating a new function with the name of the API
from django.http import JsonResponse
from .utils.recognition import faces

def api(request, *args, **kwargs):
    return JsonResponse({'message': 'Test Api'})

# def face_recognition(request, *args, **kwargs):
#     face_recognition()
#     return JsonResponse({'message': 'Test Api'})