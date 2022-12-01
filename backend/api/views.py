#Define all the API here by creating a new function with the name of the API
from django.http import JsonResponse

def api(request, *args, **kwargs):
    return JsonResponse({'message': 'Test Api'})

def faces(request, *args, **kwargs): #TODO
    return JsonResponse({'message': "Ciao :)"})