#Map here the API you create with an url
from django.urls import path

from . import views

urlpatterns = [
    path('', views.api), #This API will be reachable at the address localhost:8000/api/
    # path('', views.face_recognition)
]