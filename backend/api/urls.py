#Map here the API you create with an url
from django.urls import path

from . import views

urlpatterns = [
    path('', views.api), #This API will be reachable at the address localhost:8000/api/
    path('login', views.login),
    path('get_user_info', views.get_user_info),
    path('get_attendance_list', views.get_attendance_list),
    path('add_attendance', views.add_attendance),
    path('get_photo_list', views.get_photo_list),
    path('delete_photo', views.delete_photo),
    path('upload_photo_enrollment', views.upload_photo_enrollment)
]