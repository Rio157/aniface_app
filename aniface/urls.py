
from django.contrib import admin
from django.urls import path
from . import views

app_name = 'aniface'
urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
]


