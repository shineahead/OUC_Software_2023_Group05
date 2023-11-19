from django.contrib import admin
from django.urls import path

from .common import *

urlpatterns = [
    path('newTask/', chack)
]

