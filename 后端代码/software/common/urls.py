from django.contrib import admin
from django.urls import path

from .common import SARView

urlpatterns = [
    path('SARDetection/', SARView.as_view())
]

