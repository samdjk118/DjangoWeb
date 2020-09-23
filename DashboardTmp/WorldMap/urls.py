from django.urls import path
from WorldMap import views


app_name = 'WorldMap'
urlpatterns = [
    path('', views.chart, name='WorldMap'),
]
