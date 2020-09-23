from django.urls import path
from SimChart import views


app_name = 'SimChart'
urlpatterns = [
    path('', views.chart, name='Simchart'),
]
