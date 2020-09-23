from django.urls import path
from TimeAxis import views


app_name = 'TimeAxis'
urlpatterns = [
    path('', views.chart, name='TimeAxis'),
]
