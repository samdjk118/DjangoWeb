from django.urls import path
from PieChart import views


app_name = 'PieChart'
urlpatterns = [
    path('', views.chart, name='PieChart'),
]
