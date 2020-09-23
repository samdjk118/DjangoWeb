from django.urls import path
from AngularGauge import views


app_name = 'AngularGauge'
urlpatterns = [
    path('', views.chart, name='AngularGauge'),
]
