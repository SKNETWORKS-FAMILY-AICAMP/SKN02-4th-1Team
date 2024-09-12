from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chatbot/', views.chatbot_response, name='chatbot_response'),
    path('askme/', views.askme, name='askme'),
    path('about/', views.about, name='about'),
    path('team/', views.team, name='team'),
]