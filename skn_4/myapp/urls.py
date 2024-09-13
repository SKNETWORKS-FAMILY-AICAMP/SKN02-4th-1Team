from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home 페이지
    path('about/', views.about, name='about'),  # About 페이지
    path('team/', views.team, name='team'),  # Contact 페이지
    path('chatbot/', views.chatbot_response, name='chatbot_response'),
]