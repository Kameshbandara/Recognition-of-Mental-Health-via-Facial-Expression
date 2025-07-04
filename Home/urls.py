from django.urls import path
from . import views

urlpatterns = [
    path('', views.stress_detection_page, name='Home'),  # Map to the main page
    path('video_feed/', views.video_feed, name='video_feed'),  # Stream video feed
]