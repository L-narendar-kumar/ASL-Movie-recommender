# my_movie_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('',views.startPage),
    path('movie/',views.index2),
    path('movieSearch/', views.movie_recommendations, name='movie_recommendations'),
    # ... any other URL patterns you might have for your app
]