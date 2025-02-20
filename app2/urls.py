# # app2/urls.py

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.index, name='index'),
#     path('signup/', views.signup, name='signup'),
#     path('login/', views.login, name='login'),
#     path('profile/', views.profile, name='profile'),
#     path('logout/', views.logout, name='logout'),
#     path('update/', views.update, name='update'),
#     path('delete/', views.delete, name='delete'),
#     path('forecasting/', views.forecasting, name='forecasting'),
#     path('datainsight/', views.data_insight, name='data_insight'),
#     path('aboutus/', views.about_us, name='about_us'),
#     path('contactus/', views.contact_us, name='contact_us'),

# ]

# app2/urls.py
from django.urls import path
from . import views
# from .views import forecasting


urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.index, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login, name='login'),
    path('profile/', views.profile, name='profile'),
    path('logout/', views.logout, name='logout'),
    path('update/', views.update, name='update'),
    path('delete/', views.delete, name='delete'),
    path('forecasting/', views.forecasting, name='forecasting'),
    path('datainsight/', views.datainsight, name='datainsight'),
    path('aboutus/', views.aboutus, name='aboutus'),
    path('contactus/', views.contactus, name='contactus'),
    # path('', forecasting, name='forecasting'),
    
    path('forecasting/', views.forecasting, name='forecasting'),
]
