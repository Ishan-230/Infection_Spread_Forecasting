from django.db import models

# 
from django.contrib.auth.models import User

# Create your models here.



class User_data(models.Model):
    username = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=30)

    def __str__(self):
        return self.username
