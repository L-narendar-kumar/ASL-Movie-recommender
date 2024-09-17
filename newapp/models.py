# models.py (in your Django app)
from django.db import models

class Movie(models.Model):
    title = models.CharField(max_length=255)
    # ... other fields you need
    poster_url = models.URLField()  # Field for the poster URL