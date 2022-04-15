#from django.db import models

# Create your models here.
from django.db import models
#from phonenumber_field.modelfields import PhoneNumberField

# Create your models here.
class TrainModel(models.Model):

    facename=models.CharField(max_length=12)
    
