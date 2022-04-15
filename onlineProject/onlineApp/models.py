#from django.db import models

# Create your models here.
from django.db import models
#from phonenumber_field.modelfields import PhoneNumberField

# Create your models here.
class OnlineModel(models.Model):

    ShopNumber=models.CharField(max_length=12)
    CustomerNumber=models.CharField(max_length=12)
    #pedigree=models.FloatField()
    Amount =models.FloatField()
