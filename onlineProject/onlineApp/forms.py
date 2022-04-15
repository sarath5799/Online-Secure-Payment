from django import forms
from .models import *


class OnlineForm(forms.ModelForm):
    class Meta():
        model=OnlineModel
        fields=['ShopNumber','CustomerNumber','Amount']
