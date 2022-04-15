from django import forms
from .models import *


class TrainForm(forms.ModelForm):
    class Meta():
        model=TrainModel
        fields=['facename']
