"""Diabetes URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from onlineApp import views
from trainApp import viewss
from django.conf.urls import url,include

urlpatterns = [
    path('/',include('onlineApp.urls',namespace='onlineApp')),
    path('train/',include('trainApp.urls',namespace='trainApp')),
    path('', views.dataUploadView.as_view(), name = 'data'),
    path('result', views.resultcheck1.as_view(), name = 'result'),
    path('train/photo', viewss.photodataView.as_view(), name = 'traindata'),
    path('train/extract', viewss.extractFace.as_view(), name = 'face'),
    path('train/embed', viewss.Faceembedd.as_view(), name = 'embed'),
    path('train/svcmodel', viewss.getsvc.as_view(), name = 'svc'),

    #path('whats', views.WhatsappAnalaysis.as_view(), name = 'whats'),
    #path('success', views.Success.as_view(), name = 'success'),
    #path('fail',views.Failure.as_view(),name='fail'),
    #path('filenot',views.FileNotfound.as_view(), name='filenot'),
    path('aboutus',views.AboutUs.as_view(), name='aboutus'),
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
