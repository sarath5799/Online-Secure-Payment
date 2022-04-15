from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
import numpy as np
import pandas as pd
import time
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
                                ListView,DetailView,
                                CreateView,DeleteView,
                                UpdateView)
from . import models
from django.core.files.storage import FileSystemStorage
import os
import imutils
from trainApp.extract2 import step2
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from numpy import asarray
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from mtcnn.mtcnn import MTCNN
import pickle
#from numpy import savez_compressed
from keras.models import load_model
class photodataView(View):
    form_class = TrainForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            foldername= request.POST.get('facename')
            path=r'static/data/train'

            path=path+'/'+foldername

            print(path)

            os.mkdir(path)

            #os.chdir(path)
            import cv2



            for i in range(0,10):
                        start=time.time()
                        cap=cv2.VideoCapture(0)
                        print("Taking Image {}".format(i))
                        ret, im =cap.read()
                        im = imutils.resize(im, width=450)
                        file = path+'/'+'input{}.png'.format(i)
                        cv2.imwrite(file, im)
                        success=1

            path1=r'static/data/test'

            path1=path1+'/'+foldername

            print(path1)

            os.mkdir(path1)

            #os.chdir(path)
            #import cv2



            for i in range(0,10):
                        start=time.time()
                        cap=cv2.VideoCapture(0)
                        print("Taking Image {}".format(i))
                        ret, im =cap.read()
                        im = imutils.resize(im, width=450)
                        file = path1+'/'+'input{}.png'.format(i)
                        cv2.imwrite(file, im)
                        success=1

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #        break
            final=time.time()
            exe=final-start
            print("Execution Time:",exe)
            cap.release()
            cv2.destroyAllWindows()

            return render(request, "succ_msgs.html", {'success':success,'foldername':foldername})
        else:
            success=0
            return render(request, "succ_msgs.html", {'success':success,'foldername':foldername})
class extractFace(View):
    form_class = TrainForm
    success_url = reverse_lazy('success')
    template_name = 'result.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        start=time.time()
        obj=step2()
                # load train dataset
        path=r'static/data/train/'
        path1=r'static/data/test/'
        try:
            trainX, trainy = obj.load_dataset(path)
            print(trainX.shape, trainy.shape)
            success="fail"
        except IndexError:
        # load test dataset
            return HttpResponse('No face was found')
        try:
            testX, testy = obj.load_dataset(path1)
            success="fail"
        except IndexError:
            return HttpResponse('No face was found')
        # save arrays to one file in compressed format
        savez_compressed('extract-faces-dataset.npz', trainX, trainy, testX, testy)

        success="suc"
        final=time.time()

        exe=final-start

        print("Execution Time:",exe)
                #form = self.form_class()
                #return render(request, self.template_name, {'form': form})

        return render(request, "succ_msgs.html", {'success':success})

    def post(self, request, *args, **kwargs):
        print('inside post')
class Faceembedd(View):
        form_class = TrainForm
        success_url = reverse_lazy('success')
        template_name = 'result.html'
        failure_url= reverse_lazy('fail')
        filenot_url= reverse_lazy('filenot')
        def get(self, request, *args, **kwargs):
            start=time.time()
            obj=step2()
                    # load train dataset
            data = load('extract-faces-dataset.npz')
            trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
            print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
            # load the facenet model
            model = load_model('facenet_keras.h5')
            print('Loaded Model')
            # convert each face in the train set to an embedding
            newTrainX = list()
            for face_pixels in trainX:
            	embedding = obj.get_embedding(model, face_pixels)
            	newTrainX.append(embedding)
            newTrainX = asarray(newTrainX)
            print(newTrainX.shape)
            # convert each face in the test set to an embedding
            newTestX = list()
            for face_pixels in testX:
            	embedding = obj.get_embedding(model, face_pixels)
            	newTestX.append(embedding)
            newTestX = asarray(newTestX)
            print(newTestX.shape)
            # save arrays to one file in compressed format
            savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

            success="suc1"

            return render(request, "succ_msgs.html", {'success':success})

        def post(self, request, *args, **kwargs):
            print('inside post')
class getsvc(View):
        form_class = TrainForm
        success_url = reverse_lazy('success')
        template_name = 'result.html'
        failure_url= reverse_lazy('fail')
        filenot_url= reverse_lazy('filenot')
        def get(self, request, *args, **kwargs):
            datas = load('extract-faces-dataset.npz')
            testX_faces = datas['arr_2']
            # load face embeddings
            datass = load('faces-embeddings.npz')
            trainXs, trainys, testXs, testys = datass['arr_0'], datass['arr_1'], datass['arr_2'], datass['arr_3']
            # normalize input vectors
            in_encoder = Normalizer(norm='l2')
            trainXs = in_encoder.transform(trainXs)
            testXs = in_encoder.transform(testXs)
            # label encode targets
            out_encoder = LabelEncoder()
            out_encoder.fit(trainys)
            trainys = out_encoder.transform(trainys)
            testys = out_encoder.transform(testys)
            # fit model
            models = SVC(kernel='linear', probability=True)
            models.fit(trainXs, trainys)
            filename = 'SVCModel.sav'
            pickle.dump(models, open(filename, 'wb'))

            success="suc2"

            return render(request, "succ_msgs.html", {'success':success})

        def post(self, request, *args, **kwargs):
            print('inside post')


class Success(TemplateView):
    template_name='succ_msgs.html'

class Failure(TemplateView):
    template_name='fail.html'

class FileNotfound(TemplateView):
    tempalte_name='filenot.html'
class AboutUs(TemplateView):
    template_name='aboutus.html'
