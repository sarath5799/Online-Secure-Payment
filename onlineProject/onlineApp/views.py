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
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import date, datetime
import cv2
start=time.time()
import imutils
from matplotlib import pyplot
from keras.models import load_model
#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras import backend as K
from numpy import asarray
from PIL import Image
from mtcnn.mtcnn import MTCNN
from datetime import date, datetime
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from skimage.feature import greycoprops
from skimage.feature import greycomatrix
from onlineApp.onlineFuntion import onlinefunction
class dataUploadView(View):
    form_class = OnlineForm
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
            shopnumber= request.POST.get('ShopNumber')
            yournumber=request.POST.get('CustomerNumber')
            EnterAmount=request.POST.get('Amount')
            obj=onlinefunction()
            #print (data)
            today=date.today()
            d1 = today.strftime("%d-%m-%Y")
            now=datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            customer=pd.read_csv('Customer_bank.csv')
            customer_mob=list(customer['Mobile_num'])
            Retailer=pd.read_csv('Retailer_bank.csv')
            if (int(yournumber) in customer_mob):

                #detectedname=input("Enter your name:")
                modelname = 'SVCModel.sav'
                model = pickle.load(open(modelname, 'rb'))
                data = load('faces-embeddings.npz')
                trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
                out_encoder = LabelEncoder()
                out_encoder.fit(trainy)


                model1 = load_model('facenet_keras.h5')
                print('Loaded Model')
                predicted_names=[]
                #cap = cv2.VideoCapture('http://192.168.1.37:8080/video')
                #while(True):
                for i in range(0,1):
                            start=time.time()
                            cap=cv2.VideoCapture(0)
                            print("Taking Image {}".format(i))
                            ret, im =cap.read()
                            im = imutils.resize(im, width=450)
                            file = r"static/output/input{}.png".format(i)
                            cv2.imwrite(file, im)
                            selection=file
                            try:
                                random_face_pixels =obj.extract_face(selection,required_size=(160, 160))
                                #mean=imgetech(img)
                            except IndexError:
                                return HttpResponse('No face was found, try again with proper light')
                                #pass
                            file2=r"static/output/input1.png"
                            try:
                                cv2.imwrite(file2,random_face_pixels )
                            except UnboundLocalError:
                                return HttpResponse('No face was found, try again with proper light')

                            mean=obj.imgetech(file2)
                            if(mean<=120):
                                random_face_emb = obj.get_embedding(model1, asarray(random_face_pixels))
                                samples = expand_dims(random_face_emb, axis=0)
                                yhat_class = model.predict(samples)
                                yhat_prob = model.predict_proba(samples)
                                # get name
                                class_index = yhat_class[0]
                                class_probability = yhat_prob[0,class_index] * 100
                                predict_names = out_encoder.inverse_transform(yhat_class)
                                #predicted_names.append(predict_names)
                                pyplot.imshow(random_face_pixels)
                                title = '%s (%.3f)' % (predict_names[0], class_probability)
                                pyplot.title(title)
                                pyplot.savefig("static/output/output.png")
                                #pyplot.show()
                                final=time.time()


                                detectedname=predict_names
								#print(detectedname)
								#print(detectedname)
                                ind=customer[customer['Mobile_num']==int(yournumber)].index[0]
                                namemobcheck=customer[customer['Mobile_num']==int(yournumber)]['Name'][ind]

                                if(detectedname==namemobcheck):
                                    print("Same name")
                                    cusbalance=customer[customer['Mobile_num']==int(yournumber)]['Balance'][ind]
                                    if(int(cusbalance)>int(EnterAmount)):
                                        idx_R=Retailer[Retailer['Mobile_num']==int(shopnumber)].index[0]
                                        idx_RR=Retailer[Retailer['Mobile_num']==int(shopnumber)].index
                                        retbalance=Retailer[Retailer['Mobile_num']==int(shopnumber)]['Balance'][idx_R]
                                        retbalance=retbalance+int(EnterAmount)
                                        Retailer.loc[idx_RR,'Balance']= retbalance
                                        Retailer.loc[idx_RR,'UpdatedDate']=d1 + dt_string
                                        print("Transaction Successfull")
                                        cusbalance=cusbalance-int(EnterAmount)
                                        idx_C=customer[customer['Mobile_num']==int(yournumber)].index
                                        customer.loc[idx_C,'Balance']= cusbalance
                                        customer.loc[idx_C,'updatedDate']= d1 + dt_string
                                        print("amount detected from customer")
                                        customer.to_csv("Customer_bank.csv", index=False)
                                        Retailer.to_csv("Retailer_bank.csv",index=False)
                                        success="Transaction Successfull"

                                        return render(request, "succ_msg.html", {'success':success})

                                    else:
                                        print("Low Balance")
                                        #low="Low Balance"
                                        return HttpResponse('Low Balance')

                                else:
                                    print("Not same name")
                                    #notsame="Not Same person"
                                    return HttpResponse('not same name')
                            else:
                                print("Dont try for proxy",mean)
                                #proxyyy="Dont Try for Proxy"
                                return HttpResponse('Dont try for proxy')

            else:
                print("Shop number or your mobile is incorrect")
                #incorrects="Shop number or your mobile is incorrect"
                return HttpResponse('Shop number or your mobile is incorrect')

            final=time.time()
            exe=final-start
            print("Execution Time:",exe)
        else:
            return redirect(self.failure_url)
class resultcheck1(View):
    form_class = OnlineForm
    success_url = reverse_lazy('success')
    template_name = 'result.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        #form = self.form_class()
        #return render(request, self.template_name, {'form': form})
        customer=pd.read_csv('Customer_bank.csv')
        Retailer=pd.read_csv('Retailer_bank.csv')
        return render(request, "process.html", {'customer':customer,'Retailer':Retailer})

    def post(self, request, *args, **kwargs):
        print('inside post')





class Success(TemplateView):
    template_name='succ_msg.html'

class Failure(TemplateView):
    template_name='fail.html'

class FileNotfound(TemplateView):
    tempalte_name='filenot.html'
class AboutUs(TemplateView):
    template_name='aboutus.html'
