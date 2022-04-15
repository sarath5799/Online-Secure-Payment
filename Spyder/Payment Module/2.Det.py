import pandas as pd
from datetime import date, datetime
import cv2
import pickle
import time
start=time.time()
import cv2
import imutils
import pandas as pd
from matplotlib import pyplot
from keras.models import load_model
from numpy import asarray
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
from datetime import date, datetime
# develop a classifier for the 5 Celebrity Faces Dataset
#from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
today=date.today()
d1 = today.strftime("%d-%m-%Y")

now=datetime.now()
dt_string = now.strftime("%H:%M:%S")

customer=pd.read_csv('data/Customer_bank.csv')

customer_mob=list(customer['Mobile_num'])



Retailer=pd.read_csv('data/Retailer_bank.csv')

shopnumber=input("Enter Shop Number:")

yournumber=input("Enter yournumber Number:")

EnterAmount=input("Enter amount to be paid:")


if (int(yournumber) in customer_mob):

    #detectedname=input("Enter your name:")
    modelname = 'Face_detect/finalized_model.sav'
    model = pickle.load(open(modelname, 'rb'))
    data = load('Face_detect/5-celebrity-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    def extract_face(filename, required_size=(160, 160)):
        	# load image from file
        	image = Image.open(filename)
        	# convert to RGB, if needed
        	#image = image.convert('RGB')
        	# convert to array
        	pixels = asarray(image)
        	# create the detector, using default weights
        	detector = MTCNN()
        	# detect faces in the image
        	results = detector.detect_faces(pixels)
        	# extract the bounding box from the first face
        	x1, y1, width, height = results[0]['box']
        	# bug fix
        	x1, y1 = abs(x1), abs(y1)
        	x2, y2 = x1 + width, y1 + height
        	# extract the face
        	face = pixels[y1:y2, x1:x2]
        	# resize pixels to the model size
        	image = Image.fromarray(face)
        	image = image.resize(required_size)
        	face_array = asarray(image)
        	return face_array
        
    def get_embedding(model, face_pixels):
        	# scale pixel values
        	face_pixels = face_pixels.astype('float32')
        	# standardize pixel values across channels (global)
        	mean, std = face_pixels.mean(), face_pixels.std()
        	face_pixels = (face_pixels - mean) / std
        	# transform face into one sample
        	samples = expand_dims(face_pixels, axis=0)
        	# make prediction to get embedding
        	yhat = model.predict(samples)
        	return yhat[0]
    
    model1 = load_model('face_detect/facenet_keras.h5')
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
                file = r"input{}.jpg".format(i)
                cv2.imwrite(file, im)
                selection=file
                try:
                    random_face_pixels =extract_face(selection,required_size=(160, 160))
                    
                except IndexError: 
                    pass
                random_face_emb = get_embedding(model1, asarray(random_face_pixels))
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
                pyplot.savefig("output.png")
                pyplot.show()
                final=time.time()
                
                
                detectedname=predict_names
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
                        customer.to_csv("data/Customer_bank.csv", index=False)
                        Retailer.to_csv("data/Retailer_bank.csv",index=False)
                        
                    else:
                        print("Low Balance")
                        
                   
                else:
                    print("Not same name")
else:
    print("Shop number or your mobile is incorrect")
final=time.time()

exe=final-start

print("Execution Time:",exe)    
    
    
    

