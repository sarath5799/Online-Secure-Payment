import pandas as pd
from datetime import date, datetime
import cv2
import pickle
import time
start=time.time()
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
from PIL import Image
from skimage.feature import greycoprops
from skimage.feature import greycomatrix
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#rom keras import backend as K


class onlinefunction:
    def __init__(self):
        pass

    def imgetech(self,img):
        rim=Image.open(img,'r')
        #rim = rim.convert('RGB')
        rim=rim.resize((160,160))
        rpix_val=list(rim.getdata())
        #rpix_val=facepixel
        rpix_val_flat=[x for sets in rpix_val for x in sets]
        rmean=sum(rpix_val_flat)/len(rpix_val_flat)
        #rmeans.append(rmean)
        print(rmean)
        return rmean

    def extract_face(self,filename, required_size=(160, 160)):
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

    def get_embedding(self,model, face_pixels):
                # scale pixel values
                face_pixels = face_pixels.astype('float32')
                # standardize pixel values across channels (global)
                mean, std = face_pixels.mean(), face_pixels.std()
                face_pixels = (face_pixels - mean) / std
                # transform face into one sample
                samples = expand_dims(face_pixels, axis=0)
                # make prediction to get embedding
                yhat = model.predict(samples)
                #K.clear_session()
                return yhat[0]
