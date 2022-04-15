import time
start=time.time()
#folder = 'E:/Ramisha_Hands_on_Project/Infoziant/Face_New/dataset/data/train/ben_afflek/'
# demonstrate face detection on 5 Celebrity Faces Dataset
# face detection for the 5 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import PIL
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
# extract a single face from a given photograph
class step2:
    def __init__(self):
        pass
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

    # load images and extract faces for all images in a directory
    def load_faces(self,directory):
    	faces = list()
    	# enumerate files
    	for filename in listdir(directory):
    		# path
    		path = directory + filename
    		# get face
    		face = obj.extract_face(path)
    		# store
    		faces.append(face)
    	return faces

    # load a dataset that contains one subdir for each class that in turn contains images
    def load_dataset(self,directory):
    	X, y = list(), list()
    	# enumerate folders, on per class
    	for subdir in listdir(directory):
    		# path
    		path = directory + subdir + '/'
    		# skip any files that might be in the dir
    		if not isdir(path):
    			continue
    		# load all faces in the subdirectory
    		faces = obj.load_faces(path)
    		# create labels
    		labels = [subdir for _ in range(len(faces))]
    		# summarize progress
    		print('>loaded %d examples for class: %s' % (len(faces), subdir))
    		# store
    		X.extend(faces)
    		y.extend(labels)
    	return asarray(X), asarray(y)

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
    	return yhat[0]

obj=step2()
