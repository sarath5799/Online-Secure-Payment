import os
import time
import imutils
foldername= input("Enter the Folder Name:")

path=os.getcwd()

path=path+'/'+foldername

print(path)

os.mkdir(path)

os.chdir(path)
import cv2


while(True):
    for i in range(0,20):
            start=time.time()
            cap=cv2.VideoCapture(0)
            print("Taking Image {}".format(i))
            ret, im =cap.read()
            im = imutils.resize(im, width=450)
            file = r"input{}.jpg".format(i)
            cv2.imwrite(file, im)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    final=time.time()       
    exe=final-start
    print("Execution Time:",exe)
cap.release()
cv2.destroyAllWindows()            
            