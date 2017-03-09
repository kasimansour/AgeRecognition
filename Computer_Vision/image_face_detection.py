import numpy as np
import cv2
import glob, os
import re
from PIL import Image
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')

def detectFaces_allFiles(directory):
    files_list = glob.glob(directory)
    for file in files_list:
        img = cv2.imread(file)
        plt.imshow(img)
        plt.show()
        """height, width = img.shape
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        print("File : " + file)
        age = find_age(file)
        cv2.imshow('img', img)
        if age >= 0:
            for (x,y,w,h) in faces:
                deltah = int(0.1 * h)
                roi_color = img[max(0,y - deltah):min(y + h, height), x:x + w]
                cv2.imshow('Face', roi_color)
                #cv2.imwrite("./photos/{}.jpg".format(), roi_color)
        while(1):
            if cv2.waitKey(1) & 0xFF == ord('n'):
                break
    cv2.destroyAllWindows()"""



def find_age(str):
    regex = r"_([0-9]+).+_([0-9]+)"
    matches = re.search(regex, str)
    if matches:
        birth = matches.group(1)
        picture = matches.group(2)
        age = int(picture) - int(birth)
        return age

def create_array(directory):
    files_list = glob.glob(directory)
    im_array = np.array([np.array(cv2.imread(file)) for file in files_list if find_age(file) >= 0])
    return im_array

def save_array(array, filename):
    np.save(filename, array)

if __name__ == '__main__':
    dir = "./photos/00/*.jpg"
    #detectFaces_allFiles(dir)
    my_array = create_array(dir)
    save_array(my_array, 'my_array.npy')
