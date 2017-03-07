import numpy as np
import cv2
import glob, os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')

def detectFaces_allFiles():
    for file in glob.glob("./FacePhoto/*.jpg"):
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(file)
        for (x,y,w,h) in faces:
            deltah = int(0.1 * h)
            deltaw = int(0.1 * w)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[max(0,y - deltah):min(y + h, height) + deltah, x:x + w]
            #cv2.rectangle(img,(x,y-deltah),(x+w,y+h+deltah),(255,0,0),2)

            """eyes = eye_cascade.detectMultiScale(roi_gray)
            smiles = smile_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            for (ex,ey,ew,eh) in smiles:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)"""
            print(x, y, w, h)

            cv2.imshow('img',img)
            cv2.imshow('Face', roi_color)
        while(1):
            if cv2.waitKey(1) & 0xFF == ord('n'):
                break
    cv2.destroyAllWindows()

def detectFaces_oneFile(src):
    img = cv2.imread(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[max(0, y - deltah):min(y + h, height) + deltah, x:x + w]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detectFaces_allFiles()