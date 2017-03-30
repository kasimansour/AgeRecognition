import numpy as np
import cv2
import glob, os
import re
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_smile.xml')

def displayFaces_allFiles(directory):
    """Displays :
           - images contained in a directory
            - faces detected in these images"""
    files_list = glob.glob(directory)
    for file in files_list:
        img = cv2.imread(file)
        height, width, channel = img.shape
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        #print("File : " + file)
        age = find_age(file)

        if age >= 0:
            if faces != ():
                cv2.imshow('img', img)
                for (x, y, w, h) in faces:
                    # On decale y vers le haut pour mieux centrer le visage
                    if y - int(0.1*h) >= 0:
                        y -= int(0.1*h)
                        h *= 1.2
                    else:
                        h += y + int(0.1*h)
                        y = 0
                    if h > width:
                        h = width
                    # A partir de l'origine du visage (point en haut a gauche), on definit
                    # notre carre, de cote le nouveau h
                    if x + 0.8*h > width:
                        x_right = width
                        x_left = width - int(h)
                    elif x - 0.2*h < 0:
                        x_left = 0
                        x_right = int(h)
                    else:
                        x_right = int(min(int(x) + int(0.8*h), int(width)))
                        x_left = x_right - int(h)
                    y_top = int(y)
                    y_bottom = int(y) + int(h)
                    roi_color = img[y_top:y_bottom, x_left:x_right]
                    cv2.imshow('Face', roi_color)
                    # Debug
                    if roi_color.shape[0] != roi_color.shape[1]:
                        print(file)
                        print(height, width)
                        print(y_top, y_bottom, x_left, x_right)
                        print(roi_color.shape)
                        print("EROOOOOOOR !!!!!!")
                """while(1):
                    if cv2.waitKey(1) & 0xFF == ord('n'):
                        break"""
    cv2.destroyAllWindows()


def detectFaces_allFiles(directory):
    """Detect faces in images, returns a list with images that contain faces
        and saves these face images in the FacePhoto directory"""
    files_list = glob.glob(directory)
    for file in files_list:
        img = cv2.imread(file)
        height, width, channel = img.shape
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        age = find_age(file)

        if age >= 0:
            if faces != ():
                for (x, y, w, h) in faces:
                    # On decale y vers le haut pour mieux centrer le visage
                    if y - int(0.1*h) >= 0:
                        y -= int(0.1*h)
                        h *= 1.2
                    else:
                        h += y + int(0.1*h)
                        y = 0
                    if h > width:
                        h = width
                    # A partir de l'origine du visage (point en haut a gauche), on definit
                    # notre carre, de cote le nouveau h
                    if x + 0.8*h > width:
                        x_right = width
                        x_left = width - int(h)
                    elif x - 0.2*h < 0:
                        x_left = 0
                        x_right = int(h)
                    else:
                        x_right = min(int(x) + int(0.8*h), int(width))
                        x_left = int(x_right) - int(h)
                    y_top = int(y)
                    y_bottom = int(y) + int(h)
                    roi_color = img[y_top:y_bottom, x_left:x_right]
                    cv2.imwrite("./FacePhoto/{}.jpg".format(extract_filename(file)), roi_color)
                    print("{} saved !".format(extract_filename(file)))
            else:
                files_list.remove(file)
        else:
            files_list.remove(file)
    cv2.destroyAllWindows()
    return files_list



def find_age(str):
    """Finds the age of the person on the photo, based on the filename"""
    regex = r"_([0-9]+).+_([0-9]+)"
    matches = re.search(regex, str)
    if matches:
        birth = matches.group(1)
        picture = matches.group(2)
        age = int(picture) - int(birth)
        return age


def extract_filename(str):
    """Finds the age of the person on the photo, based on the filename"""
    regex = r"([0-9_-]+).jpg"
    matches = re.search(regex, str)
    if matches:
        return matches.group(1)


def create_array(files_list):
    """Creates an array of images"""
    im_array = np.array([np.array(cv2.imread(file)) for file in files_list])
    return im_array

def save_array(array, filename):
    """Saves an array under the name 'filename' """
    np.save(filename, array)

def resize_image(image, side):
    """Resize an image to a square image (width = height = side)"""
    cv2.imshow("Original", image)
    small = cv2.resize(image, (side,side), interpolation = cv2.INTER_AREA)
    cv2.imshow("Small", small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    dir = "./photos/00/*.jpg"
    #displayFaces_allFiles(dir)
    detectFaces_allFiles(dir)
    face_dir = "./FacePhoto/*.jpg"
    face_photos_list = glob.glob(face_dir)
    my_array = create_array(face_photos_list)
    save_array(my_array, 'my_array.npy')
