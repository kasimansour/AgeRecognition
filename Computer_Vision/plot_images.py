import image_face_detection as ifd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


if __name__ == '__main__':
    dir = "./photos/00/*.jpg"
    file = glob.glob(dir)[0]
    img = cv2.imread(file)
    plt.imshow(img)
    plt.show()

    data = np.load('my_array.npy')
    img = data[0][:][:][:]
    plt.imshow(img)
    plt.show()

    """data = np.load('my_array.npy')
    for i in range(0, 10):
        img = data[0][:][:][:]
        plt.imshow(img)
        plt.show()"""