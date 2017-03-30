import image_face_detection as ifd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


if __name__ == '__main__':
    data = np.load('image_array.npy')
    for i in range(0, 10):
        img = data[i][:][:][:]
        plt.imshow(img)
        plt.show()