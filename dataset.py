import numpy as np
#import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tempfile import TemporaryFile


class Dataset:
    def __init__(self): 
        # Init params
        self.train_size = 4668 # size of train set
        self.test_size = 5532 # size of test set
        self.scale_size = 227 
        self.mean = np.array([104., 117., 124.]) # RGB values of the mean-image of ILSVRC
        self.n_classes = 7
        self.cur = 0 # for train
        self.cur_test = 0 # for test    

    def next_batch(self, batch_size, phase, loaded_img, loaded_lab):
        # Get next batch of images and labels
        if phase == 'train':
            if self.cur + batch_size < self.train_size: 
                images = loaded_img[self.cur:self.cur+batch_size][:][:][:]
                one_hot_labels = loaded_lab[self.cur:self.cur+batch_size][:]
                self.cur += batch_size
            else:
                images = np.concatenate((loaded_img[self.cur:][:][:][:],loaded_img[:(self.cur+batch_size)%self.train_size][:][:][:] ),0)
                one_hot_labels = np.concatenate((loaded_lab[self.cur:][:],loaded_lab[:(self.cur+batch_size)%self.train_size][:] ),0)
                self.cur = (self.cur+batch_size)%self.train_size
        elif phase == 'test':
            if self.cur_test + batch_size < self.test_size:
                images = loaded_img[self.cur_test:self.cur_test+batch_size][:][:][:]
                one_hot_labels = loaded_lab[self.cur_test:self.cur_test+batch_size][:]
                self.cur_test += batch_size
            else:
                images = np.concatenate((loaded_img[self.cur_test:][:][:][:] , loaded_img[:(self.cur_test+batch_size)%self.test_size][:][:][:] ),0)
                one_hot_labels = np.concatenate((loaded_lab[self.cur_test:][:],loaded_lab[:(self.cur_test+batch_size)%self.test_size][:] ),0)
                self.cur_test = (self.cur_test+batch_size)%self.test_size
        else:
            return None, None
        
        return images, one_hot_labels



