import tensorflow as tf
import sys
from model import Model
from dataset import Dataset
from network import load_with_skip
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tempfile import TemporaryFile
import numpy as np

import struct # for lazy load

def main():    
    # Load dataset
    dataset = Dataset() 
    #loaded_img_train = np.load("train_images.npy")
    #loaded_lab_train = np.load("train_labels.npy")
    #loaded_img_test = np.load("/home/yt244701/Enseignement/CentraleSupelec/TP_DeepLearning/TP3_finetuning/test_images.npy")
    #loaded_lab_test = np.load("/home/yt244701/Enseignement/CentraleSupelec/TP_DeepLearning/TP3_finetuning/test_labels.npy")
    
    # Alternative "lazy load"
    #"""
    with open('train_images.npy', 'r') as f:
        junk, header_len = struct.unpack('<8sh', f.read(10))
        f.close()
    loaded_img_train= np.memmap('train_images.npy',dtype='float64',shape=(4000, 227, 227, 3),offset=6+2+2+header_len)
        
        
    with open('train_labels.npy', 'r') as f:
        junk, header_len = struct.unpack('<8sh', f.read(10))
        f.close()
    loaded_lab_train= np.memmap('train_labels.npy',dtype='float64',shape=(4000, 40),offset=6+2+2+header_len)
    
    with open('test_images.npy', 'r') as f:
        junk, header_len = struct.unpack('<8sh', f.read(10))
        f.close()
    loaded_img_test= np.memmap('test_images.npy',dtype='float64',shape=(5532, 227, 227, 3),offset=6+2+2+header_len)
        
        
    with open('test_labels.npy', 'r') as f:
        junk, header_len = struct.unpack('<8sh', f.read(10))
        f.close()
    loaded_lab_test= np.memmap('test_labels.npy',dtype='float64',shape=(5532, 40),offset=6+2+2+header_len)
    
    
    #"""
    

    
    # Display 10 training images (after mean-image substraction and resize to 227x227)
    for i in range(0,10):
       img = loaded_img_train[i][:][:][:]
       plt.imshow(img)
       plt.show()
 
        
    # Learning params
    learning_rate = 0.001
    batch_size = 20
    training_iters = 1000 # nbr_epochs = (batch_size * training_iters)/nbr_training_images 
    display_step = 1 # display training information (loss, training accuracy, ...) every 10 iterations
    test_step = 100 # test every test_step iterations 

    # Network params
    n_classes = 7
    keep_rate = 0.5 # for dropout

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    pred = Model.alexnet(x, keep_var) # definition of the network architecture
    
    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Init
    init = tf.global_variables_initializer()    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Load pretrained model
        load_with_skip('pretrained_alexnet.npy', sess, ['fc8']) # Skip weights from fc8 (fine-tuning)

        print('Start training.')
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train', loaded_img_train, loaded_lab_train)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})
           
            #"""
            # Display testing status
            if step%test_step == 0:
                test_acc = 0.
                test_count = 0
                for _ in range(int(dataset.test_size/batch_size)+1): # test accuracy by group of batch_size images
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test', loaded_img_test, loaded_lab_test)
                    print(batch_tx.shape)
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print ("{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc), file=sys.stderr)
            #"""
            # Display training status
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.}) # Training-accuracy
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.}) # Training-loss
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc),file=sys.stderr)
     
            step += 1
        print("Finish!")
         
if __name__ == '__main__':
    main()













