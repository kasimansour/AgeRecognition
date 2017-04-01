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
from sklearn import model_selection

import struct # for lazy load

def main():    
    # Load dataset
    dataset = Dataset() 
    X = np.load("Computer_Vision/train_val_images.npy")
    Y = np.load("train_val_labels.npy")
    print(X.shape)
    print(Y.shape)
    #loaded_img_test = np.load("/home/yt244701/Enseignement/CentraleSupelec/TP_DeepLearning/TP3_finetuning/test_images.npy")
    #loaded_lab_test = np.load("/home/yt244701/Enseignement/CentraleSupelec/TP_DeepLearning/TP3_finetuning/test_labels.npy")

    # """ Alternative "lazy load"

    # with open('Computer_Vision/train_validation_image_array.npy', 'rb') as f:
    #     junk, header_len = struct.unpack('<8sh', f.read(10))
    #     f.close()
    # X = np.memmap('Computer_Vision/train_validation_image_array.npy',dtype='float64',shape=(12646, 227, 227, 3),offset=6+2+2+header_len)
    # print(X.shape)
    #
    # with open('train_validation_label_array.npy', 'rb') as f:
    #     junk, header_len = struct.unpack('<8sh', f.read(10))
    #     f.close()
    #
    # Y = np.memmap('train_validation_label_array.npy',dtype='float64',shape=(12646, 7),offset=6+2+2+header_len)

    # with open('test_images.npy', 'r') as f:
    #     junk, header_len = struct.unpack('<8sh', f.read(10))
    #     f.close()
    # loaded_img_test= np.memmap('test_images.npy',dtype='float64',shape=(5532, 227, 227, 3),offset=6+2+2+header_len)
    #
    #
    # with open('test_labels.npy', 'r') as f:
    #     junk, header_len = struct.unpack('<8sh', f.read(10))
    #     f.close()
    # loaded_lab_test= np.memmap('test_labels.npy',dtype='float64',shape=(5532, 40),offset=6+2+2+header_len)


    # """

    kf = model_selection.KFold(n_splits=4)
    for train_index, val_index in kf.split(X):
        results = []
        loaded_img_train, loaded_img_val = X[train_index], X[val_index]
        loaded_lab_train, loaded_lab_val = Y[train_index], Y[val_index]
        # Display repartition of our classes for the train input.
        train_nb = len(loaded_lab_train)
        repartition = np.array([1, 0, 0, 0, 0, 0, 0])
        for i in range(len(loaded_lab_train)):
            repartition = repartition + loaded_lab_train[i]

        print(repartition/train_nb*100)


        print(loaded_img_train.shape)
        print(loaded_lab_train.shape)

        # Display 10 training images (after mean-image substraction and resize to 227x227)
        # for i in range(0,10):
        #    img = loaded_img_train[i][:][:][:]
        #    plt.imshow(img)
        #    plt.show()


        # Learning params
        learning_rate = 0.003
        batch_size = 40
        training_iters = 4000 # nbr_epochs = (batch_size * training_iters)/nbr_training_images
        display_step = 10 # display training information (loss, training accuracy, ...) every 10 iterations
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
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))+reg_loss

        # regularizer = tf.nn.l2_loss(weights)
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
            load_with_skip('../TD-3/pretrained_alexnet.npy', sess, ['fc8']) # Skip weights from fc8 (fine-tuning)

            print('Start training.')
            step = 1
            total_val_acc = 0
            total_train_acc = 0
            while step < training_iters:
                batch_xs, batch_ys = dataset.next_batch(batch_size, 'train', loaded_img_train, loaded_lab_train)
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})


            #    Display testing status
                if step%test_step == 0:
                    val_count = 0
                    for _ in range(int(dataset.test_size/batch_size)+1): # test accuracy by group of batch_size images
                        batch_vx, batch_vy = dataset.next_batch(batch_size, 'test', loaded_img_val, loaded_lab_val)
                        # print(batch_vx.shape)
                        acc = sess.run(accuracy, feed_dict={x: batch_vx, y: batch_vy, keep_var: 1.})
                        total_val_acc += acc
                        val_count += 1
                    total_val_acc /= val_count
                    print ("{} Iter {}: Validation Accuracy = {:.4f}".format(datetime.now(), step, total_val_acc), file=sys.stderr)

                # Display training status
                if step%display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.}) # Training-accuracy
                    batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.}) # Training-loss
                    print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc),file=sys.stderr)

                step += 1

            i = 0
            while i <= loaded_img_train.shape[0]//batch_size:
                batch_xs, batch_ys = dataset.next_batch(batch_size, 'train', loaded_img_train, loaded_lab_train)
                i += 1
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                total_train_acc += acc
            total_train_acc /= i

            print("Finish!")
            results.append((total_train_acc, total_val_acc))
            print(results)
        break


if __name__ == '__main__':
    main()

