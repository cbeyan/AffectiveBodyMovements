# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:13:24 2020

@author: Beyan et al.
@functionality: To import images from dataset and create training/test set with mini batches
Then to use a custom made CNN to classify images into emotional states

"""

'''-------------------------------------------------------------------------'''
'''### Libraries import ###'''
import math
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()



'''### Functions ###'''
def import_images():
    # Imports images as numpy array from source location
    images_c1 = []
    images_c2 = []
    images_c3 = []
    images_c4 = []
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0

    for i in glob.glob('Original/Branch1/Angry/*.png'): # put all original Branch 1 images having emotion class Angry under this folder

        c1 = c1 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c1.append(temp)

    for i in glob.glob('Original/Branch1/Happy/*.png'): # put all original Branch 1  images having emotion class Happy under this folder

        c2 = c2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c2.append(temp)

    for i in glob.glob('Original/Branch1/Insecure/*.png'): # put all original Branch 1  images having emotion class Insecure under this folder

        c3 = c3 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c3.append(temp)

    for i in glob.glob('Original/Branch1/Sad/*.png'): # put all original Branch 1  images having emotion class Sad under this folder

        c4 = c4 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c4.append(temp)

    images_c1 = np.array(images_c1)
    images_c2 = np.array(images_c2)
    images_c3 = np.array(images_c3)
    images_c4 = np.array(images_c4)
    return images_c1, images_c2, images_c3, images_c4, c1, c2, c3, c4

def import_images_augmentation():
    # Imports images as numpy array from source location
    c1_2 = 0
    c2_2 = 0
    c3_2 = 0
    c4_2 = 0
    images_c1_2 = []
    images_c2_2 = []
    images_c3_2 = []
    images_c4_2=[]

    for i in glob.glob('Augmented/Branch1/Angry/*.png'): # put all augmented Branch 1 images having emotion class Angry under this folder
        c1_2 = c1_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c1_2.append(temp)

    for i in glob.glob('Augmented/Branch1/Happy/*.png'): # put all augmented Branch 1 images having emotion class Happy under this folder
        c2_2 = c2_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c2_2.append(temp)

    for i in glob.glob('Augmented/Branch1/Insecure/*.png'): # put all augmented Branch 1 images having emotion class Insecure under this folder
        c3_2 = c3_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c3_2.append(temp)

    for i in glob.glob('Augmented/Branch1/Sad/*.png'): # put all augmented Branch 1 images having emotion class Sad under this folder
        c4_2 = c4_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c4_2.append(temp)

    images_c1_2 = np.array(images_c1_2)
    images_c2_2 = np.array(images_c2_2)
    images_c3_2 = np.array(images_c3_2)
    images_c4_2 = np.array(images_c4_2)
    return images_c1_2, images_c2_2, images_c3_2, images_c4_2, c1_2, c2_2, c3_2, c4_2

def import_images_branch2():
    # Imports images as numpy array from source location

    images_c1 = []
    images_c2 = []
    images_c3 = []
    images_c4 = []
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0

    for i in glob.glob('Original/Branch2/Angry/*.png'): # put all original Branch 2 images having emotion class Angry under this folder
        c1 = c1 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c1.append(temp)

    for i in glob.glob('Original/Branch2/Happy/*.png'): # put all original Branch 2 images having emotion class Happy under this folder
        c2 = c2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c2.append(temp)

    for i in glob.glob('Original/Branch2/Insecure/*.png'): # put all original Branch 2 images having emotion class Insecure under this folder
        c3 = c3 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c3.append(temp)

    for i in glob.glob('Original/Branch2/Sad/*.png'): # put all original Branch 2 images having emotion class Sad under this folder
        c4 = c4 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c4.append(temp)

    images_c1 = np.array(images_c1)
    images_c2 = np.array(images_c2)
    images_c3 = np.array(images_c3)
    images_c4 = np.array(images_c4)
    return images_c1, images_c2, images_c3, images_c4

def import_images_augmentation_branch2():
    # Imports images as numpy array from source location
    c1_2 = 0
    c2_2 = 0
    c3_2 = 0
    c4_2 = 0
    images_c1_2 = []
    images_c2_2 = []
    images_c3_2 = []
    images_c4_2=[]

    for i in glob.glob('Augmented/Branch2/Angry/*.png'): # put all augmented Branch 2 images having emotion class Angry under this folder
        c1_2 = c1_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c1_2.append(temp)

    for i in glob.glob('Augmented/Branch2/Happy/*.png'): # put all augmented Branch 2 images having emotion class Happy under this folder
        c2_2 = c2_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c2_2.append(temp)

    for i in glob.glob('Augmented/Branch2/Insecure/*.png'): # put all augmented Branch 2 images having emotion class Insecure under this folder
        c3_2 = c3_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c3_2.append(temp)

    for i in glob.glob('Augmented/Branch2/Sad/*.png'): # put all augmented Branch 2 images having emotion class Sad under this folder
        #print(i)
        c4_2 = c4_2 + 1
        temp = plt.imread(i)
        temp = np.asarray(temp)
        images_c4_2.append(temp)

    images_c1_2 = np.array(images_c1_2)
    images_c2_2 = np.array(images_c2_2)
    images_c3_2 = np.array(images_c3_2)
    images_c4_2 = np.array(images_c4_2)
    return images_c1_2, images_c2_2, images_c3_2, images_c4_2



def create_labels(c1, c2, c3, c4):
    # Manually create labels according to number of images
    array1 = np.zeros((1, c1))
    array2 = np.ones((1, c2)) * 1
    labels = np.concatenate((array1, array2), axis=1)

    array3 = np.ones((1, c3)) * 2
    labels = np.concatenate((labels, array3), axis=1)

    array4 = np.ones((1, c4)) * 3
    labels = np.concatenate((labels, array4), axis=1)

    labels = np.array(labels)

    labels_c1 = np.array(array1)
    labels_c2 = np.array(array2)
    labels_c3 = np.array(array3)
    labels_c4 = np.array(array4)

    return labels, labels_c1, labels_c2, labels_c3, labels_c4


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    #Creates the placeholders for the tensorflow session.

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    
#    Coarse = tf.placeholder(tf.float32, [None, 30, n_W0, n_C0])
    Fine = tf.placeholder(tf.float32, [None, 30, n_W0, n_C0])
    
    return X, Y, Fine

def initialize_parameters():
    #Initialize convolution kernals to be used
    
    W1 = tf.get_variable("W1", [3, 5, 3, 16], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [3, 5, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [3, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3}
    return parameters

def forward_propogation(X, Fine, parameters):
    #Forward prop of the CNN
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    #Coarse part
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')    
    #Second layer
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    #Third layer
    Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')
    
    X1 = tf.contrib.layers.flatten(P3)
    #D1 = tf.layers.dropout(inputs = X1, rate = 0.4)
    X11 = tf.contrib.layers.fully_connected(X1,4,activation_fn=None)
    #dropour1 = tf.layers.dropout(inputs = X11, rate = 0.4)
    
    #Fine part
    Z1 = tf.nn.conv2d(Fine, W1, strides=[1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')    
    #Second layer
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    #Third layer
    Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')
    
    X2 = tf.contrib.layers.flatten(P3)
    #D2 = tf.layers.dropout(inputs = X2, rate = 0.4)
    X22 = tf.contrib.layers.fully_connected(X2,4, activation_fn = None)
    #dropour2 = tf.layers.dropout(inputs = X22, rate = 0.4)
    
    #Fully connected layer
    #X = tf.concat([X1,X2],1)
    X3 = X11 + X22
    #Z3 = tf.contrib.layers.fully_connected(X, 150, activation_fn=tf.nn.relu)
    #dropour = tf.layers.dropout(inputs = Z3, rate = 0.4)
    #Z4 = tf.contrib.layers.fully_connected(X, 4, activation_fn=None)
    # = tf.layers.dropout(inputs = Z4, rate = 0.4)
    
    return X3

def compute_cost(Z4, Y, parameters, lambd = 0.1):
    #For backprop, cost calculation
    
    #regular_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z4, labels=Y))
    regular_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z4, labels=Y))
    
    cost = regular_cost
    return cost

def random_mini_batches(X, Y, X2, mini_batch_size = 64):
    #Creates a list of random minibatches from (X, Y)
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []

    # Step 1: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size:(k + 1) * mini_batch_size,:]
        mini_batch_Y = Y[k * mini_batch_size:(k + 1) * mini_batch_size,:]
        mini_batch_X2 = X2[k * mini_batch_size:(k + 1) * mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_X2)
        mini_batches.append(mini_batch)
    
    # Step 2: Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = X[num_complete_minibatches * mini_batch_size:num_complete_minibatches * mini_batch_size + end,:]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size:num_complete_minibatches * mini_batch_size + end,:]
        mini_batch_X2 = X2[num_complete_minibatches * mini_batch_size:num_complete_minibatches * mini_batch_size + end,:]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_X2)
        mini_batches.append(mini_batch)
    
    return mini_batches

def calc_Accuracy(accuracy, X,Y, Fine, x_train, y_train, x2_train, batch_size = 64):
    total_acc = 0
    m = x_train.shape[0]
    j = 0
    k = 0
    while j<= m:
        k += 1
        acc = accuracy.eval({X: x_train[j:j+63], Y: y_train[j:j+63],Fine:x2_train[j:j+63]})
        total_acc += acc
        j += 64
    
    total_acc = total_acc/k
        
    return total_acc
        

#def model(x_train, y_train,x2_train,  x_test, y_test, x2_test, x_f, y_f,x2_f, learning_rate, save_count, num_epochs = 110, minibatch_size = 64, print_cost=True):
def model(x_train, y_train, x2_train, x_test, y_test, x2_test, learning_rate, save_count,
              num_epochs=110, minibatch_size=64, print_cost=True):
    #tr, te, tF, f1_V, parameters = model(x_train, y_train, x2_train, x_test, y_test, x2_test, learn, j)
    #Final CNN model
    
    #Obtaining variable size
    ops.reset_default_graph()
    (m,n_H0,n_W0, n_C0) = x_train.shape
    n_y = y_train.shape[1]
    costs = []
    
    #Forward, back prop declarations 
    X,Y,Fine = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z4 = forward_propogation(X,Fine,parameters)
    cost = compute_cost(Z4, Y, parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    #Finally running tensorflow session
    #with tf.device("/cpu:0"):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) 
            minibatches = random_mini_batches(x_train, y_train, x2_train, minibatch_size)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y, minibatch_X2) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y,Fine:minibatch_X2})
                minibatch_cost += temp_cost / num_minibatches
                
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                
        # plot the cost
#        plt.plot(np.squeeze(costs))
#        plt.ylabel('cost')
#        plt.xlabel('iterations (per tens)')
#        plt.title("Learning rate =" + str(learning_rate))
#        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z4, 1)
        
        #acct = tf.metrics.accuracy(labels=tf.argmax(Y,1), predictions = predict_op)
        #print("Accuracy could be: ", acct)        
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        
        train_accuracy = calc_Accuracy(accuracy, X, Y, Fine, x_train, y_train, x2_train)
        ####accuracy.eval({X: x_train, Y: y_train})
        test_accuracy = accuracy.eval({X: x_test, Y: y_test, Fine:x2_test})
        #final_accuracy = accuracy.eval({X:x_f, Y:y_f, Fine:x2_f})

        # Calculation of F1 score
        # For validation set
        y_pred = predict_op.eval(feed_dict = {X:x_test, Y:y_test, Fine: x2_test})
        y_true = np.argmax(y_test, 1)
        val_f1 = f1_score(y_true, y_pred, average = 'weighted')
        val_confusion_matrix = confusion_matrix(y_true, y_pred)
        randd = tf.multiply(val_f1,1,name="op_to_re")

        
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        #print("Test Accuracy:", final_accuracy)
        print("F1 score test:" , val_f1)
        print("F1 score test:", val_confusion_matrix)
        #print("Confusion Matrix:", cm1)

        saver.save(sess, ",/Model", global_step=save_count)
                
        #return train_accuracy, test_accuracy,final_accuracy, val_f1, test_f1, parameters
        return train_accuracy, test_accuracy, val_f1, val_confusion_matrix, parameters
    

'''-------------------------------------------------------------------------'''
'''### Data import, one hot encoding ### '''
images_c1, images_c2, images_c3, images_c4, c1, c2, c3, c4= import_images() #original branch 1
images_c1_2, images_c2_2, images_c3_2, images_c4_2, c1_2, c2_2, c3_2, c4_2= import_images_augmentation() #augmentation branch 1
labels, labels_c1, labels_c2, labels_c3, labels_c4=create_labels(c1, c2, c3, c4) #original branch 1
labels_2, labels_c1_2, labels_c2_2, labels_c3_2, labels_c4_2=create_labels(c1_2, c2_2, c3_2, c4_2) #augmentation branch 1
images_c1_branch2, images_c2_branch2, images_c3_branch2, images_c4_branch2= import_images_branch2() #original branch 2
images_c1_2_branch2, images_c2_2_branch2, images_c3_2_branch2, images_c4_2_branch2= import_images_augmentation_branch2() #augmentation branch 2

#One hot encoding scheme Branch 1 Original and augmented
hot_labels = np.zeros((labels.size,4))
klabels = np.int64(labels)
hot_labels[np.arange(labels.size), klabels] = 1
hot_labels_2 = np.zeros((labels_2.size,4))
klabels_2 = np.int64(labels_2)
hot_labels_2[np.arange(labels_2.size), klabels_2] = 1

hot_labels_c1 = np.zeros((labels_c1.size,4))
klabels_c1 = np.int64(labels_c1)
hot_labels_c1[np.arange(labels_c1.size), klabels_c1] = 1
hot_labels_c1_2 = np.zeros((labels_c1_2.size,4))
klabels_c1_2 = np.int64(labels_c1_2)
hot_labels_c1_2[np.arange(labels_c1_2.size), klabels_c1_2] = 1

hot_labels_c2 = np.zeros((labels_c2.size,4))
klabels_c2 = np.int64(labels_c2)
hot_labels_c2[np.arange(labels_c2.size), klabels_c2] = 1
hot_labels_c2_2 = np.zeros((labels_c2_2.size,4))
klabels_c2_2 = np.int64(labels_c2_2)
hot_labels_c2_2[np.arange(labels_c2_2.size), klabels_c2_2] = 1

hot_labels_c3 = np.zeros((labels_c3.size,4))
klabels_c3 = np.int64(labels_c3)
hot_labels_c3[np.arange(labels_c3.size), klabels_c3] = 1
hot_labels_c3_2 = np.zeros((labels_c3_2.size,4))
klabels_c3_2 = np.int64(labels_c3_2)
hot_labels_c3_2[np.arange(labels_c3_2.size), klabels_c3_2] = 1

hot_labels_c4 = np.zeros((labels_c4.size,4))
klabels_c4 = np.int64(labels_c4)
hot_labels_c4[np.arange(labels_c4.size), klabels_c4] = 1
hot_labels_c4_2 = np.zeros((labels_c4_2.size,4))
klabels_c4_2 = np.int64(labels_c4_2)
hot_labels_c4_2[np.arange(labels_c4_2.size), klabels_c4_2] = 1

#Now to shuffle and split training, validation , test set
#Take average of 5 readings
for k in range(2,3):
    learn = 0.00045*k
    Tr = 0
    Te = 0
    Tf = 0
    f1_val= 0
    f1_test = 0
    curr_val_confusion_matrix = np.zeros((4, 4))
    
    for j in range(0, 5):
        #how many images we have in total which will be divided into two as train and validation per class=1 to 4.
        len_images_c1 = len(images_c1)
        len_images_c2 = len(images_c2)
        len_images_c3 = len(images_c3)
        len_images_c4 = len(images_c4)
        print("The total amount of images the class 1: " + str(images_c1.shape[0]))
        print("The total amount of images the class 2: " + str(images_c2.shape[0]))
        print("The total amount of images the class 3: " + str(images_c3.shape[0]))
        print("The total amount of images the class 4: " + str(images_c4.shape[0]))

        #shuffle the data
        idx_c1 = np.random.permutation(len_images_c1)
        idx_c2 = np.random.permutation(len_images_c2)
        idx_c3 = np.random.permutation(len_images_c3)
        idx_c4 = np.random.permutation(len_images_c4)
        # the shuffling will be the same for all training data (train+validation) and also the corresponding augmentation data
        x_shuffled_c1, y_shuffled_c1, x_shuffled_c1_2, y_shuffled_c1_2, x_branch2_shuffled_c1,x_branch2_shuffled_c1_2 = images_c1[idx_c1], hot_labels_c1[idx_c1], images_c1_2[idx_c1], hot_labels_c1_2[idx_c1],images_c1_branch2[idx_c1],images_c1_2_branch2[idx_c1]
        x_shuffled_c2, y_shuffled_c2, x_shuffled_c2_2, y_shuffled_c2_2, x_branch2_shuffled_c2,x_branch2_shuffled_c2_2 = images_c2[idx_c2], hot_labels_c2[idx_c2], images_c2_2[idx_c2], hot_labels_c2_2[idx_c2],images_c2_branch2[idx_c2],images_c2_2_branch2[idx_c2]
        x_shuffled_c3, y_shuffled_c3, x_shuffled_c3_2, y_shuffled_c3_2, x_branch2_shuffled_c3,x_branch2_shuffled_c3_2 = images_c3[idx_c3], hot_labels_c3[idx_c3], images_c3_2[idx_c3], hot_labels_c3_2[idx_c3],images_c3_branch2[idx_c3],images_c3_2_branch2[idx_c3]
        x_shuffled_c4, y_shuffled_c4, x_shuffled_c4_2, y_shuffled_c4_2, x_branch2_shuffled_c4,x_branch2_shuffled_c4_2 = images_c4[idx_c4], hot_labels_c4[idx_c4], images_c4_2[idx_c4], hot_labels_c4_2[idx_c4],images_c4_branch2[idx_c4],images_c4_2_branch2[idx_c4]

        # take the 70% of the data as training set
        split_n_c1 = int(round(0.70 * len_images_c1))
        split_n_c2 = int(round(0.70 * len_images_c2))
        split_n_c3 = int(round(0.70 * len_images_c3))
        split_n_c4 = int(round(0.70 * len_images_c4))

        x_train_c1 = x_shuffled_c1[0:split_n_c1]
        x_train_c2 = x_shuffled_c2[0:split_n_c2]
        x_train_c3 = x_shuffled_c3[0:split_n_c3]
        x_train_c4 = x_shuffled_c4[0:split_n_c4]
        y_train_c1 = y_shuffled_c1[0:split_n_c1]
        y_train_c2 = y_shuffled_c2[0:split_n_c2]
        y_train_c3 = y_shuffled_c3[0:split_n_c3]
        y_train_c4 = y_shuffled_c4[0:split_n_c4]
        x_train_c1_2 = x_shuffled_c1_2[0:split_n_c1]
        x_train_c2_2 = x_shuffled_c2_2[0:split_n_c2]
        x_train_c3_2 = x_shuffled_c3_2[0:split_n_c3]
        x_train_c4_2 = x_shuffled_c4_2[0:split_n_c4]
        y_train_c1_2 = y_shuffled_c1_2[0:split_n_c1]
        y_train_c2_2 = y_shuffled_c2_2[0:split_n_c2]
        y_train_c3_2= y_shuffled_c3_2[0:split_n_c3]
        y_train_c4_2= y_shuffled_c4_2[0:split_n_c4]

        x_train_c1_branch2 = x_branch2_shuffled_c1[0:split_n_c1]
        x_train_c2_branch2 = x_branch2_shuffled_c2[0:split_n_c2]
        x_train_c3_branch2 = x_branch2_shuffled_c3[0:split_n_c3]
        x_train_c4_branch2 = x_branch2_shuffled_c4[0:split_n_c4]
        x_train_c1_2_branch2 = x_branch2_shuffled_c1_2[0:split_n_c1]
        x_train_c2_2_branch2 = x_branch2_shuffled_c2_2[0:split_n_c2]
        x_train_c3_2_branch2 = x_branch2_shuffled_c3_2[0:split_n_c3]
        x_train_c4_2_branch2 = x_branch2_shuffled_c4_2[0:split_n_c4]

        train_size_before_aug_c1=x_train_c1.shape[0]
        train_size_before_aug_c2=x_train_c2.shape[0]
        train_size_before_aug_c3=x_train_c3.shape[0]
        train_size_before_aug_c4=x_train_c4.shape[0]
        train_size_before_aug_c1_b2=x_train_c1_branch2.shape[0]
        train_size_before_aug_c2_b2=x_train_c2_branch2.shape[0]
        train_size_before_aug_c3_b2=x_train_c3_branch2.shape[0]
        train_size_before_aug_c4_b2=x_train_c4_branch2.shape[0]

        print("Size training BEFORE AUG. class 1 branch 1 and 2: " + str(train_size_before_aug_c1) + " " + str(train_size_before_aug_c1_b2))
        print("Size training BEFORE AUG. class 2 branch 1 and 2: " + str(train_size_before_aug_c2) + " " + str(train_size_before_aug_c2_b2))
        print("Size training BEFORE AUG. class 3 branch 1 and 2: " + str(train_size_before_aug_c3) + " " + str(train_size_before_aug_c3_b2))
        print("Size training BEFORE AUG. class 4 branch 1 and 2: " + str(train_size_before_aug_c4) + " " + str(train_size_before_aug_c4_b2))

        size_majorityClass_BeforeAug=max(train_size_before_aug_c1,train_size_before_aug_c2,train_size_before_aug_c3,train_size_before_aug_c4)
        #Branch 1
        x_train_c1 = np.concatenate((x_train_c1, x_train_c1_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c1+1)]))
        y_train_c1 = np.concatenate((y_train_c1, y_train_c1_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c1 + 1)]))
        x_train_c2 = np.concatenate((x_train_c2, x_train_c2_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c2+1)]))
        y_train_c2 = np.concatenate((y_train_c2, y_train_c2_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c2 + 1)]))
        x_train_c3 = np.concatenate((x_train_c3, x_train_c3_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c3+1)]))
        y_train_c3 = np.concatenate((y_train_c3, y_train_c3_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c3 + 1)]))
        x_train_c4 = np.concatenate((x_train_c4, x_train_c4_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c4+1)]))
        y_train_c4 = np.concatenate((y_train_c4, y_train_c4_2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c4+1)]))
        #Branch 2
        x_train_c1_branch2 = np.concatenate((x_train_c1_branch2, x_train_c1_2_branch2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c1+1)]))
        x_train_c2_branch2 = np.concatenate((x_train_c2_branch2, x_train_c2_2_branch2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c2+1)]))
        x_train_c3_branch2 = np.concatenate((x_train_c3_branch2, x_train_c3_2_branch2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c3+1)]))
        x_train_c4_branch2 = np.concatenate((x_train_c4_branch2, x_train_c4_2_branch2[1:(size_majorityClass_BeforeAug - train_size_before_aug_c4+1)]))

        x_train = np.concatenate((x_train_c1, x_train_c2, x_train_c3, x_train_c4))
        y_train = np.concatenate((y_train_c1, y_train_c2, y_train_c3, y_train_c4))
        x_train_branch2 = np.concatenate((x_train_c1_branch2, x_train_c2_branch2, x_train_c3_branch2, x_train_c4_branch2))
        idx_train = np.random.permutation(x_train.shape[0])
        x_train, y_train,x_train_branch2 = x_train[idx_train], y_train[idx_train],x_train_branch2[idx_train]

        #test set is the rest of the data i.e. 30% of the remaining data
        x_test_c1 = x_shuffled_c1[split_n_c1 + 1:len_images_c1]
        x_test_c2 = x_shuffled_c2[split_n_c2 + 1:len_images_c2]
        x_test_c3 = x_shuffled_c3[split_n_c3 + 1:len_images_c3]
        x_test_c4 = x_shuffled_c4[split_n_c4 + 1:len_images_c4]
        y_test_c1 = y_shuffled_c1[split_n_c1 + 1:len_images_c1]
        y_test_c2 = y_shuffled_c2[split_n_c2 + 1:len_images_c2]
        y_test_c3 = y_shuffled_c3[split_n_c3 + 1:len_images_c3]
        y_test_c4 = y_shuffled_c4[split_n_c4 + 1:len_images_c4]
        x_test_c1_branch2 = x_branch2_shuffled_c1[split_n_c1 + 1:len_images_c1]
        x_test_c2_branch2 = x_branch2_shuffled_c2[split_n_c2 + 1:len_images_c2]
        x_test_c3_branch2 = x_branch2_shuffled_c3[split_n_c3 + 1:len_images_c3]
        x_test_c4_branch2 = x_branch2_shuffled_c4[split_n_c4 + 1:len_images_c4]

        x_test = np.concatenate((x_test_c1, x_test_c2, x_test_c3, x_test_c4))
        x_test_branch2 = np.concatenate((x_test_c1_branch2, x_test_c2_branch2, x_test_c3_branch2, x_test_c4_branch2))
        y_test = np.concatenate((y_test_c1, y_test_c2, y_test_c3, y_test_c4))

        print("Size of training AFTER AUG. in class 1 in branch 1 and 2: " + str(x_train_c1.shape) + " " + str(x_train_c1_branch2.shape))
        print("Size of training AFTER AUG. in class 2 in branch 1 and 2: " + str(x_train_c2.shape) + " " + str(x_train_c2_branch2.shape))
        print("Size of training AFTER AUG. in class 3 in branch 1 and 2: " + str(x_train_c3.shape) + " " + str(x_train_c3_branch2.shape))
        print("Size of training AFTER AUG. in class 4 in branch 1 and 2: " + str(x_train_c4.shape) + " " + str(x_train_c4_branch2.shape))

        print("Size of test class 1 branch 1 and 2: " + str(x_test_c1.shape) + " " + str(x_test_c1_branch2.shape) )
        print("Size of test class 1 branch 1 and 2: " + str(x_test_c2.shape) + " " + str(x_test_c2_branch2.shape) )
        print("Size of test class 1 branch 1 and 2: " + str(x_test_c3.shape) + " " + str(x_test_c3_branch2.shape) )
        print("Size of test class 1 branch 1 and 2: " + str(x_test_c4.shape) + " " + str(x_test_c4_branch2.shape) )

        print("training set size branch 1 and 2: " + str(x_train.shape) + " " + str(x_train_branch2.shape))
        print("test set size branch 1 and 2: " + str(x_test.shape) + " " + str(x_test_branch2.shape))

        tr, te, f1_V, val_confusion_matrix, parameters = model(x_train, y_train, x_train_branch2, x_test, y_test, x_test_branch2, learn, j)

        Tr = Tr+tr
        Te = Te+te
        #Tf = Tf+tF
        f1_val = f1_val+f1_V
        #f1_test = f1_test +f1_T
        curr_val_confusion_matrix = curr_val_confusion_matrix + val_confusion_matrix
        
        
    Tr = Tr/5
    Te = Te/5
    #Tf = Tf/5
    f1_val = f1_val/5
    #f1_test = f1_test/5
    curr_val_confusion_matrix = curr_val_confusion_matrix / 5
    
    print("AVG training accuracy for " + str(learn) +" is " +str(Tr))
    print("AVG test accuracy for " + str(learn) +" is " +str(Te))
    #print("test average accuracy for " + str(learn) +" is " +str(Tf))
    print("AVG weighted F1 score for test " + str(learn) +" is " +str(f1_val))
    #print("F1 score for test  " + str(learn) +" is " +str(f1_test))
    print("AVG Confusion matrix for test  " + str(learn) + " is " + str(curr_val_confusion_matrix))
