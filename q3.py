import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random

def make_sets():
    train_data = sio.loadmat('train_32x32.mat')
    test_data = sio.loadmat('test_32x32.mat')
    X_train = train_data['X'].T
    y_train = train_data['y'] - 1
    X_test = test_data['X'].T
    y_test = test_data['y'] - 1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    return (X_train, y_train), (X_test, y_test)
(X_train, y_train), (X_test, y_test) = make_sets()
n_classes = len(np.unique(y_train))

img_dim = X_train.shape[-2:]
classes, counts = np.unique(y_train,return_counts=True)
n_classes = len(classes)
print('\nDimension of the images: {}x{}'.format(img_dim[0],img_dim[1]))
print('Number of different classes: {}'.format(n_classes))

def get_info(set_X,set_y, n=5):
    N = set_X.shape[0]
    classes, counts = np.unique(set_y,return_counts=True)
    for i in range(len(classes)):
        print('{} examples of class {}'.format(counts[i],classes[i]))
    rand_photos =  random.sample(range(N), n)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        phot = rand_photos[i]
        plt.imshow(set_X[phot,:,:,:].T)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    return
#print('\nTraining set:')
#get_info(X_train,y_train)
#print('\nTesting set:')
#get_info(X_test,y_test)


from keras.utils import np_utils
X_train /= 255
print(X_train.shape)
X_test /= 255
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,AveragePooling2D
model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu',input_shape=(3,32,32)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(512, 7, 7, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.summary()

from keras.optimizers import SGD, Adadelta, Adagrad
model.compile(loss='binary_crossentropy', optimizer=Adagrad(), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=1280, nb_epoch=12, verbose=1, validation_data=(X_test, Y_test))





