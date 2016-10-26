from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def make_sets(nval=1000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x_val = x_train[-nval:]
    y_val = y_train[-nval:]
    x_train = x_train[:-nval]
    y_train = y_train[:-nval]
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_val = np_utils.to_categorical(y_val, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, Y_train), (x_test, Y_test), (x_val, Y_val)

    
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD

def reduce_dim(out_dim, encoder_activation='sigmoid'):
    input_img = Input(shape=(784,))
    encoded = Dense(out_dim, activation='sigmoid')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder     = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(out_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    hist = autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=25, shuffle=True, validation_data=(x_val, x_val), verbose=1)
    autoencoder.save('basic_autoencoder_768x' + str(out_dim) + '.h5')
    encoder.save('basic_encoder_768x' + str(out_dim) + '.h5')
    decoder.save('basic_decoder_768x' + str(out_dim) + '.h5')
    err = hist.history['val_loss'][-1]
    return err
(x_train, Y_train), (x_test, Y_test), (x_val, Y_val) = make_sets(1000)

dims = [2, 8, 32, 64]
err = np.empty((4, 1))
for i in range(0, 4):
    err[i] = reduce_dim(dims[i])

err_relu = np.empty((4, 1))
for i in range(0, 4):
    err_relu[i] = reduce_dim(dims[i],'relu')

print('Using sigmoid activation for the encoder:')
for i in range(0, 4):
    cr = 1 - dims[i]/784
    print('Compression to {} dimensions; Compression rate: {}; Binary Crossentropy: {}'.format(dims[i],cr,err[i]))
print('Using ReLu activation for the encoder:')
for i in range(0, 4):
    cr = 1 - dims[i]/784
    print('Compression to {} dimensions; Compression rate: {}; Binary Crossentropy: {}'.format(dims[i],cr,err_relu[i]))








