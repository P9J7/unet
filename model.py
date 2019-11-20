from functools import reduce
from operator import mul
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.initializers import Initializer
from keras import optimizers


# glorot_normal
class myInit(Initializer):
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, shape=(3, 3), dtype=None):
        return K.random_normal(shape, mean=0.0, stddev=np.sqrt(2 / reduce(mul, shape, 1) * self.channels), dtype=dtype)


class myInit2(Initializer):
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, shape=(2, 2), dtype=None):
        return K.random_normal(shape, mean=0.0, stddev=np.sqrt(2 / reduce(mul, shape, 1) * self.channels), dtype=dtype)


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def rawUnet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    my_init = myInit(1)
    conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer=my_init)(inputs)
    my_init = myInit(64)
    conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        pool1)
    my_init = myInit(128)
    conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        pool2)
    my_init = myInit(256)
    conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        pool3)
    my_init = myInit(512)
    conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        drop4)
    my_init = myInit(1024)
    conv5 = Conv2D(1024, 3, activation='relu', padding='valid',
                   kernel_initializer=my_init)(conv5)
    # drop5 = Dropout(0.5)(conv5)
    my_init2 = myInit2(1024)
    up6 = Conv2D(512, 2, activation='relu', padding='valid', kernel_initializer=my_init2)(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5))
    merge6 = concatenate([Cropping2D(((4, 4), (4, 4)))(conv4), up6], axis=3)
    my_init = myInit(512)
    conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv6)
    my_init2 = myInit2(512)
    up7 = Conv2D(256, 2, activation='relu', padding='valid', kernel_initializer=my_init2)(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6))
    merge7 = concatenate([Cropping2D((16, 16), (16, 16))(conv3), up7], axis=3)
    my_init = myInit(256)
    conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv7)
    my_init2 = myInit2(256)
    up8 = Conv2D(128, 2, activation='relu', padding='valid', kernel_initializer=my_init2())(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7))
    merge8 = concatenate([Cropping2D((40, 40), (40, 40))(conv2), up8], axis=3)
    my_init = myInit(128)
    conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv8)
    my_init2 = myInit2(128)
    up9 = Conv2D(64, 2, activation='relu', padding='valid', kernel_initializer=my_init())(
        UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8))
    merge9 = concatenate([Cropping2D(((88, 88), (88, 88)))(conv1), up9], axis=3)
    my_init = myInit(64)
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='valid', kernel_initializer=my_init)(
        conv9)
    # ????
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
