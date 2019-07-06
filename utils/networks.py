import keras
print(keras.__version__)
from keras.applications.nasnet import NASNetMobile
from keras.applications.vgg16 import VGG16
from keras.applications import MobileNetV2
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def mlp(num_classes=10, input_shape=(784,)):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=(RMSprop()),
      metrics=[
     'accuracy'])
    return model


def simple_cnn(num_classes=10, input_shape=(32, 32, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
      kernel_initializer='he_normal',
      input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
      activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=(keras.losses.categorical_crossentropy), optimizer='adam',
      metrics=[
     'accuracy'])
    return model


def simple_cnn_dropout(num_classes=10, input_shape=(32, 32, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
      kernel_initializer='he_normal',
      input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
      activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=(keras.losses.categorical_crossentropy), optimizer='adam',
      metrics=[
     'accuracy'])
    return model


def simple_cnn_batchnorm(num_classes=10, input_shape=(32, 32, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
      kernel_initializer='he_normal',
      input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=(keras.losses.categorical_crossentropy), optimizer='adam',
      metrics=[
     'accuracy'])
    return model


def simple_cnn_batchnorm_Dropout(num_classes=10, input_shape=(32, 32, 1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
      kernel_initializer='he_normal',
      input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=(keras.losses.categorical_crossentropy), optimizer='adam',
      metrics=[
     'accuracy'])
    return model


def alexnet_model(img_shape=(32, 32, 1), n_classes=10, l2_reg=0.0):
    alexnet = Sequential()
    alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape, padding='same',
      kernel_regularizer=(l2(l2_reg))))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))
    alexnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return alexnet


def mobilenetV2(num_classes=10, input_shape=(32, 32, 1)):
    mobilenetv2 = MobileNetV2(input_shape=input_shape, alpha=1.0,
      include_top=True,
      weights=None,
      input_tensor=None,
      pooling=None,
      classes=num_classes)
    mobilenetv2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return mobilenetv2


def vgg16(num_classes=10, input_shape=(32, 32, 1)):
    vgg = VGG16(include_top=True, weights=None,
      input_tensor=None,
      input_shape=input_shape,
      pooling=None,
      classes=num_classes)
    vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return vgg


def resnet50(num_classes=10, input_shape=(32, 32, 1)):
    resnet = ResNet50(include_top=True, weights=None,
      input_tensor=None,
      input_shape=input_shape,
      pooling=None,
      classes=num_classes)
    resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return resnet


def inceptionV3(num_classes=10, input_shape=(32, 32, 1)):
    inception = InceptionV3(include_top=True, weights=None,
      input_tensor=None,
      input_shape=input_shape,
      pooling=None,
      classes=num_classes)
    inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return inception


def nasnet(num_classes=10, input_shape=(32, 32, 1)):
    nasnet = NASNetMobile(input_shape=input_shape, include_top=True,
      weights=None,
      input_tensor=None,
      pooling=None,
      classes=num_classes)
    nasnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return nasnet

