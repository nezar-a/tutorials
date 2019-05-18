# AlexNet in keras
# Paper: Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks

#By: Nezar Assawiel, Nov. 2017


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization


model = Sequential()

#1st Conv -> Pool -> Norm
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

#2nd Conv -> Pool -> Norm
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

#3rd Conv -> Norm
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#4th Conv -> Norm
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#5th Conv -> Pool -> Norm
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())


model.add(Flatten())

#1st Dense -> Dropout -> Norm
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

#2nd Dense -> Dropout -> Norm
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

#3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('softmax'))

#fit model and compile as you like!
