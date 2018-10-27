import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalMaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt

from todloop.base import Routine

class TrainCNNModel(Routine):
    def __init__(self, output_file):
        self.model = None
        self._num_classes = 2
        self._output_file = output_file

    def initialize(self):
        # define CNN model

        input_shape = (1, None, 1) # varaible input size
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(1, 10), strides=(1, 2),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Conv2D(64, (1, 10), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))

        model.add(Conv2D(64, (1, 10), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        # model.add(Flatten())  # useful when shape is not (1, None, 1)
        model.add(GlobalMaxPooling2D()) # try to fix the flatten problem with GlobalMaxPooling2D
        model.add(Dropout(0.25))

        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self._num_classes, activation='softmax'))

        # compile model
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        self.model = model

    def execute(self):
        # retrieve data and label
        data = self.get_store().get("data")
        label = self.get_store().get("label")
        data = data.astype('float32')
        data = data.reshape(data.shape[0], 1, data.shape[1], 1)
        print data.shape
        
        label = keras.utils.to_categorical(label, self._num_classes)
        
        epochs = 10
        self.model.fit(data, label,
                       epochs=epochs,
                       verbose=1)
        # self.model.train_on_batch(data, label)

        
    def finalize(self):
        self.model.save(self._output_file)
