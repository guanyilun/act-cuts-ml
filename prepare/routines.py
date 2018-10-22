from todloop.base import Routine
import moby2
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalMaxPooling2D
from keras.utils import np_utils

class CleanTOD(Routine):
    def __init__(self, tod_container, output_container):
        self._tod_container = tod_container
        self._output_container = output_container

    def execute(self):
        print '[INFO] Cleaning TOD ...'
        tod = self.get_store().get(self._tod_container)

        # remove mce cuts
        mce_cuts = moby2.tod.get_mce_cuts(tod)
        moby2.tod.fill_cuts(tod, mce_cuts, no_noise=True)

        # clean data
        moby2.tod.remove_mean(tod)
        moby2.tod.detrend_tod(tod)

        # fix optical signs
        optical_signs = tod.info.array_data['optical_sign']
        tod.data = tod.data*optical_signs[:, np.newaxis]
        
        # normalize data
        tod.data = preprocessing.normalize(tod.data, norm='max', copy=False)   

        # export to data store
        self.get_store().set(self._output_container, tod)
        

class GetDataLabel(Routine):
    def __init__(self, tod_container, downsample=None):
        self._tod_container = tod_container
        self._downsample = downsample
        self.labels = None
        
    def initialize(self):
        # load training label
        # TODO: make this an input parameter
        self.labels = np.load("../data/train_label.npy")
        

    def execute(self):
        tod = self.get_store().get(self._tod_container)

        # get label corresponding to this tod
        i = self.get_id()
        label = self.labels[:, i]

        # get data corresponding to this tod
        if self._downsample:
            data = tod.data[:, ::self._downsample]
        else:
            data = tod.data

        # the shapes of our relevant variables are
        # data:  (1056, nsamples/downsample) 
        # label: (1056, 1)
        
        # output the data and label
        self.get_store().set("data", data)
        self.get_store().set("label", label)


class TrainCNNModel(Routine):
    def __init__(self):
        self.model = None
        self._num_classes = 2

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
        self.model.save('1021_model.h5')
