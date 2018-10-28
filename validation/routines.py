from todloop.base import Routine
from training.routines import GetTrainingDataLabel

import numpy as np
import keras
from keras.models import load_model


class GetValidationDataLabel(GetTrainingDataLabel):
    def __init__(self, tod_container, data_container, label_container, downsample=None):
        GetTrainingDataLabel.__init__(self, tod_container, data_container, label_container, downsample)
        
    def initialize(self):
        # load testing label
        self.labels = np.load("data/test_label.npy")
        

class ValidateModel(Routine):
    def __init__(self, model, label_container="label", prediction_container="prediction"):
        self._model_path = model
        self.model = None
        self._num_classes=2
        
    def initialize(self):
        # load the provided model
        self.model = load_model(self._model_path)
        self._total = 0
        self._correct = 0

    def execute(self):
        # retrieve data and label
        data = self.get_store().get("data")
        label = self.get_store().get("label")
        
        data = data.astype('float32')
        data = data.reshape(data.shape[0], 1, data.shape[1], 1)
        
        label = keras.utils.to_categorical(label, self._num_classes)
        
        prediction = self.model.predict(data)

        # check the number of accurate guesses
        correct = ((prediction[:,1] > prediction[:,0]) & (label[:,1] > label[:,0])) | ((prediction[:,1] < prediction[:,0]) & (label[:,1] < label[:, 0]))
        
        n_correct = np.count_nonzero(correct)
        n_total = prediction.shape[0]
        self._total += n_total
        self._correct += n_correct
        
        print '[INFO] Number of correct:', n_correct
        print '[INFO] Total:', n_total
        print '[INFO] Accuracy:', float(n_correct*100.0)/prediction.shape[0]

        # output the prediction and label for further processing if any
        self.get_store().set("prediction", prediction)
        self.get_store().set("label", label)
        
    def finalize(self):
        print 'Total label:', self._total
        print 'Total correct:', self._correct
        print 'Total accuracy:', self._correct * 100.0 / self._total


            
        
