from todloop.base import Routine
import numpy as np

class GetTrainingDataLabel(Routine):
    def __init__(self, tod_container, data_container, label_container, downsample=None):
        self._tod_container = tod_container
        self._downsample = downsample
        self._data_container = data_container
        self._label_container = label_container
        self.labels = None
        
    def initialize(self):
        # load training label
        # TODO: make this an input parameter
        self.labels = np.load("data/train_label.npy")
        
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
        self.get_store().set(self._data_container, data)
        self.get_store().set(self._label_container, label)

        
class FairSample(Routine):
    def __init__(self, data_container, label_container):
        """Sample good and bad detectors with equal probability for training"""
        self._data_container = data_container
        self._label_container = label_container

    def execute(self):
        # retrieve data and label from data store
        data = self.get_store().get(self._data_container)
        label = self.get_store().get(self._label_container)

        # get good and bad detectors
        n_dets = len(label)
        good_dets = np.where(label==1)[0] # i.e. [0,10,13, ...]
        bad_dets = np.where(label==0)[0]

        # if either list is empty, skip the fair sampling procedure
        if len(good_dets)==0 or len(bad_dets)==0:
            return
        
        # define the ratio of sampling, with n labels to start with
        # and with a sampling factor of 2 means that after sampling
        # there will be 2n labels
        sample_factor = 1

        # sample equal number of good and bad detectors for training
        good_dets_sample = np.random.choice(good_dets, sample_factor*n_dets)
        bad_dets_sample  = np.random.choice(bad_dets,  sample_factor*n_dets)

        # merge the two set of indices and shuffle them
        dets_sample = np.hstack([good_dets_sample, bad_dets_sample])
        np.random.shuffle(dets_sample)
        print "dets_sample.shape", dets_sample.shape
        
        # recompose data and label with sampled indices
        new_data = data[dets_sample, :]
        new_label = label[dets_sample]

        self.get_store().set(self._data_container, new_data)
        self.get_store().set(self._label_container, new_label)
