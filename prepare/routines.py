from todloop.base import Routine
import moby2
import numpy as np
from sklearn import preprocessing


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
        self.get_store().set("data", data)
        self.get_store().set("label", label)


