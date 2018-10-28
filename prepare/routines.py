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
        



        
        


