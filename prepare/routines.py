from todloop.base import Routine
import moby2
import numpy as np
from sklearn import preprocessing
import os
import h5py
import cPickle

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
        

class SaveData(Routine):
    def __init__(self, input_key, output_dir):
        self._input_key = input_key
        self._output_dir = output_dir

    def initialize(self):
        if not os.path.exists(self._output_dir):
            print '[INFO] Path %s does not exist, creating ...' % self._output_dir
            os.makedirs(self._output_dir)
            
    def execute(self):
        data = self.get_store().get(self._input_key)
        filepath = os.path.join(self._output_dir, "%d.npy" % self.get_id())
        np.save(filepath, data)


class PrepareDataLabel(Routine):
    def __init__(self, tod_container, downsample=None, pickle_file=None,
                 output_file=None, group=None):
        """Prepare an HDF5 data set that contains all relevant metedata
        and detector timeseries as a basis for various ML studies
        """
        self._tod_container = tod_container
        self._downsample = downsample
        self._pickle_file = pickle_file
        self._output_file = output_file
        self._group_name = group
        self._keys = ['corrLive', 'rmsLive', 'kurtLive', 'DELive', 'MFELive',
                      'skewLive', 'normLive', 'darkRatioLive', 'jumpLive', 'gainLive']
        
    def initialize(self):
        # load pickle file
        with open(self._pickle_file, "r") as f:
            self._pickle_data = cPickle.load(f)

        # create output h5 file
        if os.path.isfile(self._output_file):
            # file exists
            self._hf = h5py.File(self._output_file, 'a')
        else:
            # file doesn't exist
            self._hf = h5py.File(self._output_file, 'w')
            
        try:
            self._group = self._hf.create_group(self._group_name)
        except ValueError:
            self._group = self._hf[self._group_name]
            
        
    def execute(self):
        tod = self.get_store().get(self._tod_container)

        # get relevant metadata for this tod from pickle file
        tod_name = self.get_name()
        pickle_id = self._pickle_data['name'].index(tod_name)

        # get tes mask
        tes_mask = list(np.where(tod.info.array_data['det_type'] == 'tes'))[0]

        # store each det timeseries in hdf5
        for tes_det in tes_mask:
            if self._downsample:
                data = tod.data[tes_det, ::self._downsample]
            else:
                data = tod.data[tes_det]
        
            # generate a unique detector id
            det_uid = '%d.%d' % (self.get_id(), tes_det)
            try:
                dataset = self._group.create_dataset(det_uid, data=data)
            except RuntimeError:
                dataset = self._group[det_uid]
                dataset[:] = data

            # save relevant metadata from pickle file
            for k in self._keys:
                dataset.attrs[k] = self._pickle_data[k][tes_det, pickle_id]
                
            # save label
            dataset.attrs['label'] = int(self._pickle_data['sel'][tes_det, pickle_id])
 
    def finalize(self):
        self._hf.close()
        


