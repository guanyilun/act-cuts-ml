from todloop.tod import TODLoader
from prepare.routines import CleanTOD
from training.routines import GetTrainingDataLabel, FairSample
from training.simplecnn import TrainCNNModel

from todloop.base import TODLoop


loop = TODLoop()
loop.add_tod_list("data/2016_ar3_train.txt")
loop.add_routine(TODLoader(output_key="tod_data"))
loop.add_routine(CleanTOD(tod_container="tod_data", output_container="tod_data"))
loop.add_routine(GetTrainingDataLabel(tod_container="tod_data", downsample=20, \
                                      data_container="data", label_container="label"))
loop.add_routine(FairSample(data_container="data", label_container="label"))
loop.add_routine(TrainCNNModel(output_file="1026.h5"))
loop.run(0, 80)
