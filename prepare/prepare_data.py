from todloop.tod import TODLoader, TODInfoLoader
from routines import CleanTOD, GetDataLabel, TrainCNNModel
from todloop.base import TODLoop


loop = TODLoop()
loop.add_tod_list("../data/2016_ar3_train.txt")
loop.add_routine(TODLoader(output_key="tod_data"))
loop.add_routine(CleanTOD(tod_container="tod_data", output_container="tod_data"))
loop.add_routine(GetDataLabel(tod_container="tod_data", downsample=20))
loop.add_routine(TrainCNNModel())
loop.run(0, 80)
