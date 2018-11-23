from todloop.routines import DataLoader
from todloop.tod import TODLoader
from prepare.routines import CleanTOD, PrepareDataLabel

from todloop.base import TODLoop

loop = TODLoop()
loop.add_tod_list("data/2016_ar3_train.txt")
loop.add_routine(TODLoader(output_key="tod_data"))
loop.add_routine(CleanTOD(tod_container="tod_data", output_container="tod_data"))
loop.add_routine(PrepareDataLabel(tod_container="tod_data", downsample=40, pickle_file="/home/lmaurin/cuts/s16/pa3_f90/c10/pa3_f90_s16_c10_v1_results.pickle", output_file="dataset.h5", group='train'))
loop.run(0, 80)

