from todloop.base import TODLoop
from todloop.tod import TODLoader
from prepare.routines import CleanTOD
from validation.routines import GetValidationDataLabel, ValidateModel

loop = TODLoop()
loop.add_tod_list("data/2016_ar3_test.txt")
loop.add_routine(TODLoader(output_key="tod_data"))
loop.add_routine(CleanTOD(tod_container="tod_data", output_container="tod_data"))
loop.add_routine(GetValidationDataLabel(tod_container="tod_data", downsample=20, \
                                        data_container="data", label_container="label"))
loop.add_routine(ValidateModel(model="saved_models/1029_globalfairsample.h5"))
loop.run(0, 20)

