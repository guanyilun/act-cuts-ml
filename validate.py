import sys
if len(sys.argv)<=1:
    print "Usage: python validate.py your_model.h5"
    sys.exit(0)

from todloop.base import TODLoop
from todloop.tod import TODLoader
from prepare.routines import CleanTOD, SaveData
from validation.routines import GetValidationDataLabel, ValidateModel

model = sys.argv[1]
print "Model:", model
loop = TODLoop()
loop.add_tod_list("data/2016_ar3_test.txt")
loop.add_routine(TODLoader(output_key="tod_data"))
loop.add_routine(CleanTOD(tod_container="tod_data", output_container="tod_data"))
loop.add_routine(GetValidationDataLabel(tod_container="tod_data", downsample=20, \
                                        data_container="data", label_container="label"))
# loop.add_routine(SaveData(input_key="data", output_dir="outputs/test_20xds_npy/"))
loop.add_routine(ValidateModel(model=model))
loop.run(0, 20)

