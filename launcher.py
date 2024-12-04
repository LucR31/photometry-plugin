import pathlib
from aiida import orm, plugins, engine
from photutils.datasets import load_star_image
import numpy as np
from astropy.io import fits

#load basic image
#hdu = load_star_image() 
##data = hdu.data[0:401, 0:401]  
#data = np.ones((100, 100))
#matrix = orm.ArrayData()
#matrix.set_array('data', data)
#positions = [(30.0, 30.0), (40.0, 40.0)]
#rad = [3,5,6]

#builder initialize
workflow = plugins.WorkflowFactory('images.reduction')
builder = workflow.get_builder()

#filling builder
builder.directory_input = orm.Str('/home/jovyan/files')
builder.directory_output = orm.Str('/home/jovyan/calibrated')
#builder.image = orm.Str('image/path')
builder.aggregate_method = orm.Str('average')
#run
engine.run(builder)