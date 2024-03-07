import pathlib
from aiida import orm, plugins, engine
from photutils.datasets import load_star_image
import numpy as np
#load basic image
hdu = load_star_image() 
data = hdu.data[0:401, 0:401]  
#data = np.ones((100, 100))
matrix = orm.ArrayData()
matrix.set_array('data', data)

#builder initialize
workflow = plugins.WorkflowFactory('fill.mat')
builder = workflow.get_builder()

#filling builder
builder.data = matrix
builder.aperture_method = orm.Str('exact') #subpixel, center

#run
engine.run(builder)