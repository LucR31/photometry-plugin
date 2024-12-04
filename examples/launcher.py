from aiida import orm, plugins, engine
from photutils.datasets import load_star_image
import numpy as np
from astropy.io import fits

#builder initialize
workflow = plugins.WorkflowFactory('images.reduction')
builder = workflow.get_builder()
builder.directory_input = orm.Str('/home/jovyan/files')
builder.directory_output = orm.Str('/home/jovyan/calibrated')
#builder.image = orm.Str('image/path')
builder.combination_params = orm.Dict({'unit':'adu'})
builder.aggregate_method = orm.Str('average')

#run
engine.run(builder)