# -*- coding: utf-8 -*-
from aiida import orm, plugins, engine
from photutils.datasets import load_star_image
import numpy as np
from astropy.io import fits

FITSDATA = plugins.DataFactory("fits.data")
ArrayData = plugins.DataFactory("core.array")

hdu = load_star_image()  # FITS test image
img_array = hdu.data  # narray image

array = ArrayData()
array.set_array("example", np.array([3]))

# builder initialize
workflow = plugins.WorkflowFactory("aperture.photometry")
builder = workflow.get_builder()
builder.fits_image = FITSDATA("/home/jovyan/calibrated/combined_bias.fit")
builder.bck = array
builder.positions = orm.List([(10, 10)])
builder.radii = orm.List([3])

# run
engine.run(builder)
