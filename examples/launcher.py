import numpy as np
import os
from astropy.io import fits
from aiida import orm
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import run
from aiida import load_profile, orm, plugins, engine
from aiida.plugins import DataFactory
from photutils.datasets import load_star_image
import tempfile

FitsData = DataFactory("fits.data")

hdu_M67 = load_star_image()
def hdu_to_fitsdata(hdu):
    """
    Convert an in-memory HDUList to FitsData.
    """
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        hdu.writeto(tmp.name, overwrite=True)
        path = os.path.abspath(tmp.name)

    node = FitsData(file=path).store()

    # Optional cleanup of temp file
    os.remove(path)

    return node

# BUILDER
workflow_pipeline = plugins.WorkflowFactory('photometry.pipeline')
image_node = hdu_to_fitsdata(hdu_M67)
 
builder = workflow_pipeline.get_builder()
builder.detection.detection_params = orm.Dict(dict = {"threshold": 3,
                                                "min_separation": 5})
    
#builder.aperture.aperture = orm.Dict(dict= "exact")
#builder.aperture.method = orm.Str("exact")
#builder.aperture.photometry_options = orm.Dict(dict={"method":5}) 
builder.background.method = orm.Str("background_2d")
builder.background.parameters = orm.Dict(dict={})
builder.image = image_node
    
#run
engine.run(builder)
