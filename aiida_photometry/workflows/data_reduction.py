from aiida.engine import calcfunction
from aiida import orm, engine, plugins
from astropy.nddata import CCDData

import numpy as np
import os
from astropy.io import fits
import ccdproc

FITSDATA = plugins.DataFactory('fits.data')

def get_image_by_type(path_collection:str,
                      image_type:str)->ccdproc.ImageFileCollection:
    # Utility function to return images of a certain image type, i.e., BIAS, DARKS...
    im_collection = ccdproc.ImageFileCollection(path_collection)
    return im_collection.files_filtered(imagetyp=image_type,
                                        include_path=True)

@calcfunction
def sub_overscan(path_collection:orm.Str,
                 image_type:str):
    images = get_image_by_type(path_collection.value, image_type)
    first_bias = CCDData.read(images[0], unit='adu')
    result_substraction = ccdproc.subtract_overscan(first_bias,
                                     overscan=first_bias[:, 2055:], 
                                     median=True)
    return FITSDATA(result_substraction)

@calcfunction
def trim_image(image_to_trim):
    trimmed_image = ccdproc.trim_image(image_to_trim[:, :2048])
    return FITSDATA(trimmed_image)

@calcfunction
def make_bias_master(path_collection:orm.Str,
                     calibrated_path:orm.Str, 
                     comb_method:orm.Str,
                     comb_params_dict:orm.Dict
                     ):
    """
    Returns a FITS file with the combined bias
    """
    biases_im = get_image_by_type(path_collection.value, 'Bias Frame')
    combined_bias = ccdproc.combine(biases_im, 
                            method=comb_method, 
                            **comb_params_dict
                            )
    combined_bias.meta['combined'] = True
    combined_bias.write(calibrated_path.value+'/combined_bias.fit')
    return FITSDATA(calibrated_path.value+'/combined_bias.fit')

class DataReduction(engine.WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('directory_input', valid_type = orm.Str)
        spec.input('directory_output', valid_type = orm.Str)
        spec.input('aggregate_method', valid_type = orm.Str, help = '')
        spec.input('combination_params', valid_type = orm.Dict, help = '')

        spec.outputs.dynamic = True
        spec.outline(
            cls.agg_bias,
            cls.result_cal
        )

    def agg_bias(self):
        self.ctx.bias = make_bias_master(self.inputs.directory_input,
                                         self.inputs.directory_output,
                                         self.inputs.aggregate_method,
                                         self.inputs.comb_params_dict
                                         )
        
    def result_cal(self):
        self.out('result', self.ctx.bias)

    


