from aiida.engine import calcfunction
from aiida.orm import Int, ArrayData, Float
from aiida import orm, engine, plugins
from astropy.stats import mad_std

import numpy as np
import os
from astropy.io import fits
import ccdproc

FITSDATA = plugins.DataFactory('fits.data')

@calcfunction
def make_bias_master(path_collection:orm.Str,
                     calibrated_path:orm.Str, 
                     comb_method:orm.Str,
                     #comb_params_dict:orm.Dict
                     ):
    """
    Returns a FITS file with the combined bias
    """
    im_collection = ccdproc.ImageFileCollection(path_collection.value)
    biases_im = im_collection.files_filtered(imagetyp='Bias Frame',
                                             include_path=True)
    combined_bias = ccdproc.combine(biases_im, 
                            method=comb_method, 
                            #**comb_params_dict
                            sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                            sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                            mem_limit=350e6,unit='adu',
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
        spec.input('aggregate_method', valid_type = orm.Str)
        spec.input('combination_params', valid_type = orm.Dict, 
                                         required = False)

        spec.outputs.dynamic = True
        spec.outline(
            cls.agg_bias,
            cls.result_cal
        )

    def agg_bias(self):
        #Aggragation of bias images
        self.ctx.bias = make_bias_master(self.inputs.directory_input,
                                         self.inputs.directory_output,
                                         self.inputs.aggregate_method,
                                         #self.inputs.comb_params_dict
                                         )
        
    def result_cal(self):
        self.out('result', self.ctx.bias)

    


