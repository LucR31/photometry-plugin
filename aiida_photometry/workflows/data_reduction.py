from aiida.engine import calcfunction
from aiida.orm import Int, ArrayData, Float
from aiida import orm, engine, plugin

import numpy as np
import os
from astropy.io import fits
from ccdproc import ImageFileCollection,combine

FITSDATA = plugin.DataFactory('super.fitsdata')

def combine_images(images:ImageFileCollection,
                   comb_method:str,
                   dict_params:dict):
    """
    Combines previously calibrated images
    See 'astropy.org' for a detail description
    """
    return combine(images, method=comb_method, **dict_params)

@calcfunction
def make_bias_master(path_collection:orm.Str,
                     calibrated_path:orm.Str, 
                     comb_method:orm.Str,
                     comb_params_dict:orm.Dict):
    """
    Returns a FITS file with the combined bias
    """
    im_collection = ImageFileCollection(path_collection.value)
    biases_im = im_collection.files_filtered(imagetyp='Bias Frame',
                                             include_path=True)
    combined_bias = combine_images(biases_im, 
                                   comb_method, 
                                   comb_params_dict)
    combined_bias.meta['combined'] = True
    combined_bias.write(calibrated_path.value+'/combined_bias.fit')
    return FITSDATA(calibrated_path.value+'/combined_bias.fit')

@calcfunction
def apply_calibration(bias,dark,flat,image):
    pass

@calcfunction
def make_masters(path_collection, calibrated_path, imagetyp, comb_method):
    """
    Convenience function for combining multiple images and types and save
    those in the specified directoy.

    Parameters
    -----------
    path_collection: str
        Contains all the images to be processed.

    imagetype: str
        declares the type of image to combine, options are Bias Frame,
        Dark Frame and Flat Frame.

    comb_method: str
        Method for combining the images. Options:
        average, sum and median

    Returns
    --------
    """
    im_collection = ImageFileCollection(path_collection.value)

    if imagetyp == 'Dark Frame':
        darks = im_collection.summary['imagetyp'] == 'Dark Frame'
        for et in sorted(set(im_collection.summary['exptime'][darks])):
            darks_im = im_collection.files_filtered(imagetyp=imagetyp,
                                                     exptime=et,
                                                     include_path=True)
            combined_dark = combine_images(darks_im, comb_method)
            combined_dark.meta['combined'] = True
            dark_file_name = '/combined_dark_{:6.3f}.fit'.format(et)
            combined_dark.write(calibrated_path.value + dark_file_name)
        return orm.SinglefileData(calibrated_path.value + dark_file_name)

    elif imagetyp == 'Flat Frame':
        flat_filters = set(h['filter'] for h in im_collection.headers(imagetyp='Flat Frame'))
        #flat_file_name =''
        for filt in flat_filters:
            flats_im = im_collection.files_filtered(imagetyp=imagetyp,
                                                     filter=filt,
                                                     include_path=True)
            combined_flat = combine_images(flats_im, comb_method)
            combined_flat.meta['combined'] = True
            flat_file_name = '/combined_flat_filter_{}.fit'.format(filt.replace("''", "p"))
            combined_flat.write(calibrated_path.value + flat_file_name)
        #return orm.SinglefileData(calibrated_path.value + flat_file_name)
             
class DataReduction(engine.WorkChain):

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('directory_input', valid_type = orm.Str)
        spec.input('image', valid_type = orm.Str)
        spec.input('directory_output', valid_type = orm.Str)
        spec.input('aggregate_method', valid_type = orm.Str)
        spec.input('combination_params', valid_type = orm.Dict)

        spec.outputs.dynamic = True
        spec.outline(
            cls.agg_bias,
            #cls.agg_dark,
            #cls.agg_flat,
            #cls.apply_cal,
            cls.result_cal
        )

    def agg_bias(self):
        #Aggragation of bias images
        self.ctx.bias = make_bias_master(self.inputs.directory_input,
                     self.inputs.directory_output,
                     self.inputs.aggregate_method,
                     self.inputs.comb_params_dict)
        
    def agg_dark(self):
        self.ctx.dark = make_masters(self.inputs.directory_input,
                     self.inputs.directory_output,
                    'Dark Frame',
                     self.inputs.aggregate_method)
    def agg_flat(self):
        make_masters(self.inputs.directory_input,
                     self.inputs.directory_output,
                    'Flat Frame',
                     self.inputs.aggregate_method)
    def apply_cal(self):
        self.ctx.res = apply_calibration(self.ctx.bias,
                                         self.ctx.dark,
                                         self.ctx.bias,
                                         self.inputs.image)
    def result_cal(self):
        self.out('result', self.ctx.bias)

    


