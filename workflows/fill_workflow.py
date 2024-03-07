from aiida.engine import calcfunction
from aiida.orm import Int, ArrayData, Float
from aiida import orm, engine
import numpy as np
import math
from photutils.aperture import aperture_photometry,CircularAperture
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

@calcfunction
def mask_segmentation(data):
    data_m = data.get_array('data')
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(data_m, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(data_m, threshold, npixels=10)
    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)
    return Int(0)
     
@calcfunction
def remove_background(data, sigma, mask):
    data_m = data.get_array('data')
    _, median, _ = sigma_clipped_stats(data_m, sigma)
    data_m = data_m - median  # subtract background from the data
    new_mat  = ArrayData()
    new_mat.set_array('masked', data_m)
    return new_mat

@calcfunction
def circular_aperture(data, method):
    data_m = data.get_array('masked')
    positions = [(30.0, 30.0), (40.0, 40.0)]
    aperture = CircularAperture(positions, r=3)
    phot_table = aperture_photometry(data_m, aperture, method)  
    return Float(phot_table['aperture_sum'][0])

class FillWorkflow(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('data', valid_type=orm.ArrayData, help='')
        spec.input('aperture_method', valid_type=orm.Str)
        spec.outputs.dynamic = True

        spec.outline(
            cls.mask_apply,
            cls.background,
            cls.aperture,
            cls.results,
        )
        spec.output('result')
    def mask_apply(self):
        self.ctx.mask = mask_segmentation(self.inputs.data)
    def background(self):
        self.ctx.data = remove_background(self.inputs.data,3,self.ctx.mask)
    def aperture(self):
        self.ctx.table = circular_aperture(self.ctx.data, self.inputs.aperture_method)
        
    def results(self):
        self.out('result', self.ctx.table)
        #self.out('re', self.ctx.matrix)
        