from aiida.engine import calcfunction
from aiida.orm import Int, ArrayData, Float
from aiida import orm, engine
import numpy as np
import math
from photutils.aperture import aperture_photometry,CircularAperture,CircularAnnulus
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

def plot_apertures(data, aperture, annulus_aperture):

    norm = simple_norm(data, 'sqrt', percent=99)
    plt.imshow(data, norm=norm, interpolation='nearest')
    plt.xlim(0, 170)
    plt.ylim(130, 250)
    ap_patches = aperture.plot(color='white', lw=2,
                            label='Photometry aperture')
    ann_patches = annulus_aperture.plot(color='red', lw=2,
                                        label='Background annulus')
    handles = (ap_patches[0], ann_patches[0])
    plt.legend(loc=(0.17, 0.05), facecolor='#458989', labelcolor='white',
            handles=handles, prop={'weight': 'bold', 'size': 11})


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
def compute_photometry(data, apertures, method):
    data_m = data.get_array('masked')
    phot_table = aperture_photometry(data_m, apertures.value, method)  
    return Float(phot_table['aperture_sum'][0])

@calcfunction
def generate_apertures(positions, rad):
    #TODO type of aperture: annulus, ellipse, rectangular...
    apertures = [CircularAperture(positions.value, r) for r in rad.value]
    annulus_aperture = CircularAnnulus(positions, r_in=10, r_out=15)
    return orm.List(apertures)

class FillWorkflow(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('data', valid_type=orm.ArrayData, help='')
        spec.input('positions', valid_type = orm.List)
        spec.input('rad',valid_type = orm.List)
        spec.input('aperture_method', valid_type=orm.Str)
        spec.outputs.dynamic = True

        spec.outline(
            cls.mask_apply,
            cls.background,
            cls.gen_apt,
            cls.aperture,
            cls.results,
        )
        spec.output('result')

    def gen_apt(self):
        self.ctx.apertures = generate_apertures(self.inputs.positions, 
                                                self.inputs.rad)
    def mask_apply(self):
        self.ctx.mask = mask_segmentation(self.inputs.data)
        
    def background(self):
        self.ctx.data = remove_background(self.inputs.data, 
                                          3,
                                          self.ctx.mask)
    def aperture(self):
        self.ctx.table = compute_photometry(self.ctx.data,
                                           self.ctx.apertures,
                                           self.inputs.aperture_method)
        
    def results(self):
        self.out('result', self.ctx.table)
        
        