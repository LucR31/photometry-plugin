# -*- coding: utf-8 -*-
from aiida.engine import calcfunction
from aiida import orm, engine, plugins

from photutils.aperture import CircularAperture, aperture_photometry

FITSDATA = plugins.DataFactory("fits.data")
ArrayData = plugins.DataFactory("core.array")


@calcfunction
def circular_aperture_photometry(
    data,
    # positions,
    # radius
):

    aperture = [CircularAperture([(10, 10)], r=i) for i in [3, 4]]
    qtable = aperture_photometry(data, aperture)
    array = ArrayData()
    array.set_array("as_array", qtable.as_array())
    return array


@calcfunction
def substract_bck(img, bck):
    array = ArrayData()
    array.set_array("substracted", img.data_to_plot() - bck)
    return array


class AperturePhotometry(engine.WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("fits_image", valid_type=FITSDATA)
        spec.input("bck", valid_type=ArrayData)
        spec.input("positions", valid_type=orm.List)
        spec.input("radii", valid_type=orm.List)

        spec.outputs.dynamic = True
        spec.outline(cls.remove_bck, cls.aperture_photometry, cls.result_cal)

    def remove_bck(self):
        self.ctx.img_wo_bck = substract_bck(self.inputs.fits_image, self.inputs.bck)

    def aperture_photometry(self):
        self.ctx.result = circular_aperture_photometry(
            self.ctx.img_wo_bck,
            # self.inputs.positions,
            # self.inputs.radii,
        )

    def result_cal(self):
        self.out("result", self.ctx.result)
