from aiida.engine import calcfunction
from aiida.orm import Dict, List
from aiida.plugins import DataFactory

from astropy.nddata import CCDData
import ccdproc
import tempfile
import os
import numpy as np

from aiida_photometry.data.fits_data import FitsData


def _write_ccd_to_fitsdata(ccd, extra_attrs=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.fits")
        ccd.write(path, overwrite=True)

        node = FitsData(file=path)

        if extra_attrs:
            for k, v in extra_attrs.items():
                node.base.attributes.set(k, v)

        return node


def _validate_same_shape(ccd_list):
    shapes = [ccd.data.shape for ccd in ccd_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"Shape mismatch in input frames: {shapes}")


@calcfunction
def create_master_bias(parameters, **frames):
    """
    Combine bias frames into master bias.
    """
    params = parameters.get_dict()
    method = params.get("combine_method", "median")  # median as default!
    sigma_clip = params.get("sigma_clip", True)  # true as default!

    ccd_list = [f.get_ccddata() for f in frames.values()]

    _validate_same_shape(ccd_list)

    master = ccdproc.combine(ccd_list, method=method, sigma_clip=sigma_clip)

    master.meta["CALTYPE"] = "MASTER_BIAS"

    node = _write_ccd_to_fitsdata(
        master,
        extra_attrs={
            "is_master": True,
            "master_type": "bias",
            "combine_method": method,
        },
    )

    return node


@calcfunction
def create_master_dark(frames: List, master_bias: FitsData, parameters: Dict):
    """
    Create master dark from dark frames.
    """
    params = parameters.get_dict()
    method = params.get("combine_method", "median")

    bias_ccd = master_bias.get_ccddata()

    calibrated = []
    for f in frames:
        dark_ccd = f.get_ccddata()
        dark_sub = ccdproc.subtract_bias(dark_ccd, bias_ccd)
        calibrated.append(dark_sub)

    _validate_same_shape(calibrated)

    master = ccdproc.combine(calibrated, method=method)
    master.meta["CALTYPE"] = "MASTER_DARK"

    node = _write_ccd_to_fitsdata(master)
    node.base.attributes.set("is_master", True)
    node.base.attributes.set("master_type", "dark")
    node.base.attributes.set("combine_method", method)

    return node


@calcfunction
def create_master_flat(
    frames: List, master_bias: FitsData, master_dark: FitsData, parameters: Dict
):
    """
    Create master flat (per filter).
    """
    params = parameters.get_dict()
    method = params.get("combine_method", "median")

    bias_ccd = master_bias.get_ccddata()
    dark_ccd = master_dark.get_ccddata()

    calibrated = []

    for f in frames:
        flat_ccd = f.get_ccddata()

        flat_sub = ccdproc.subtract_bias(flat_ccd, bias_ccd)
        flat_sub = ccdproc.subtract_dark(flat_sub, dark_ccd)

        # Normalize by median
        norm_value = np.median(flat_sub.data)
        flat_norm = flat_sub.divide(norm_value)

        calibrated.append(flat_norm)

    _validate_same_shape(calibrated)

    master = ccdproc.combine(calibrated, method=method)
    master.meta["CALTYPE"] = "MASTER_FLAT"

    node = _write_ccd_to_fitsdata(master)
    node.base.attributes.set("is_master", True)
    node.base.attributes.set("master_type", "flat")
    node.base.attributes.set("combine_method", method)

    return node


@calcfunction
def calibrate_science(
    science: FitsData,
    master_bias: FitsData,
    master_dark: FitsData,
    master_flat: FitsData,
    parameters: Dict,
):
    """
    Apply full CCD calibration to a science frame.
    """

    bias = master_bias.get_ccddata()
    dark = master_dark.get_ccddata()
    flat = master_flat.get_ccddata()

    sci = science.get_ccddata()

    # Bias subtraction
    sci = ccdproc.subtract_bias(sci, bias)

    # Dark subtraction
    sci = ccdproc.subtract_dark(sci, dark)

    # Flat correction
    sci = ccdproc.flat_correct(sci, flat)

    sci.meta["CALIBRATED"] = True

    node = _write_ccd_to_fitsdata(sci, extra_attrs={"is_calibrated": True})

    return node
