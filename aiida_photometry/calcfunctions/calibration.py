from aiida.engine import calcfunction
from aiida.orm import Dict

import ccdproc
import tempfile
import os
import numpy as np
from astropy import units as u

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
    method = params.get("combine_method", "median")
    sigma_clip = params.get("sigma_clip", True)

    master = ccdproc.combine(
        [f.get_ccddata() for f in frames.values()], method=method, sigma_clip=sigma_clip
    )
    master.meta["CALTYPE"] = "MASTER_BIAS"

    return _write_ccd_to_fitsdata(
        master,
        extra_attrs={
            "is_master": True,
            "master_type": "bias",
            "combine_method": method,
        },
    )


@calcfunction
def create_master_dark(master_bias: FitsData = None, parameters: Dict = None, **frames):
    """
    Create master dark from dark frames.
    """
    params = parameters.get_dict()
    method = params.get("combine_method", "median")
    subtract_bias_flag = params.get("subtract_bias", True)

    # Substract Bias if subtract_bias = True
    bias_ccd = master_bias.get_ccddata()
    calibrated = [
        ccdproc.subtract_bias(f.get_ccddata(), bias_ccd)
        if subtract_bias_flag and master_bias is not None
        else f.get_ccddata()
        for f in frames.values()
    ]

    # Combine Darks
    master = ccdproc.combine(calibrated, method=method)
    master.meta["CALTYPE"] = "MASTER_DARK"

    return _write_ccd_to_fitsdata(
        master,
        extra_attrs={
            "is_master": True,
            "master_type": "dark",
            "combine_method": method,
            "bias_subtracted": subtract_bias_flag
        },
    )


@calcfunction
def create_master_flat(
    master_bias: FitsData, master_dark: FitsData, parameters: Dict, **frames
):
    """
    Create master flat (per filter).
    """
    params = parameters.get_dict()
    method = params.get("combine_method", "median")

    bias_ccd = master_bias.get_ccddata()
    dark_ccd = master_dark.get_ccddata()

    calibrated = []

    for f in frames.values():
        flat_ccd = f.get_ccddata()

        flat_sub = ccdproc.subtract_bias(flat_ccd, bias_ccd)
        flat_sub = ccdproc.subtract_dark(
            flat_sub, dark_ccd, exposure_time="EXPTIME", exposure_unit=u.second
        )

        # Normalize by median
        norm_value = np.median(flat_sub.data)
        flat_norm = flat_sub.divide(norm_value)

        calibrated.append(flat_norm)

    # Combine Flats
    master = ccdproc.combine(calibrated, method=method)
    master.meta["CALTYPE"] = "MASTER_FLAT"

    return _write_ccd_to_fitsdata(
        master,
        extra_attrs={
            "is_master": True,
            "master_type": "flat",
            "combine_method": method,
        },
    )


@calcfunction
def subtract_bias_cf(image: FitsData, master_bias: FitsData) -> FitsData:
    """
    Subtract master bias from an image.
    """

    return _write_ccd_to_fitsdata(
        ccdproc.subtract_bias(image.get_ccddata(), master_bias.get_ccddata()),
        extra_attrs = {"calibration": "bias_subtracted"},
    )

@calcfunction
def flat_correct_cf(image: FitsData, master_flat: FitsData) -> FitsData:
    """
    Apply flat field correction.
    """

    return _write_ccd_to_fitsdata(
        ccdproc.flat_correct(image.get_ccddata(), master_flat.get_ccddata()),
        extra_attrs={"calibration": "flat_corrected"},
    )


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
