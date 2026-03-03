from photutils.background import Background2D, MedianBackground

from aiida_photometry.data.fits_data import FitsData
from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict
from aiida.engine import calcfunction

from astropy.stats import sigma_clipped_stats
import numpy as np


@calcfunction
def global_background_cf(
    image: FitsData,
    parameters: Dict,
):
    """
    Estimate global background using sigma clipping.

    parameters:
        Dict with keys:
            - sigma (float)
            - maxiters (int)
    """

    params = parameters.get_dict()
    sigma = params.get("sigma", 3.0)
    maxiters = params.get("maxiters", 5)

    ccd = image.get_ccddata()
    data = ccd.data

    mean, median, std = sigma_clipped_stats(
        data,
        sigma=sigma,
        maxiters=maxiters
    )

    result = {
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "unit": str(ccd.unit),
    }

    return Dict(dict=result)

@calcfunction
def background_2d_cf(
    image: FitsData,
    parameters: Dict,
):
    """
    Compute 2D background model.

    parameters:
        Dict with keys:
            - box_size (int)
            - filter_size (int)
    """

    params = parameters.get_dict()
    box_size = params.get("box_size", 50)
    filter_size = params.get("filter_size", 3)

    ccd = image.get_ccddata()
    data = ccd.data

    bkg = Background2D(
        data,
        box_size,
        filter_size=filter_size,
        bkg_estimator=MedianBackground()
    )

    # Store background map in ArrayData
    bkg_node = ArrayData()
    bkg_node.set_array("background", bkg.background)
    bkg_node.set_array("background_rms", bkg.background_rms)

    return bkg_node