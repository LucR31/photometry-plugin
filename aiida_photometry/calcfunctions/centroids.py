from aiida.engine import calcfunction
from aiida.orm import Dict, ArrayData
import numpy as np
from photutils import DAOStarFinder

from photutils.centroids import (
    centroid_com,
    centroid_quadratic,
    centroid_1dg,
    centroid_2dg,
    centroid_sources
)


def _centroid_to_dict(x, y):
    return Dict(
        dict={
            "x": float(x),
            "y": float(y),
        }
    )


@calcfunction
def centroid_com_cf(
    data: ArrayData,
    options: Dict,
):
    """
    Wrapper around photutils.centroids.centroid_com

    options:
        Passed directly as **kwargs (e.g. mask)
    """
    image = data.get_array("image")
    kwargs = options.get_dict()

    x, y = centroid_com(image, **kwargs)

    return _centroid_to_dict(x, y)


@calcfunction
def centroid_quadratic_cf(
    data: ArrayData,
    options: Dict,
):
    """
    Wrapper around photutils.centroids.centroid_quadratic

    options:
        Passed directly as **kwargs (e.g. xpeak, ypeak, fit_boxsize)
    """
    image = data.get_array("image")
    kwargs = options.get_dict()

    x, y = centroid_quadratic(image, **kwargs)

    return _centroid_to_dict(x, y)


@calcfunction
def centroid_1dg_cf(
    data: ArrayData,
    options: Dict,
):
    """
    Wrapper around photutils.centroids.centroid_1dg

    options:
        Passed directly as **kwargs (e.g. error, mask)
    """
    image = data.get_array("image")
    kwargs = options.get_dict()

    x, y = centroid_1dg(image, **kwargs)

    return _centroid_to_dict(x, y)


@calcfunction
def centroid_2dg_cf(
    data: ArrayData,
    options: Dict,
):
    """
    Wrapper around photutils.centroids.centroid_2dg

    options:
        Passed directly as **kwargs (e.g. error, mask)
    """
    image = data.get_array("image")
    kwargs = options.get_dict()

    x, y = centroid_2dg(image, **kwargs)

    return _centroid_to_dict(x, y)

@calcfunction
def centroid_sources_cf(
    image: ArrayData,
    positions: ArrayData,
    options: Dict
) -> ArrayData:
    """
    Refine given source positions using centroid_sources.
    """
    img_array = image.get_array("image")
    xpos = np.array(positions.get_array("x"), dtype=float)
    ypos = np.array(positions.get_array("y"), dtype=float)
    kwargs = options.get_dict()

    xcen, ycen = centroid_sources(img_array, xpos, ypos, **kwargs)

    result = ArrayData()
    result.set_array("x", np.array(xcen, dtype=float))
    result.set_array("y", np.array(ycen, dtype=float))
    return result

@calcfunction
def detect_sources_cf(image: ArrayData, options: Dict) -> ArrayData:
    """
    Detect sources in an image using DAOStarFinder.

    Returns ArrayData with 'x' and 'y'.
    """
    img_array = image.get_array("image")
    kwargs = options.get_dict()

    threshold = kwargs.get("threshold", 3.0)
    fwhm = kwargs.get("fwhm", 3.0)
    daofinder = DAOStarFinder(threshold=threshold, fwhm=fwhm)

    sources_table = daofinder(img_array)

    if sources_table is None or len(sources_table) == 0:
        xpos = np.array([], dtype=float)
        ypos = np.array([], dtype=float)
    else:
        xpos = getattr(sources_table["xcentroid"], "value", sources_table["xcentroid"])
        ypos = getattr(sources_table["ycentroid"], "value", sources_table["ycentroid"])
        xpos = np.array(xpos, dtype=float)
        ypos = np.array(ypos, dtype=float)

    result = ArrayData()
    result.set_array("x", xpos)
    result.set_array("y", ypos)
    return result