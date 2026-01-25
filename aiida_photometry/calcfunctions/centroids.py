from aiida.engine import calcfunction
from aiida.orm import Dict, ArrayData

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
    data: ArrayData,
    positions: ArrayData,
    options: Dict,
):
    """
    Wrapper around photutils.centroids.centroid_sources

    positions:
        ArrayData with arrays:
          - 'x'
          - 'y'

    options:
        Passed as **kwargs:
          - box_size
          - footprint
          - error
          - mask
          - centroid_func
    """
    image = data.get_array('image')

    xpos = positions.get_array('x')
    ypos = positions.get_array('y')

    kwargs = options.get_dict()

    xcen, ycen = centroid_sources(
        image,
        xpos,
        ypos,
        **kwargs
    )

    result = Dict(dict={
        "x": xcen.tolist(),
        "y": ycen.tolist(),
    })

    return result

