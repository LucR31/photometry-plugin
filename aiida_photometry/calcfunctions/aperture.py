from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict

from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    EllipticalAperture,
    EllipticalAnnulus,
    aperture_photometry,
)

def _table_to_dict(table):
    return Dict(dict={
        name: table[name].tolist()
        for name in table.colnames
    })


@calcfunction
def circular_aperture_photometry_cf(
    data: ArrayData,
    positions: ArrayData,
    radius: Dict,
    options: Dict,
):
    """
    Circular aperture photometry.

    positions:
        ArrayData with arrays 'x', 'y'

    radius:
        Dict with key 'r'

    options:
        Passed to aperture_photometry (**kwargs)
    """
    image = data.get_array("image")
    x = positions.get_array("x")
    y = positions.get_array("y")

    r = radius.get_dict()["r"]

    apertures = CircularAperture(list(zip(x, y)), r=r)

    table = aperture_photometry(
        image,
        apertures,
        **options.get_dict()
    )

    return _table_to_dict(table)


@calcfunction
def circular_annulus_photometry_cf(
    data: ArrayData,
    positions: ArrayData,
    radii: Dict,
    options: Dict,
):
    """
    Circular annulus photometry.

    radii:
        Dict with keys 'r_in', 'r_out'
    """
    image = data.get_array("image")
    x = positions.get_array("x")
    y = positions.get_array("y")

    r_in = radii.get_dict()["r_in"]
    r_out = radii.get_dict()["r_out"]

    apertures = CircularAnnulus(
        list(zip(x, y)),
        r_in=r_in,
        r_out=r_out,
    )

    table = aperture_photometry(
        image,
        apertures,
        **options.get_dict()
    )

    return _table_to_dict(table)

@calcfunction
def elliptical_aperture_photometry_cf(
    data: ArrayData,
    positions: ArrayData,
    geometry: Dict,
    options: Dict,
):
    """
    Elliptical aperture photometry.

    geometry:
        Dict with keys:
          - a
          - b
          - theta
    """
    image = data.get_array("image")
    x = positions.get_array("x")
    y = positions.get_array("y")

    g = geometry.get_dict()

    apertures = EllipticalAperture(
        list(zip(x, y)),
        a=g["a"],
        b=g["b"],
        theta=g["theta"],
    )

    table = aperture_photometry(
        image,
        apertures,
        **options.get_dict()
    )

    return _table_to_dict(table)

@calcfunction
def elliptical_annulus_photometry_cf(
    data: ArrayData,
    positions: ArrayData,
    geometry: Dict,
    options: Dict,
):
    """
    Elliptical annulus photometry.

    geometry:
        Dict with keys:
          - a_in
          - a_out
          - b_in
          - b_out
          - theta
    """
    image = data.get_array("image")
    x = positions.get_array("x")
    y = positions.get_array("y")

    g = geometry.get_dict()

    apertures = EllipticalAnnulus(
        list(zip(x, y)),
        a_in=g["a_in"],
        a_out=g["a_out"],
        b_in=g["b_in"],
        b_out=g["b_out"],
        theta=g["theta"],
    )

    table = aperture_photometry(
        image,
        apertures,
        **options.get_dict()
    )

    return _table_to_dict(table)
