from aiida.engine import calcfunction
from aiida.orm import ArrayData, Dict

from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    EllipticalAperture,
    EllipticalAnnulus,
    aperture_photometry,
    RectangularAperture,
    RectangularAnnulus
)

from aiida_photometry.data.fits_data import FitsData


def _table_to_dict(table):
    data = {}
    units = {}

    for name in table.colnames:
        col = table[name]

        if hasattr(col, "unit"):
            data[name] = col.value.tolist()
            units[name] = str(col.unit)
        else:
            data[name] = col.tolist()

    return Dict(
        dict={
            "data": data,
            "units": units,
        }
    )


@calcfunction
def circular_aperture_photometry_cf(
    image: FitsData,
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
    ccd = image.get_ccddata()
    data = ccd.data

    x = positions.get_array("x")
    y = positions.get_array("y")
    r = radius.get_dict()["r"]

    apertures = CircularAperture(list(zip(x, y)), r=r)

    table = aperture_photometry(data, apertures, **options.get_dict())

    return _table_to_dict(table)


@calcfunction
def circular_annulus_photometry_cf(
    image: FitsData,
    positions: ArrayData,
    radii: Dict,
    options: Dict,
):
    """
    Circular annulus photometry.

    radii:
        Dict with keys 'r_in', 'r_out'
    """
    ccd = image.get_ccddata()
    data = ccd.data
    x = positions.get_array("x")
    y = positions.get_array("y")
    r_in = radii.get_dict()["r_in"]
    r_out = radii.get_dict()["r_out"]

    apertures = CircularAnnulus(
        list(zip(x, y)),
        r_in=r_in,
        r_out=r_out,
    )

    table = aperture_photometry(data, apertures, **options.get_dict())

    return _table_to_dict(table)


@calcfunction
def elliptical_aperture_photometry_cf(
    image: FitsData,
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
    ccd = image.get_ccddata()
    data = ccd.data
    x = positions.get_array("x")
    y = positions.get_array("y")
    g = geometry.get_dict()

    apertures = EllipticalAperture(
        list(zip(x, y)),
        a=g["a"],
        b=g["b"],
        theta=g["theta"],
    )

    table = aperture_photometry(data, apertures, **options.get_dict())

    return _table_to_dict(table)


@calcfunction
def elliptical_annulus_photometry_cf(
    image: FitsData,
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
    ccd = image.get_ccddata()
    data = ccd.data
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

    table = aperture_photometry(data, apertures, **options.get_dict())

    return _table_to_dict(table)

@calcfunction
def rectangular_aperture_photometry_cf(
    image: FitsData,
    positions: ArrayData,
    geometry: Dict,
    options: Dict,
):
    """
    Rectangular aperture photometry.

    geometry:
        Dict with keys:
          - w      (width)
          - h      (height)
          - theta  (rotation angle in radians)
    """

    ccd = image.get_ccddata()
    data = ccd.data
    x = positions.get_array("x")
    y = positions.get_array("y")
    g = geometry.get_dict()

    apertures = RectangularAperture(
        list(zip(x, y)),
        w=g["w"],
        h=g["h"],
        theta=g.get("theta", 0.0),
    )

    table = aperture_photometry(data, apertures, **options.get_dict())

    return _table_to_dict(table)

@calcfunction
def rectangular_annulus_photometry_cf(
    image: FitsData,
    positions: ArrayData,
    geometry: Dict,
    options: Dict,
):
    """
    Rectangular annulus photometry.

    geometry:
        Dict with keys:
          - w_in
          - w_out
          - h_in
          - h_out
          - theta
    """

    ccd = image.get_ccddata()
    data = ccd.data
    x = positions.get_array("x")
    y = positions.get_array("y")
    g = geometry.get_dict()

    apertures = RectangularAnnulus(
        list(zip(x, y)),
        w_in=g["w_in"],
        w_out=g["w_out"],
        h_in=g["h_in"],
        h_out=g["h_out"],
        theta=g.get("theta", 0.0),
    )

    table = aperture_photometry(data, apertures, **options.get_dict())

    return _table_to_dict(table)
