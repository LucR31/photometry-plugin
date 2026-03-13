import ccdproc
import ast
import numpy as np
from aiida import orm
import tempfile
import os
from aiida_photometry.data.fits_data import FitsData



def get_image_by_type(
    path_collection: str, image_type: str
) -> ccdproc.ImageFileCollection:
    # Utility function to return images of a certain image type, i.e., BIAS, DARKS...
    im_collection = ccdproc.ImageFileCollection(path_collection)
    return im_collection.files_filtered(imagetyp=image_type, include_path=True)


def positions_from_string(pos_string):
    """
    Convert a string like '[(x1, y1), (x2, y2)]' into ArrayData.
    """
    try:
        positions = ast.literal_eval(pos_string)
    except Exception:
        raise ValueError("Invalid positions string")

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Positions must be a list of (x, y) pairs")

    array_positions = orm.ArrayData()
    array_positions.set_array("x", positions[:, 0])
    array_positions.set_array("y", positions[:, 1])

    return array_positions

def _write_ccd_to_fitsdata(ccd, extra_attrs=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.fits")
        ccd.write(path, overwrite=True)

        node = FitsData(file=path)

        if extra_attrs:
            for k, v in extra_attrs.items():
                node.base.attributes.set(k, v)

        return node
