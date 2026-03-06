import numpy as np
import os
from astropy.io import fits
from aiida import orm
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import run

FitsData = DataFactory("fits.data")

bias_dir = "/home/jovyan/work/data/calibration_images/2026-02-27_14_24_Bias"
dark_dir = "/home/jovyan/work/data/calibration_images/2026-02-27_14_21_Dark_10s"

bias_nodes = {}
for i, filename in enumerate(sorted(os.listdir(bias_dir))):
    if filename.endswith(".fit"):
        path = os.path.abspath(os.path.join(bias_dir, filename))
        node = FitsData(file=path).store()
        bias_nodes[f"bias_{i}"] = node

dark_nodes = {}
for i, filename in enumerate(sorted(os.listdir(dark_dir))):
    if filename.endswith(".fit"):
        path = os.path.abspath(os.path.join(dark_dir, filename))
        node = FitsData(file=path).store()
        dark_nodes[f"dark_{i}"] = node

flat_nodes = {}
for i, filename in enumerate(sorted(os.listdir(dark_dir))):
    if filename.endswith(".fit"):
        path = os.path.abspath(os.path.join(dark_dir, filename))
        node = FitsData(file=path).store()
        flat_nodes[f"flat_{i}"] = node

def create_synthetic_fits(path, value, shape=(50, 50)):
    """Create a FITS file with constant pixel values."""
    data = np.full(shape, value, dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["BUNIT"] = "adu"
    hdu.header["EXPTIME"] = 60.0
    hdu.header["FILTER"] = "R"
    hdu.writeto(path, overwrite=True)
    return path

# Synthetic science frame
science_path = create_synthetic_fits("science.fits", value=1100)
science_node = FitsData(file=os.path.abspath(science_path)).store()

# BUILDER
WorkflowClass = WorkflowFactory("images.reduction")
builder = WorkflowClass.get_builder()
builder.bias_frames = bias_nodes
builder.dark_frames = dark_nodes
builder.flat_frames = flat_nodes
builder.raw_science = science_node
builder.parameters = orm.Dict({"unit": "adu"})
result = run(builder)
