from aiida.orm import SinglefileData
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData

IMPORTANT_HEADER_KEYS = [
    "EXPTIME",
    "FILTER",
    "DATE-OBS",
    "INSTRUME",
    "TELESCOP",
    "OBJECT",
    "GAIN",
    "RDNOISE",
]


class FitsData(SinglefileData):
    """
    AiiDA data type for FITS images with validated metadata extraction.
    """

    def __init__(self, file=None, **kwargs):
        super().__init__(file=file, **kwargs)

        # Only extract metadata if file is provided and node is not stored
        if file is not None and not self.is_stored:
            self._validate_and_extract_metadata()

    def _validate_and_extract_metadata(self):
        with self.open(mode="rb") as handle:
            with fits.open(handle) as hdul:
                header = hdul[0].header

                # Store curated metadata
                curated = {
                    key: header[key] for key in IMPORTANT_HEADER_KEYS if key in header
                }

                self.base.attributes.set("fits_header", curated)

                # Store shape
                if hdul[0].data is not None:
                    self.base.attributes.set("shape", hdul[0].data.shape)

                # Store unit if present
                if "BUNIT" in header:
                    self.base.attributes.set("unit", header["BUNIT"])

    @property
    def header(self):
        return self.base.attributes.get("fits_header", {})

    def get_ccddata(self, hdu_index=0, default_unit="adu"):
        """
        Return image as CCDData object from FITS file.
        Needed to use ccdproc tools.
        """
        # try reading with header-defined unit first
        try:
            with self.open(mode="rb") as handle:
                return CCDData.read(handle, hdu=hdu_index)
        except ValueError:
            # fallback if no BUNIT present
            with self.open(mode="rb") as handle:
                return CCDData.read(handle, hdu=hdu_index, unit=u.Unit(default_unit))
