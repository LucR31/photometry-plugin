import os
from aiida.orm import Data
from astropy.io import fits
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path

class FitsData(Data):
    """Data class that can be used to wrap a single text file by storing it in its file repository."""

    def __init__(self, filepath, **kwargs):
        """Construct a new instance and set the contents to that of the file.

        :param file: an absolute filepath of the file to wrap
        """
        super().__init__(**kwargs)

        filename = os.path.basename(filepath)  # Get the filename from the absolute path
        self.put_object_from_file(filepath, filename)  # Store the file in the repository under the given filename
        self.base.attributes.set('filename', filename)  # Store in the attributes what the filename is

    
    def get_info(self):
        with tempfile.TemporaryDirectory() as td:
            self.copy_tree(Path(td))
            with fits.open(Path(td)/self.base.attributes.get('filename')) as f:
                    print(f.info())

    def data_to_plot(self):
        with tempfile.TemporaryDirectory() as td:
            self.copy_tree(Path(td))
            with fits.open(Path(td)/self.base.attributes.get('filename')) as f:
                return  f[0].data

