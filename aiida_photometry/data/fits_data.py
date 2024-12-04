import os
from aiida.orm import Data
from astropy.io import fits
import tempfile
from pathlib import Path

class FitsData(Data):
    
    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        filename = os.path.basename(filepath)  
        self.put_object_from_file(filepath, filename)  
        self.base.attributes.set('filename', filename)  
    
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

