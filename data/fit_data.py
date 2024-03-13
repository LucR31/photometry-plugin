import os
from aiida.orm import Data
from astropy.io import fits

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

    def get_content(self):
        """Return the content of the single file stored for this data node.

        :return: the content of the file as a string
        """
        filename = self.base.attributes.get('filename')
        return self.get_object_content(filename)
    
    def get_info(self):
        with fits.open(self.base.attributes.get('filename')) as hdul:
            print(hdul.info())
