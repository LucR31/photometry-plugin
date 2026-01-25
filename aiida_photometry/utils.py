import ccdproc


def get_image_by_type(
    path_collection: str, image_type: str
) -> ccdproc.ImageFileCollection:
    # Utility function to return images of a certain image type, i.e., BIAS, DARKS...
    im_collection = ccdproc.ImageFileCollection(path_collection)
    return im_collection.files_filtered(imagetyp=image_type, include_path=True)
