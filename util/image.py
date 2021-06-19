import uuid


def get_extension(filename: str) -> str:
    """ Gets the extension from provided filename. """
    return filename.split(".")[-1]


def validate_image_type(filename: str) -> bool:
    """
    Checks if the given image is of supported type.

    Types supported: .png, .jpg, .jpeg
    """
    supported_extensions = ("png", "jpg", "jpeg")
    return (filename not in (None, "")) and (get_extension(filename) in supported_extensions)


def generate_filename(filename: str) -> str:
    """ Generates new filename for safe storage on server. """
    return f"{str(uuid.uuid4())}.{get_extension(filename)}"
