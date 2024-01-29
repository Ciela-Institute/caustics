from pathlib import Path
import json
import struct

DEFAULT_ENCODING = "utf-8"
SAFETENSORS_METADATA = "__metadata__"


def _normalize_path(path: "str | Path") -> Path:
    # Convert string path to Path object
    if isinstance(path, str):
        path = Path(path)

    # Get absolute path
    return path.absolute()


def to_file(
    path: "str | Path", data: "str | bytes", encoding: str = DEFAULT_ENCODING
) -> str:
    """
    Save data string or bytes to specified file path

    Parameters
    ----------
    path : str or Path
        The path to save the data to
    data : str | bytes
        The data string or bytes to save to file
    encoding : str, optional
        The string encoding to use, by default "utf-8"

    Returns
    -------
    str
        The path string where the data is saved
    """
    # TODO: Update to allow for remote paths saving

    # Convert string data to bytes
    if isinstance(data, str):
        data = data.encode(encoding)

    # Normalize path to pathlib.Path object
    path = _normalize_path(path)

    with open(path, "wb") as f:
        f.write(data)

    return str(path.absolute())


def from_file(path: "str | Path") -> bytes:
    """
    Load data from specified file path

    Parameters
    ----------
    path : str or Path
        The path to load the data from

    Returns
    -------
    bytes
        The data bytes loaded from the file
    """
    # TODO: Update to allow for remote paths loading

    # Normalize path to pathlib.Path object
    path = _normalize_path(path)

    return path.read_bytes()


def _get_safetensors_header(path: "str | Path") -> dict:
    """
    Read specified file header to a dictionary

    Parameters
    ----------
    path : str or Path
        The path to get header from

    Returns
    -------
    dict
        The header dictionary
    """
    # TODO: Update to allow for remote paths loading of header

    # Normalize path to pathlib.Path object
    path = _normalize_path(path)

    # Doing this avoids reading the whole safetensors
    # file in case that it's large
    with open(path, "rb") as f:
        # Get the size of the header by reading first 8 bytes
        (length_of_header,) = struct.unpack("<Q", f.read(8))

        # Get the full header
        header = json.loads(f.read(length_of_header))

        # Only return the metadata
        # if it's not even there, just return blank dict
        return header


def get_safetensors_metadata(path: "str | Path") -> dict:
    """
    Get the metadata from the specified file path

    Parameters
    ----------
    path : str or Path
        The path to get the metadata from

    Returns
    -------
    dict
        The metadata dictionary
    """
    header = _get_safetensors_header(path)

    # Only return the metadata
    # if it's not even there, just return blank dict
    return header.get(SAFETENSORS_METADATA, {})
