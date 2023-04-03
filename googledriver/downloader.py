# Partial Reference https://github.com/huggingface/transformers/utils/hub

import requests
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
_is_offline_mode = (
    True
    if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES
    else False
)
_CACHED_NO_EXIST = object()

cache_root = os.path.join(os.path.expanduser("~"), ".cache/googledriver")
if not os.path.exists(cache_root):
    try:
        os.makedirs(cache_root)
    except OSError:
        pass

DEFAULT_CACHE_FOLDER = cache_root


def is_offline_mode():
    return _is_offline_mode


def download(URL: str, save_path: str, cached_filename) -> str:
    """Just put the full file path in the local area and the Google Drive file path accessible to everyone, and you can download it.

    :param URL: Google Drive file path accessible to everyone
    :type URL: str
    :param save_path: Full file name to save to local storage
    :type save_path: str
    :param cached_filename: _description_, defaults to None
    :type cached_filename: File save name if you want to use it as a cache, optional
    :return: saved path
    :rtype: str
    """

    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_token(response)
    if token:
        response = session.get(URL, stream=True)

    if cached_filename is not None:
        cached = try_to_load_from_cache(cached_filename, None)
        if cached == _CACHED_NO_EXIST:
            cached_path = os.path.join(DEFAULT_CACHE_FOLDER, cached_filename)
            save_file(response, cached_path)
            saved_path = cached_path
    else:
        save_file(response, save_path)
        saved_path = save_path

    return save_path


def get_token(response: str) -> str:
    """The response to the Google Drive request is stored in the token.

    :param response: Responding to Google Drive requests
    :type response: str
    :return: Returns if a warning occurs
    :rtype: str
    """
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            return v


def save_file(response: str, save_path: str) -> None:
    """Save the file to local storage in response to the request.

    :param response: Responding to Google Drive requests
    :type response: str
    :param save_path: Full file name to save to local storage
    :type save_path: str
    """
    CHUNK_SIZE = 99999999
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    return None


def try_to_load_from_cache(
    cached_filename: str, cache_dir: Union[str, Path, None] = None,
) -> Optional[str]:

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_FOLDER

    cached_file = os.path.join(cache_dir, cached_filename)
    if not os.path.isfile(cached_file):
        return _CACHED_NO_EXIST

    return cached_file if os.path.isfile(cached_file) else None


def cached_file(
    cached_filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    local_files_only: bool = False,
):

    if is_offline_mode() and not local_files_only:
        print("OFF-LINE MODE: forcing local_files_only")
        local_files_only = True

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_FOLDER

    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    try:
        # Load from URL or cache if already cached
        resolved_file = os.path.join(DEFAULT_CACHE_FOLDER, cached_filename)
    except Exception as e:
        print(e)

    return resolved_file


def get_cachefile_from_driver(
    URL: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
):
    download(URL, "", filename)

    return cached_file(
        cached_filename=filename,
        cache_dir=cache_dir,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
