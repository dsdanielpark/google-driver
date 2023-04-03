# Partial Reference https://github.com/huggingface/transformers/utils/hub

import requests
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse
# import logging

# logger = logging.getLogger()

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
_is_offline_mode = (
    True
    if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES
    else False
)

torch_cache_home = os.getenv(
    "TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch")
)
gd_cache_home = os.path.join(torch_cache_home, "googledriver")
DEFAULT_CACHE_FOLDER = os.path.join(gd_cache_home, "hub")
_CACHED_NO_EXIST = object()


def is_offline_mode():
    return _is_offline_mode


def download(URL: str, local_storage_full_path: str, cached_filename=None) -> str:
    """Just put the full file path in the local area and the Google Drive file path accessible to everyone, and you can download it.

    :param URL: Google Drive file path accessible to everyone
    :type URL: str
    :param local_storage_full_path: Full file name to save to local storage
    :type local_storage_full_path: str
    :param cached_filename: _description_, defaults to None
    :type cached_filename: File save name if you want to use it as a cache, optional
    :return: cache file storage path
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
            save_file(response, os.path.join(DEFAULT_CACHE_FOLDER, cached_filename))
    else:
        cached = save_file(response, local_storage_full_path)

    return cached


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


def save_file(response: str, local_storage_full_path: str) -> None:
    """Save the file to local storage in response to the request.

    :param response: Responding to Google Drive requests
    :type response: str
    :param local_storage_full_path: Full file name to save to local storage
    :type local_storage_full_path: str
    """
    CHUNK_SIZE = 40000
    with open(local_storage_full_path, "wb") as f:
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
    force_download: bool = False,
    resume_download: bool = False,
    local_files_only: bool = False,
):

    if is_offline_mode() and not local_files_only:
        print("Offline mode: forcing local_files_only=True")
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
    force_download: bool = False,
    resume_download: bool = False,
    local_files_only: bool = False,
):
    download(URL, "", filename)

    return cached_file(
        cached_filename=filename,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        local_files_only=local_files_only,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )


# def download_url(url, proxies=None):
#     """
#     Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use is
#     for deprecated behavior allowing to download config/models with a single url instead of using the Hub.

#     Args:
#         url (`str`): The url of the file to download.
#         proxies (`Dict[str, str]`, *optional*):
#             A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
#             'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.

#     Returns:
#         `str`: The location of the temporary file where the url was downloaded.
#     """
#     warnings.warn(
#         f"Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
#         " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
#         " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
#         " multiple processes (each process will download the file in a different temporary file)."
#     )
#     tmp_file = tempfile.mktemp()
#     with open(tmp_file, "wb") as f:
#         http_get(url, f, proxies=proxies)
#     return tmp_file


# def has_file(
#     path_or_repo: Union[str, os.PathLike],
#     filename: str,
#     revision: Optional[str] = None,
#     proxies: Optional[Dict[str, str]] = None,
#     use_auth_token: Optional[Union[bool, str]] = None,
# ):
#     """
#     Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

#     <Tip warning={false}>

#     This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
#     this repo, but will return False for regular connection errors.

#     </Tip>
#     """
#     if os.path.isdir(path_or_repo):
#         return os.path.isfile(os.path.join(path_or_repo, filename))

#     url = hf_hub_url(path_or_repo, filename=filename, revision=revision)
#     headers = build_hf_headers(use_auth_token=use_auth_token, user_agent=http_user_agent())

#     r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=10)
#     try:
#         hf_raise_for_status(r)
#         return True
#     except RepositoryNotFoundError as e:
#         logger.error(e)
#         raise EnvironmentError(f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'.")
#     except RevisionNotFoundError as e:
#         logger.error(e)
#         raise EnvironmentError(
#             f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
#             f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
#         )
#     except requests.HTTPError:
#         # We return false for EntryNotFoundError (logical) as well as any connection error.
#         return False
