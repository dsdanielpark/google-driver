import requests
import os
import shutil
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def download(URL: str, local_storage_full_path: str) -> None:
    """Just put the full file path in the local area and the Google Drive file path accessible to everyone, and you can download it.

    :param URL: Google Drive file path accessible to everyone
    :type URL: str
    :param local_storage_full_path: Full file name to save to local storage
    :type local_storage_full_path: str
    """
    session = requests.Session()
    response = session.get(URL, stream = True)
    token = get_token(response)
    if token:
        response = session.get(URL, stream = True)
    save_file(response, local_storage_full_path)    

def get_token(response: str) -> str:
    """The response to the Google Drive request is stored in the token.

    :param response: Responding to Google Drive requests
    :type response: str
    :return: Returns if a warning occurs
    :rtype: str
    """
    for k, v in response.cookies.items():
        if k.startswith('download_warning'):
            return v

def save_file(response: str, local_storage_full_path: str):
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


# https://github.com/huggingface/transformers/utils/hub
_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False

def is_offline_mode():
    return _is_offline_mode

torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
old_default_cache_path = os.path.join(torch_cache_home, "transformers")
# New default cache, shared with the Datasets library
hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "hub")

# Onetime move from the old location to the new one if no ENV variable has been set.
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(default_cache_path)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    logger.warning(
        "In Transformers v4.0.0, the default path to cache downloaded models changed from"
        " '~/.cache/torch/transformers' to '~/.cache/huggingface/transformers'. Since you don't seem to have"
        " overridden and '~/.cache/torch/transformers' is a directory that exists, we're moving it to"
        " '~/.cache/huggingface/transformers' to avoid redownloading models you have already in the cache. You should"
        " only see this message once."
    )
    shutil.move(old_default_cache_path, default_cache_path)

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", PYTORCH_TRANSFORMERS_CACHE)