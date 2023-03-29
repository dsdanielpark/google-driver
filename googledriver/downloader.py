import requests

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

