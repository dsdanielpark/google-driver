# -*- encoding: utf-8 -*-
# Partial Reference https://github.com/wkentaro/gdown


from __future__ import print_function
from googledriver import download
import itertools
import json
import os
import os.path as ospath
import re
import sys
import textwrap
import warnings
import bs4
import requests


class _GoogleDriveObj(object):
    FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, id, name, type, children=None):
        self.id = id
        self.name = name
        self.type = type
        self.children = children if children is not None else []

    def is_folder(self):
        return self.type == self.FOLDER


def _parse_google_drive_obj(URL, content):

    folder_soup = bs4.BeautifulSoup(content, features="html.parser")

    encoded_data = None
    for script in folder_soup.select("script"):
        inner_html = script.decode_contents()

        if "_DRIVE_ivd" in inner_html:
            regex_iter = re.compile(r"'((?:[^'\\]|\\.)*)'").finditer(inner_html)
            try:
                encoded_data = next(itertools.islice(regex_iter, 1, None)).group(1)
            except StopIteration:
                raise RuntimeError("Couldn't find the folder encoded JS string")
            break

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        decoded = encoded_data.encode("utf-8").decode("unicode_escape")
    folder_arr = json.loads(decoded)

    folder_contents = [] if folder_arr[0] is None else folder_arr[0]

    sep = " - "
    splitted = folder_soup.title.contents[0].split(sep)
    if len(splitted) >= 2:
        name = sep.join(splitted[:-1])
    else:
        raise RuntimeError(
            "file/folder name cannot be extracted from: {}".format(
                folder_soup.title.contents[0]
            )
        )

    google_drive_file = _GoogleDriveObj(
        id=URL.split("/")[-1], name=name, type=_GoogleDriveObj.FOLDER,
    )

    id_name_type_iter = [
        (e[0], e[2].encode("raw_unicode_escape").decode("utf-8"), e[3])
        for e in folder_contents
    ]

    return google_drive_file, id_name_type_iter


def _get_google_drive_objs(URL,):

    status_code = True

    # canonicalize the language into English
    if "?" in URL:
        URL += "&hl=en"
    else:
        URL += "?hl=en"

    session = requests.Session()

    try:
        res = session.get(URL)
    except requests.exceptions.ProxyError as e:
        print("An error has occurred using proxy:", session.proxies, file=sys.stderr)
        print(e, file=sys.stderr)
        return False, None

    if res.status_code != 200:
        return False, None

    google_drive_file, id_name_type_iter = _parse_google_drive_obj(
        URL=URL, content=res.text,
    )

    for child_id, child_name, child_type in id_name_type_iter:
        if child_type != _GoogleDriveObj.FOLDER:
            google_drive_file.children.append(
                _GoogleDriveObj(id=child_id, name=child_name, type=child_type,)
            )
            if not status_code:
                return status_code, None
            continue

        status_code, child = _get_google_drive_objs(
            URL="https://drive.google.com/drive/folders/" + child_id,
        )
        if not status_code:
            return status_code, None
        google_drive_file.children.append(child)

    return status_code, google_drive_file


def _get_directory_structure(google_drive_file, previous_path):
    """Converts a Google Drive folder structure into a local directory list."""

    directory_structure = []
    for file in google_drive_file.children:
        file.name = file.name.replace(ospath.sep, "_")
        if file.is_folder():
            directory_structure.append((None, ospath.join(previous_path, file.name)))
            for i in _get_directory_structure(
                file, ospath.join(previous_path, file.name)
            ):
                directory_structure.append(i)
        elif not file.children:
            directory_structure.append((file.id, ospath.join(previous_path, file.name)))
    return directory_structure


cache_root = os.path.join(os.path.expanduser("~"), ".cache\\googledriver")
if not os.path.exists(cache_root):
    try:
        os.makedirs(cache_root)
    except OSError:
        pass


def download_folder(URL=None, save_path=None, cached=None):

    session = requests.Session()

    try:
        status_code, google_drive_file = _get_google_drive_objs(URL,)
    except RuntimeError as e:
        print("Failed to retrieve folder contents:", file=sys.stderr)
        error = "\n".join(textwrap.wrap(str(e)))
        print("\n", error, "\n", file=sys.stderr)
        return

    if not status_code:
        return status_code
    if save_path is None:
        save_path = os.getcwd() + ospath.sep
    if save_path.endswith(ospath.sep):
        root_folder = ospath.join(save_path, google_drive_file.name)
    else:
        root_folder = save_path

    if cached == True:
        root_folder = cache_root

    directory_structure = _get_directory_structure(google_drive_file, root_folder)
    if not ospath.exists(root_folder):
        os.makedirs(root_folder)

    filenames = []
    for file_id, file_path in directory_structure:
        if file_id is None:
            if not ospath.exists(file_path):
                os.makedirs(file_path, exist_ok=True)
            continue

        filename = download(
            URL="https://drive.google.com/uc?id=" + file_id,
            save_path=file_path,
            cached_filename=None,
        )

        if filename is None:
            return
        filenames.append(filename)

    if cached == True:
        filenames = cache_root

    return filenames
