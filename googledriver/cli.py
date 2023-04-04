from __future__ import print_function

import argparse
import os.path
import re
import sys
import warnings

import six

from . import __version__
from .downloader import download
from .folder_downloader import download_folder
from .folder_downloader import MAX_DOWNLOAD_LIMIT


class _ShowVersionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(
            "google0driver {ver} at {pos}".format(
                ver=__version__, pos=os.path.dirname(os.path.dirname(__file__))
            )
        )
        parser.exit()


def file_size(argv):
    if argv is not None:
        m = re.match(r"([0-9]+)(GB|MB|KB|B)", argv)
        if not m:
            raise TypeError
        size, unit = m.groups()
        size = float(size)
        if unit == "KB":
            size *= 1024
        elif unit == "MB":
            size *= 1024 ** 2
        elif unit == "GB":
            size *= 1024 ** 3
        elif unit == "B":
            pass
        return size


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-V",
        "--version",
        action=_ShowVersionAction,
        help="display version of google-driver package.",
        nargs=0,
    )
    parser.add_argument(
        "--url", help="url(permission must be checked) to download from"
    )

    parser.add_argument("-O", "--output", help="output path")

    parser.add_argument(
        "--folder",
        action="store_true",
        help="download entire folder instead of a single file "
        "(max {max} files per folder)".format(max=MAX_DOWNLOAD_LIMIT),
    )

    parser.add_argument(
        "--save_path", "-O", "--output", help="full file or folder path",
    )

    args = parser.parse_args()
    url = None

    if args.output == "-":
        if six.PY3:
            args.output = sys.stdout.buffer
        else:
            args.output = sys.stdout

    if args.folder:
        filenames = download_folder(URL=url, save_path=None, cached_filename=None)
        success = filenames is not None
    else:
        filename = download(URL=url, save_path=None, cached=None)
        success = filename is not None

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
