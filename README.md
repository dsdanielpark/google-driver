# Google-Driver
![YouTuber](https://img.shields.io/badge/pypi-googledriver-blue)
![Pypi Version](https://img.shields.io/pypi/v/googledriver.svg)
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v2.0%20adopted-black.svg)](code_of_conduct.md)
[![Python Version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-black.svg)](code_of_conduct.md)
![Code convention](https://img.shields.io/badge/code%20convention-pep8-black)
![Black Fomatter](https://img.shields.io/badge/code%20style-black-000000.svg)

The `Google-Driver` Python package simplifies downloading files and folders from Google Drive, with some constraints for model management through Hugging-Face and Git-Lfs. Issues may arise due to too many files, access permissions, or large file sizes. Be cautious with large files as they may contain viruses. The `gdown` package offers better exception handling and compatibility.

<br>


# Installation
```
pip install googledriver
```

<br>

# Features
## 1. File Download
`Download to specific path` <br>
To save a file from a shared Google Drive URL to local storage, use the following code.
```python
from googledriver import download

URL = 'https://drive.google.com/file/d/xxxxxxxxx/view?usp=share_link'
download(URL, './model/tf_gpt2_model')
```

<br>

`Download to cached folder` <br>
To download a cached file (or directory) from a URL and return its path, you can use the following method.

```python
from googledriver import download

URL = 'https://drive.google.com/file/d/xxxxxxxxx/view?usp=share_link'
cached_path = download(URL, None, 'tf_model')
```
Basically, torch cached is used, and the huggingface hub module is used as a reference and wrapped.

<br>

## 2. Folder Download
The return value returns the path of the saved files. However, it is different when using it as a cache folder. <br><br>
`Download to current working directory` <br>
```python
from googledriver import download_folder

URL = 'https://drive.google.com/file/d/xxxxxxxxx/view?usp=share_link'

download_folder(URL)
```
<Br>

`Download to specific directory` <br>
```python
from googledriver import download_folder

URL = 'https://drive.google.com/file/d/xxxxxxxxx/view?usp=share_link'
save_folder = './any/path/to/save/'

download_folder(URL, save_folder)
```

<br>

`Download to cached directory` <br>
```python
from googledriver import download_folder

URL = 'https://drive.google.com/file/d/xxxxxxxxx/view?usp=share_link'

download_folder(URL, cached=True)
```
In the case of the cache folder, the return value is the cache folder path of Google Drive. Therefore, it may be difficult to cache and use multiple folders.


<br>

# References
[1] https://github.com/huggingface/transformers <br>
[2] https://github.com/wkentaro/gdown
