import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="googledriver",
    version="0.1.5",
    author="parkminwoo",
    author_email="parkminwoo1991@gmail.com",
    description="The Python package google drive facilitates access to files uploaded to Google Drive.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DSDanielPark/google-driver",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "requests"
    ])