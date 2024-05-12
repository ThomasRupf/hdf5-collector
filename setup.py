from setuptools import find_packages, setup

setup(
    name="hd5f-collector",
    version="0.1.0",
    author="Thomas Rupf",
    author_email="tm.rupf@gmail.com",
    description="TODO",
    url="TODO",
    license="Apache",
    license_files=("LICENSE",),
    long_description="TODO",
    long_description_content_type="text/markdown",
    keywords=["util"],
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    package_data={"hdf5-collector": []},
    install_requires=[],
)
