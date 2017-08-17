# Time Series Downloader (TSD)

[![Build Status](https://travis-ci.com/carlodef/tsd.svg?token=q3ppoFukgX6NERpM7HRM&branch=master)](https://travis-ci.com/carlodef/tsd)

Automatic download and registration of Sentinel, Landsat and Planet crops.

[Carlo de Franchis](mailto:carlo.de-franchis@ens-cachan.fr),
CMLA, ENS Cachan, Université Paris-Saclay, 2016-17

With contributions from [Enric Meinhardt-Llopis](mailto:enric.meinhardt@cmla.ens-cachan.fr), [Axel Davy](mailto:axel.davy@ens.fr) and [Tristan Dagobert](mailto:tristan.dagobert@cmla.ens-cachan.fr).

# Installation and dependencies
The main scripts are `get_landsat.py`, `get_sentinel1.py`, `get_sentinel2.py`
and `get_planet.py`.

They use the Python modules `search_devseed.py`, `search_scihub.py`,
`search_peps.py`, `search_planet.py` and `register.py`.

_Note_: a shell script installing all the needed stuff (`brew`, `python`,
`gdal`...) on an empty macOS is given in the file
[macos_install_from_scratch.sh](macos_install_from_scratch.sh).

## GDAL
The toughest dependency to install is GDAL. All the others are easily installed
with `pip` as shown in the [next section](#python-packages).

### On macOS
There are several ways of installing `gdal`. I recommend option 1: it
gives a version of gdal 2.1 that works with JP2 files, plus bindings
for both python 2 and 3.

#### Option 1: using the GDAL Complete Compatibility Framework.

[Download](http://www.kyngchaos.com/files/software/frameworks/GDAL_Complete-2.1.dmg)
and install the `.dmg` file. Update your `PATH` after the installation by
running this command:

    export PATH="/Library/Frameworks/GDAL.framework/Programs:$PATH"

Copy it in your `~/.profile`.

Then install the GDAL Python bindings and the rasterio package with pip:

    pip install rasterio gdal==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') --global-option build_ext --global-option=`gdal-config --cflags` --global-option build_ext --global-option=-L`gdal-config  --prefix`/unix/lib/

_Note_: installation of `rasterio` with Python 3 requires `numpy`.

The `gdal-config --version | awk -F'[.]' '{print $1"."$2}'` command retrieves
the fist two digits of your gdal version. This information is needed to install
the same version of the python bindings.

The four `--global-option build_ext` options tell `pip` where to find gdal
headers and libraries.

#### Option 2: using brew

    brew install gdal --with-complete --with-python3

This installs gdal and bindings for python 2 and 3. Note that this version
doesn't support JP2 files (hence it will fail to get Sentinel-2 crops from
AWS). Moreover, the version currently bottled in brew is only 1.11 (as of
08/2017).

### On Linux
On Linux `gdal` and its Python bindings are usually straightforward to install
through your package manager.

    sudo apt-get update
    sudo apt-get install libgdal-dev gdal-bin python-gdal


## Python packages
The required Python packages are listed in the file `requirements.txt`. They
can be installed with `pip`:

    pip install -r requirements.txt

# Usage

## From the command line
The pipeline can be used from the command line through the Python scripts
`get_*.py`. For instance, to download and process Sentinel-2 images of the
Jamnagar refinery, located at latitude 22.34806 and longitude 69.86889, run

    python get_sentinel2.py --lat 22.34806 --lon 69.86889 -b 2 3 4 -r -o test

This will download crops of size 5000 x 5000 meters from the bands 2, 3 and 4,
corresponding to the blue, green and red channels, and register them through
time. To specify the desired bands, use the `-b` or `--band` flag. The crop
size can be changed with the `--width` and `--height` flags. For instance

    python get_sentinel2.py --lat 22.34806 --lon 69.86889 -b 11 12 --width 8000 --height 6000

will download crops of size 8000 x 6000 meters, only for the SWIR channels (bands 11
and 12), without registration (no option `-r`).

All the available options are listed when using the `-h` or `--help` flag:

    python get_sentinel2.py -h

You can also run any of the `search_*.py` scripts or `registration.py` from
the command line separately to use only single blocks of the pipeline. Run them
with `-h` to get the list of available options.

## As Python modules

The Python modules can be imported to call their functions from Python. Refer
to their docstrings to get usage information. Here are some examples.

    # define an area of interest
    import utils
    lat, lon = 42, 3
    aoi = utils.geojson_geometry_object(lat, lon, 5000, 5000)

    # search Landsat-8 images available on the AOI with Development Seed's API
    import search_devseed
    x = search_devseed.search(aoi, satellite='Landsat-8')
