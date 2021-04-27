import sys
sys.path.append("..")
import tsd


import datetime
import geojson
import numpy as np                   # numeric linear algebra
import matplotlib.pyplot as plt      # plotting
import rasterio       # read/write geotiffs
import utils          # IO and coordinate system conversion tools
import pandas as pd
import random


from tsd import s2_metadata_parser
import multiprocessing
from tsd.get_sentinel2 import get_time_series


from rasterio.rio import stack
from utils import crop_aoi, rio_write, rio_dtype
import glob
from dateutil.relativedelta import relativedelta
from library import *

def get_all_L1C_bands(tile,title,outdir, flush = False):
    """ Get all bands from the level 1C
    these bands differ in spatial resolution:
    https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
    """
    if flush:
        mkdir("./"+outdir)
        os.system("rm -r ./"+outdir)
        mkdir("./"+outdir)
    bands = ["B%.2i"%(i+1) for i in range(12)]
    get_time_series(bands=bands,
                    tile_id=tile,
                    title= title+'.SAFE',
                    out_dir=outdir,
                    api='scihub',
                    mirror='gcloud',
                    no_crop=True,
                    product_type='L1C',
                    cloud_masks=False,
                    parallel_downloads=multiprocessing.cpu_count()
                    )
    paths = [glob.glob("./%s/*%s.jp2"%(outdir,b))[0] if len(glob.glob("./%s/*%s.jp2"%(outdir,b)))>0 else None for b in bands  ]
    return paths
    

def get_SCL_band(tile,title,outdir):
    mkdir("./"+outdir)
    os.system("rm -r ./"+outdir)
    mkdir("./"+outdir)
    get_time_series(bands=['SCL'],
                    tile_id=tile,
                    title= title+'.SAFE',
                    out_dir=outdir,
                    api='scihub',
                    mirror='gcloud',
                    no_crop=True,
                    product_type='L2A',
                    cloud_masks=False,
                    parallel_downloads=multiprocessing.cpu_count()
                    )
    path_scl = glob.glob("./%s/*SCL.jp2"%outdir)[0] if len(glob.glob("./%s/*SCL.jp2"%outdir))>0 else None
    return path_scl

def compute_cloud_mask(scl,upsample=False):
    mh_prob_clouds = np.logical_or(scl == 8, scl == 0).astype(np.uint8) * 175
    cirrus = (scl == 10).astype(np.uint8) * 100
    snow = (scl == 11).astype(np.uint8) * 255
    no_data = (scl == 0).astype(np.uint8) * 0

    cloud_mask = mh_prob_clouds + cirrus + snow + no_data
    if upsample:
        cloud_mask = cloud_mask.repeat(2, axis=0).repeat(2, axis=1)
    return scl

def compute_rgb(paths):
    b02 = rasterio.open(paths[1], "r")
    b03 = rasterio.open(paths[2], "r")
    b04 = rasterio.open(paths[3], "r")
    rgb = np.stack((b04.read(True),b03.read(True),b02.read(True)),axis=-1)
    b02.close()
    b03.close()
    b04.close()
    return rgb

def simple_equalization_8bit(im, percentiles=5):
    mi, ma = np.percentile(im.flatten(), (percentiles,100-percentiles))
    im = np.minimum(np.maximum(im,mi), ma) # clip
    im = (im-mi)/(ma-mi)*255.0   # scale
    im = im.astype(np.uint8)
    return im




random.seed(42)
mkdir("./dataset")

tiles = np.load("tiles.npy")
random.shuffle(tiles)
# tiles = tiles[0:2]
# tiles = ["51QUG"]
# tiles = ["T50RNN"]


start_date = datetime.datetime(2019, 12, 1)
end_date = datetime.datetime(2020, 12, 1)
count = 0
for tile in tiles:

    # run the query
    image_catalog_l1c = tsd.get_sentinel2.search(aoi=None,tile_id=tile,product_type="L1C",start_date=start_date, end_date=end_date, api='scihub')
    image_catalog_l2a = tsd.get_sentinel2.search(aoi=None,tile_id=tile,product_type="L2A",start_date=start_date, end_date=end_date, api='scihub')

    if len(image_catalog_l1c)==0 or len(image_catalog_l1c)!=len(image_catalog_l2a):
        print("Error: lenghts of two catalogs: %i - %i"% (len(image_catalog_l1c),len(image_catalog_l2a)))
        continue
    
    indices = list(range(len(image_catalog_l1c)))

    print(len(indices))
    random.shuffle(indices)
    sclflag = False
    for idx in indices:
        title_l1c = image_catalog_l1c[idx]['title']
        title_l2a = image_catalog_l2a[idx]['title']
        date = image_catalog_l1c[idx]['date']
        
        okflag = title_l1c.split("_")[2]==title_l2a.split("_")[2]
        if not okflag:
            print("Error: l1c and l2a not corresponding! continuing...")
            continue
        else:
            path_scl_A = get_SCL_band(tile, title_l2a, "query")
            if path_scl_A is None:
                continue
            try:
                scl_A = rasterio.open(path_scl_A, "r")
            except rasterio.errors.RasterioIOError:
                continue
            sclflag = True
            break
    
    if sclflag:
        print(title_l2a)        
        
        cmA = scl_A.read(True)
        clouds = np.array(cmA == 8) + np.array(cmA == 9) #+ np.array(cmA == 10)
        goodflag = np.array(cmA != 0) * np.array(cmA != 1)
        goodflag = True
        h, w = np.shape(cmA)
        if np.all(goodflag) and np.sum(clouds)>0.1*h*w and np.sum(clouds)<0.9*h*w:
            paths_A = get_all_L1C_bands(tile, title_l1c, "dataset")
            count = count + 1
            os.system("mv query/*.jp2 dataset/")
            # bands = [rasterio.open(p, "r").read(1) for p in paths_A]
            # upsampled = [b.repeat(int(10980/b.shape[0]), axis=0).repeat(int(10980/b.shape[1]), axis=1) for b in bands]
            # [rio_write("query_A/B%.2i.tif"%(i+1), b[2*x:(2*x+2*h),2*y:(2*y+2*h)]) for i,b in enumerate(upsampled)]
        scl_A.close()
        if count > 100:
            break




