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
tiles = tiles[0:2]
# tiles = ["51QUG"]
# tiles = ["T50RNN"]

first_date = datetime.datetime(2019, 12, 1)
for i in range(4):
    start_date = first_date + relativedelta(months=i*3)
    end_date = first_date + relativedelta(months=(i+1)*3) - relativedelta(days=1)
    for tile in tiles:
        # start_date = datetime.datetime(2020, 1, 1)
        # end_date = datetime.datetime(2020, 1, 31)

        # run the query
        image_catalog_l1c = tsd.get_sentinel2.search(aoi=None,tile_id=tile,product_type="L1C",start_date=start_date, end_date=end_date, api='scihub')
        image_catalog_l2a = tsd.get_sentinel2.search(aoi=None,tile_id=tile,product_type="L2A",start_date=start_date, end_date=end_date, api='scihub')

        if len(image_catalog_l1c)==0 or len(image_catalog_l1c)!=len(image_catalog_l2a):
            print("lenghts of two catalogs: %i - %i"% (len(image_catalog_l1c),len(image_catalog_l2a)))
            continue
        
        indices = []
        for i in range(len(image_catalog_l2a)-1):
            for j in np.arange(i,len(image_catalog_l2a)):
                date_A = image_catalog_l2a[i]['date']
                date_B = image_catalog_l2a[j]['date']
                rd = relativedelta(date_B, date_A)
                okdate = rd.years==0 and rd.months==0 and np.abs(rd.days) <= 7 and np.abs(rd.hours+rd.minutes/60)>2
                if okdate:
                    indices.append( [i,j] )

        if len(indices)==0:
            continue
        print(len(indices))
        idxs = list(range(len(indices)-1))
        random.shuffle(idxs)
        sclflag = False
        for idx in idxs:
            idx_A, idx_B = indices[idx]

            title_l1c_A = image_catalog_l1c[idx_A]['title']
            title_l2a_A = image_catalog_l2a[idx_A]['title']
            date_A = image_catalog_l1c[idx_A]['date']
            
            title_l1c_B = image_catalog_l1c[idx_B]['title']
            title_l2a_B = image_catalog_l2a[idx_B]['title']
            date_B = image_catalog_l1c[idx_B]['date']

            okboth = title_l1c_A.split("_")[2]==title_l2a_A.split("_")[2] and title_l1c_B.split("_")[2]==title_l2a_B.split("_")[2]
            rd = relativedelta(date_B, date_A)
            okdate = rd.years==0 and rd.months==0 and np.abs(rd.days) <= 7 and np.abs(rd.hours+rd.minutes/60)>2
            if not (okboth and okdate):
                print("Error with image pair! continuing...")
                continue
            else:
                path_scl_A = get_SCL_band(tile, title_l2a_A, "query_A")
                path_scl_B = get_SCL_band(tile, title_l2a_B, "query_B")
                try:
                    scl_A = rasterio.open(path_scl_A, "r")
                    scl_B = rasterio.open(path_scl_B, "r")
                except rasterio.errors.RasterioIOError:
                    continue
                sclflag = True
                break
        
        if sclflag:
            print(title_l2a_A,title_l2a_B)        
            
            cmA = scl_A.read(True)
            # rio_write('query_A/cloud_mask.tif', compute_thumbnail(cmA,percentiles=0,downsamplestep=2) )
            rio_write('query_A/cloud_mask.tif', cmA[::2,::2] )
            cmB = scl_B.read(True)
            # rio_write('query_B/cloud_mask.tif', compute_thumbnail(cmB,percentiles=0,downsamplestep=2) )
            rio_write('query_B/cloud_mask.tif', cmB[::2,::2] )

            found_pair = False

            for i in range(30):
                x, y = random.randint(0,cmA.shape[0]), random.randint(0,cmA.shape[1])
                h, w = 256, 256
                crop_A = cmA[x:(x+h),y:(y+h)]
                crop_B = cmB[x:(x+h),y:(y+h)]
                clouds_A = np.array(crop_A == 8) + np.array(crop_A == 9) #+ np.array(crop_A == 10)
                clouds_B = np.array(crop_B == 8) + np.array(crop_B == 9) #+ np.array(crop_B == 10)
                crop_okflag = np.all( np.array(crop_A != 0) * np.array(crop_B != 0) * np.array(crop_A != 1) * np.array(crop_B != 1) )
                if not crop_okflag:
                    continue
                if np.sum(clouds_A + clouds_B)>0.2*h*w and np.sum(clouds_A * clouds_B)<0.05*h*w:
                    found_pair = True
                    # This pair might be good to save
                    rio_write('query_A/scl_mask.tif', crop_A.repeat(2, axis=0).repeat(2, axis=1))
                    rio_write('query_B/scl_mask.tif', crop_B.repeat(2, axis=0).repeat(2, axis=1))
                    
                    # equilize all band shapes and save them
                    paths_A = get_all_L1C_bands(tile, title_l1c_A, "query_A")
                    bands = [rasterio.open(p, "r").read(1) for p in paths_A]
                    upsampled = [b.repeat(int(10980/b.shape[0]), axis=0).repeat(int(10980/b.shape[1]), axis=1) for b in bands]
                    [rio_write("query_A/B%.2i.tif"%(i+1), b[2*x:(2*x+2*h),2*y:(2*y+2*h)]) for i,b in enumerate(upsampled)]

                    paths_B = get_all_L1C_bands(tile, title_l1c_B, "query_B")
                    bands = [rasterio.open(p, "r").read(1) for p in paths_B]
                    upsampled = [b.repeat(int(10980/b.shape[0]), axis=0).repeat(int(10980/b.shape[1]), axis=1) for b in bands]
                    [rio_write("query_B/B%.2i.tif"%(i+1), b[2*x:(2*x+2*h),2*y:(2*y+2*h)]) for i,b in enumerate(upsampled)]
                

                    # paths_A = get_all_L1C_bands(tile, title_l1c_A, "query_A")
                    # rio_write('query_A/rgb.tif', compute_rgb(paths_A)[::4,::4,:] )
                    # paths_B = get_all_L1C_bands(tile, title_l1c_B, "query_B")
                    # rio_write('query_B/rgb.tif', compute_rgb(paths_B)[::4,::4,:] )
                    break
            
            scl_A.close()
            scl_B.close()

            if found_pair:
                #copy to dataset
                pass




# at least 10 percent and less 90 percent clouds
# no NO_DATA tags
# 