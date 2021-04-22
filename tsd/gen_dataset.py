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


from tsd import s2_metadata_parser
import multiprocessing
from tsd.get_sentinel2 import get_time_series


from rasterio.rio import stack
from utils import rio_write, rio_dtype
import glob
from dateutil.relativedelta import relativedelta
from library import *

mkdir("./dataset")

tiles = np.load("tiles.npy")
tiles = [tiles[0], "51QUG"]

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

        assert len(image_catalog_l1c)==len(image_catalog_l2a), "Error with: %i == %i" % (len(image_catalog_l1c),len(image_catalog_l2a))
        if len(image_catalog_l1c)==0:
            continue

        idx = 0
        title_l1c = image_catalog_l1c[idx]['title']
        title_l2a = image_catalog_l2a[idx]['title']
        assert title_l1c.split("_")[2]==title_l2a.split("_")[2]


        print(title_l1c,title_l2a)
        mkdir("./test")
        os.system("rm -r ./test")
        mkdir("./test")

        # get_time_series(bands=['B02', 'B03', 'B04'],
        #                 tile_id="51QUG",
        #                 title= title_l1c+'.SAFE',
        #                 out_dir="test",
        #                 api='scihub',
        #                 mirror='gcloud',
        #                 no_crop=True,
        #                 product_type='L1C',
        #                 cloud_masks=False,
        #                 parallel_downloads=multiprocessing.cpu_count()
        #                 )


        # get_time_series(bands=['SCL'],
        #                 tile_id="51QUG",
        #                 title= title_l2a+'.SAFE',
        #                 out_dir="test",
        #                 api='scihub',
        #                 mirror='gcloud',
        #                 no_crop=True,
        #                 product_type='L2A',
        #                 cloud_masks=False,
        #                 parallel_downloads=multiprocessing.cpu_count()
        #                 )


        # path_b02 = glob.glob("./test/*B02.jp2")[0]
        # path_b03 = glob.glob("./test/*B03.jp2")[0]
        # path_b04 = glob.glob("./test/*B04.jp2")[0]
        # path_scl = glob.glob("./test/*SCL.jp2")[0]
        # b02 = rasterio.open(path_b02, "r")
        # b03 = rasterio.open(path_b03, "r")
        # b04 = rasterio.open(path_b04, "r")
        # scl = rasterio.open(path_scl, "r")

        # upsample = b02.shape[0]/scl.shape[0] 

        # rgb = np.stack((b04.read(True),b03.read(True),b02.read(True)),axis=-1)

        # scl = scl.read(True)
        # mh_prob_clouds = np.logical_or(scl == 8, scl == 0).astype(np.uint8) * 150
        # cirrus = (scl == 10).astype(np.uint8) * 50
        # snow = (scl == 11).astype(np.uint8) * 255
        # cloud_mask = mh_prob_clouds + cirrus
        # cloud_mask = cloud_mask.repeat(2, axis=0).repeat(2, axis=1)

        # def simple_equalization_8bit(im, percentiles=5):
        #     ''' im is a numpy array
        #         returns a numpy array
        #     '''
        #     mi, ma = np.percentile(im.flatten(), (percentiles,100-percentiles))
        #     im = np.minimum(np.maximum(im,mi), ma) # clip
        #     im = (im-mi)/(ma-mi)*255.0   # scale
        #     im = im.astype(np.uint8)
        #     return im   # return equalized image

        # rio_write('test/rgb.tif', rgb )
        # rio_write('test/cloud_mask.tif', cloud_mask )