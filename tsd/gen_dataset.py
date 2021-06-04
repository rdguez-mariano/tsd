import sys
sys.path.append("..")
import tsd

from tqdm import tqdm
import datetime
import geojson
import numpy as np                   # numeric linear algebra
import matplotlib.pyplot as plt      # plotting
import rasterio       # read/write geotiffs
import utils          # IO and coordinate system conversion tools
import pandas as pd
import random

from collections import OrderedDict
import json
from tsd import s2_metadata_parser
import multiprocessing
from tsd.get_sentinel2 import get_time_series, get_time_series_metadata, get_time_series_from_metadata


from rasterio.rio import stack
from utils import crop_aoi, rio_write, rio_dtype
import glob
from dateutil.relativedelta import relativedelta
from library import *


# os.environ['COPERNICUS_LOGIN'] = 'rdguez-mariano'
# os.environ['COPERNICUS_PASSWORD'] = 'b3c5e714034282ea5c'

list_of_bands = ["B%.2i"%(i+1) for i in range(12)] + ["B8A"]

def get_all_L1C_bands(tile,title,outdir, flush = False):
    """ Get all bands from the level 1C
    these bands differ in spatial resolution:
    https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
    """
    if flush:
        mkdir("./"+outdir)
        os.system("rm -r ./"+outdir)
        mkdir("./"+outdir)

    metadata = get_time_series_metadata(bands=list_of_bands,
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
    paths = ["./"+fname.replace(".tif", ".jp2") for fname, url in metadata]
    return metadata, paths
    

def get_SCL_band(tile,title,outdir):
    mkdir("./"+outdir)
    os.system("rm -r ./"+outdir)
    mkdir("./"+outdir)
    metadata = get_time_series_metadata(bands=['SCL'],
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
    paths = ["./"+fname.replace(".tif", ".jp2") for fname, url in metadata]
    return metadata, paths[0]

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

def filter_catalogs(catalog1,catalog2):
    newc1, newc2 = [], []
    for p1 in catalog1:
        count = 0
        for p2 in catalog2:
            title_1 = p1['title']
            title_2 = p2['title']
            same_orbit = title_1.split("_")[4]==title_2.split("_")[4]
            dateflag = title_1.split("_")[2]==title_2.split("_")[2]
            if same_orbit and dateflag:
                newc1.append( p1 )
                newc2.append( p2 ) 
                count = count + 1
        assert count<=1, "Error: the assumption for catalogs is not satisfied..." 
    return newc1, newc2    

def save_ordereddict(d,filepath):
    with open(filepath, 'w') as f:
        f.write(json.dumps(list(d.items())))

def load_ordereddict(filepath):
    with open(filepath, 'r') as read_file:
        dvec = json.loads(read_file.read())
    dout = OrderedDict()
    for d in dvec:
        for i in range(int(len(d)/2)):
            dout[d[2*i]] = d[2*i+1]
    return dout

statusfile = "./dataset/status.dat"

random.seed(42)
mkdir("./dataset")

tiles = np.load("tiles.npy")
random.shuffle(tiles)
# tiles = tiles[0:2]
# tiles = ["51QUG","T50RNN"]
# tiles = ["T50RNN"]

if os.path.exists(statusfile):
    tmp = load_ordereddict(statusfile)
    code = tmp["code"]
    stats = tmp["stats"]
    tiles_done = tmp["tiles_done"]
else:
    code = OrderedDict()
    code["vegetation"] = 4
    code["not_vegetated"] = 5
    code["water"] = 6
    code["snow"] = 11
    stats = OrderedDict()
    stats["vegetation"] = 0 
    stats["not_vegetated"] = 0
    stats["water"] = 0
    stats["snow"] = 0
    tiles_done = []

def PreferedClasses(s):
    fields = [k for k,v in s.items()]
    values = [v for k,v in s.items()]
    idxs = np.argsort(values)
    return [fields[i] for i in idxs][:-1],  [code[fields[i]] for i in idxs][:-1] 

def computeScore(crop_A,crop_B, pcode, area):
    mask_A = np.array(crop_A == pcode)
    mask_B = np.array(crop_B == pcode)
    return np.sum(np.logical_or(mask_A, mask_B)) / area

def check_tile_season(tile,s_i):
    """ usage: 
    if check_tile_season(tile,s_i):
            continue
    """
    metadata = os.path.exists("./dataset/%s/S%i/metadata.log" % (tile,s_i))                    
    files_t_0 = [os.path.exists("./dataset/%s/S%i/t_0/%s.tif"%(tile,s_i,str_b)) for str_b in list_of_bands]
    files_t_1 = [os.path.exists("./dataset/%s/S%i/t_1/%s.tif"%(tile,s_i,str_b)) for str_b in list_of_bands]
    if np.all(files_t_0) and np.all(files_t_1) and metadata:
        return True
    else:
        return False

for tile in tqdm(tiles):
    if tile in tiles_done:
        continue
    first_date = datetime.datetime(2019, 12, 1)
    for s_i in tqdm(range(4)):
        start_date = first_date + relativedelta(months=s_i*3)
        end_date = first_date + relativedelta(months=(s_i+1)*3) - relativedelta(days=1)

        # start_date = datetime.datetime(2020, 1, 1)
        # end_date = datetime.datetime(2020, 1, 31)

        # run the query
        image_catalog_l1c = tsd.get_sentinel2.search(aoi=None,tile_id=tile,product_type="L1C",start_date=start_date, end_date=end_date, api='scihub')
        image_catalog_l2a = tsd.get_sentinel2.search(aoi=None,tile_id=tile,product_type="L2A",start_date=start_date, end_date=end_date, api='scihub')

        image_catalog_l1c,image_catalog_l2a = filter_catalogs(image_catalog_l1c,image_catalog_l2a)

        if len(image_catalog_l1c)==0 or len(image_catalog_l1c)!=len(image_catalog_l2a):
            print("lenghts of two catalogs: %i - %i"% (len(image_catalog_l1c),len(image_catalog_l2a)))
            continue
        
        indices = []
        for i in range(len(image_catalog_l2a)-1):
            for j in np.arange(i+1,len(image_catalog_l2a)):
                date_A = image_catalog_l2a[i]['date']
                date_B = image_catalog_l2a[j]['date']
                rd = relativedelta(date_B, date_A)
                same_orbit = image_catalog_l2a[i]['title'].split("_")[4]==image_catalog_l2a[j]['title'].split("_")[4]
                okdate = rd.years==0 and rd.months==0 and np.abs(rd.days) <= 10 and np.abs(rd.hours+rd.minutes/60)<2
                if okdate and same_orbit:
                    indices.append( [i,j] )

        if len(indices)==0:
            continue
        print(len(indices))
        idxs = list(range(len(indices)-1))
        random.shuffle(idxs)
        sclflag = False
        found_pair = False
        for idx in idxs:
            idx_A, idx_B = indices[idx]

            title_l1c_A = image_catalog_l1c[idx_A]['title']
            title_l2a_A = image_catalog_l2a[idx_A]['title']
            date_A = image_catalog_l1c[idx_A]['date']
            
            title_l1c_B = image_catalog_l1c[idx_B]['title']
            title_l2a_B = image_catalog_l2a[idx_B]['title']
            date_B = image_catalog_l1c[idx_B]['date']
            
            same_orbit = title_l1c_A.split("_")[4]==title_l2a_A.split("_")[4] and title_l1c_B.split("_")[4]==title_l2a_B.split("_")[4]
            okboth = title_l1c_A.split("_")[2]==title_l2a_A.split("_")[2] and title_l1c_B.split("_")[2]==title_l2a_B.split("_")[2]
            rd = relativedelta(date_B, date_A)
            okdate = rd.years==0 and rd.months==0 and np.abs(rd.days) <= 10 and np.abs(rd.hours+rd.minutes/60)<2
            if not (okboth and okdate and same_orbit):
                print("Error with image pair! continuing...")
                continue
            else:
                try:
                    metadata_scl_A, path_scl_A = get_SCL_band(tile, title_l2a_A, "query_A")
                    metadata_scl_B, path_scl_B = get_SCL_band(tile, title_l2a_B, "query_B")
                    metadata = metadata_scl_A + metadata_scl_B
                    get_time_series_from_metadata(metadata, pool_type='processes', parallel_downloads=2)
                    scl_A = rasterio.open(path_scl_A, "r")
                    scl_B = rasterio.open(path_scl_B, "r")
                except:
                    print("Corrupted pair of slc...")
                    continue

            print(title_l2a_A,title_l2a_B)        
            
            cmA = scl_A.read(True)
            # rio_write('query_A/cloud_mask.tif', compute_thumbnail(cmA,percentiles=0,downsamplestep=2) )
            rio_write('query_A/cloud_mask.tif', cmA[::2,::2] )
            cmB = scl_B.read(True)
            # rio_write('query_B/cloud_mask.tif', compute_thumbnail(cmB,percentiles=0,downsamplestep=2) )
            rio_write('query_B/cloud_mask.tif', cmB[::2,::2] )
            
            secondNiters = 1000
            pclass_list, pcode_list = PreferedClasses(stats)
            for pclass, pcode in zip(pclass_list,pcode_list):
                if found_pair:
                        break
                for i in range(300):
                    h, w = 256, 256
                    x, y = random.randint(0,cmA.shape[0]-h), random.randint(0,cmA.shape[1]-w)
                    crop_A = cmA[x:(x+h),y:(y+w)]
                    crop_B = cmB[x:(x+h),y:(y+w)]
                    clouds_A = np.array(crop_A == 8) + np.array(crop_A == 9) #+ np.array(crop_A == 10)
                    clouds_B = np.array(crop_B == 8) + np.array(crop_B == 9) #+ np.array(crop_B == 10)
                    
                    mask_okflag = computeScore(crop_A,crop_B, pcode, h*w)>0.05
                    if not mask_okflag:
                        continue
                    crop_okflag = np.all( np.array(crop_A != 0) * np.array(crop_B != 0) * np.array(crop_A != 1) * np.array(crop_B != 1) )
                    if not crop_okflag:
                        continue
                    if np.sum(np.logical_or(clouds_A, clouds_B))>0.2*h*w and np.sum(np.logical_or(clouds_A, clouds_B))<0.8*h*w \
                        and np.sum(np.logical_and(clouds_A, clouds_B))<0.05*h*w:
                    # if np.sum(clouds_A)<0.01*h*w  and np.sum(clouds_B)>0.1*h*w and np.sum(clouds_B)<0.9*h*w:
                        found_pair = True
                        # This pair might be good to save

                        mkdir("./dataset/"+tile)
                        mkdir("./dataset/"+tile+"/S%i"%s_i)
                        mkdir("./dataset/"+tile+"/S%i/t_0"%s_i)
                        mkdir("./dataset/"+tile+"/S%i/t_1"%s_i)
                        rio_write("./dataset/%s/S%i/t_0/scl_mask.tif"%(tile,s_i), crop_A.repeat(2, axis=0).repeat(2, axis=1))
                        rio_write("./dataset/%s/S%i/t_1/scl_mask.tif"%(tile,s_i), crop_B.repeat(2, axis=0).repeat(2, axis=1))
                        os.system("echo \"title_t_0 = %s\" > ./dataset/%s/S%i/metadata.log" % (title_l1c_A,tile,s_i) )
                        os.system("echo \"title_t_1 = %s\" >> ./dataset/%s/S%i/metadata.log" % (title_l1c_B,tile,s_i) )
                        os.system("echo \"top_left_coordinates = (%i, %i)\" >> ./dataset/%s/S%i/metadata.log" % (x,y,tile,s_i) )
                        
                        # equilize all band shapes and save them
                        metadata_A, paths_A = get_all_L1C_bands(tile, title_l1c_A, "query_A")
                        metadata_B, paths_B = get_all_L1C_bands(tile, title_l1c_B, "query_B")
                        metadata = metadata_A + metadata_B
                        get_time_series_from_metadata(metadata, pool_type='processes', parallel_downloads=26, timeout=20*60)
                        bands = [rasterio.open(p, "r").read(1) for p in paths_A]
                        upsampled = [b.repeat(int(10980/b.shape[0]), axis=0).repeat(int(10980/b.shape[1]), axis=1) for b in bands]
                        [rio_write("./dataset/%s/S%i/t_0/%s.tif"%(tile,s_i,str_b), b[2*x:(2*x+2*h),2*y:(2*y+2*h)]) for str_b,b in zip(list_of_bands,upsampled)]

                        bands = [rasterio.open(p, "r").read(1) for p in paths_B]
                        upsampled = [b.repeat(int(10980/b.shape[0]), axis=0).repeat(int(10980/b.shape[1]), axis=1) for b in bands]
                        [rio_write("./dataset/%s/S%i/t_1/%s.tif"%(tile,s_i,str_b), b[2*x:(2*x+2*h),2*y:(2*y+2*h)]) for str_b,b in zip(list_of_bands,upsampled)]
                        secondNiters = -1
                        for tclass, tcode in zip(pclass_list,pcode_list):
                            stats[tclass] = stats[tclass] + computeScore(crop_A,crop_B, tcode, h*w)
                        break

            for i in range(secondNiters):
                h, w = 256, 256
                x, y = random.randint(0,cmA.shape[0]-h), random.randint(0,cmA.shape[1]-w)
                crop_A = cmA[x:(x+h),y:(y+w)]
                crop_B = cmB[x:(x+h),y:(y+w)]
                clouds_A = np.array(crop_A == 8) + np.array(crop_A == 9) #+ np.array(crop_A == 10)
                clouds_B = np.array(crop_B == 8) + np.array(crop_B == 9) #+ np.array(crop_B == 10)
                crop_okflag = np.all( np.array(crop_A != 0) * np.array(crop_B != 0) * np.array(crop_A != 1) * np.array(crop_B != 1) )
                if not crop_okflag:
                    continue
                if np.sum(np.logical_or(clouds_A, clouds_B))>0.2*h*w and np.sum(np.logical_or(clouds_A, clouds_B))<0.8*h*w \
                    and np.sum(np.logical_and(clouds_A, clouds_B))<0.05*h*w:
                # if np.sum(clouds_A)<0.01*h*w  and np.sum(clouds_B)>0.1*h*w and np.sum(clouds_B)<0.9*h*w:
                    found_pair = True
                    # This pair might be good to save

                    mkdir("./dataset/"+tile)
                    mkdir("./dataset/"+tile+"/S%i"%s_i)
                    mkdir("./dataset/"+tile+"/S%i/t_0"%s_i)
                    mkdir("./dataset/"+tile+"/S%i/t_1"%s_i)
                    rio_write("./dataset/%s/S%i/t_0/scl_mask.tif"%(tile,s_i), crop_A.repeat(2, axis=0).repeat(2, axis=1))
                    rio_write("./dataset/%s/S%i/t_1/scl_mask.tif"%(tile,s_i), crop_B.repeat(2, axis=0).repeat(2, axis=1))
                    os.system("echo \"title_t_0 = %s\" > ./dataset/%s/S%i/metadata.log" % (title_l1c_A,tile,s_i) )
                    os.system("echo \"title_t_1 = %s\" >> ./dataset/%s/S%i/metadata.log" % (title_l1c_B,tile,s_i) )
                    os.system("echo \"top_left_coordinates = (%i, %i)\" >> ./dataset/%s/S%i/metadata.log" % (x,y,tile,s_i) )
                    
                    # equilize all band shapes and save them
                    metadata_A, paths_A = get_all_L1C_bands(tile, title_l1c_A, "query_A")
                    metadata_B, paths_B = get_all_L1C_bands(tile, title_l1c_B, "query_B")
                    metadata = metadata_A + metadata_B
                    get_time_series_from_metadata(metadata, pool_type='processes', parallel_downloads=26, timeout=20*60)
                    
                    bands = [rasterio.open(p, "r").read(1) for p in paths_A]
                    upsampled = [b.repeat(int(10980/b.shape[0]), axis=0).repeat(int(10980/b.shape[1]), axis=1) for b in bands]
                    [rio_write("./dataset/%s/S%i/t_0/%s.tif"%(tile,s_i,str_b), b[2*x:(2*x+2*h),2*y:(2*y+2*h)]) for str_b,b in zip(list_of_bands,upsampled)]

                    bands = [rasterio.open(p, "r").read(1) for p in paths_B]
                    upsampled = [b.repeat(int(10980/b.shape[0]), axis=0).repeat(int(10980/b.shape[1]), axis=1) for b in bands]
                    [rio_write("./dataset/%s/S%i/t_1/%s.tif"%(tile,s_i,str_b), b[2*x:(2*x+2*h),2*y:(2*y+2*h)]) for str_b,b in zip(list_of_bands,upsampled)]
                    
                    for tclass, tcode in zip(pclass_list,pcode_list):
                            stats[tclass] = stats[tclass] + computeScore(crop_A,crop_B, tcode, h*w)
                    break
            
            scl_A.close()
            scl_B.close()
            
            if found_pair:
                break
    
    tiles_done.append(tile)
    tmp = OrderedDict()
    tmp["code"] = code
    tmp["stats"] = stats
    tmp["tiles_done"] = tiles_done
    save_ordereddict(tmp, statusfile)