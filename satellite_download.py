import os
import subprocess
import numpy as np
import geopandas as gpd
import json
from tqdm import tqdm
import fiona
import struct
import pandas as pd
import json
from shapely.geometry import mapping, shape,asPoint
import ftplib
from ftplib import error_perm
from socket import timeout
from os.path import exists

def get_dlinks(prs):
    """
    Подготовка списка спутниковых изображений для скачивания.
    """
    f = ftplib.FTP('ftp.glcf.umd.edu',timeout = 120)
    f.login()
    full_links = {}
    for i,pr in tqdm(enumerate(prs)):
        full_links[i] = {'DLINKS':[]}
        full_links[i]['PR'] = pr
        try:
            fdir = '/glcf/Landsat/WRS2'+pr
            f.cwd(fdir)
            folder = [x for x in f.nlst() if x.startswith('L')][-1]
            link = '{}/{}'.format(fdir,folder)
            f.cwd(folder)
            full_links[i]['DLINKS'] = ['{}/{}'.format(link,x) for x in f.nlst() if 'tif' in x.lower()]
        except error_perm:
            continue
        except IndexError:
            f.cwd('/')
    return full_links

def download_images(full_links):
    """
    Загрузка изображений с ftp-сервера.
    """
    ddir = './sat_images'
    f = ftplib.FTP('ftp.glcf.umd.edu',timeout = 60)
    f.login()
    for i,item in tqdm(full_links.items()):
        dirname = os.path.join(ddir,item['PR'].replace('/',''))
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for link in item['DLINKS']:
            fname = dirname + '/' + link.split('/')[-1]
            if fname.split('/')[-1].strip('.gz') in os.listdir('/'.join(fname.split('/')[:-1])):
                continue
            try:
                with open(fname,'wb') as fd:
                    f.retrbinary('RETR {}'.format(link), lambda data: fd.write(data),blocksize=8192)
            except timeout:
                    print(i,fname)
                    os.remove(fname)
                    f.close()
                    f = ftplib.FTP('ftp.glcf.umd.edu',timeout = 60)
                    f.login()
    return

## Чистка набора данных.


wrs = gpd.read_file('./wrs2_asc_desc/wrs2_asc_desc.shp',driver='ESRI Shapefile')
wrs = wrs.loc[wrs.MODE == 'D']

names = pd.read_csv('./pleiades_data/pleiades-names.csv')
names = names.loc[names.extent.notnull()]
names = names.loc[names.pid.notnull()]
geometry = names.extent.apply(json.loads).apply(shape)
names = gpd.GeoDataFrame(names,crs=wrs.crs,geometry=geometry)

places = pd.read_csv('./pleiades_data/pleiades-places.csv')
places = places.set_index(places.id)
places = places.loc[names.pid.astype('str').str.extract('(\d+)',expand=False).astype(int).tolist()].drop_duplicates()
places = places.drop(['creators','created','uid','currentVersion','modified','authors',
                  'locationPrecision','geoContext','hasConnectionsWith','connectsWith','path','tags','reprLatLong'],axis=1)
places = places.loc[places.extent.notnull()]

ds = gpd.read_file('totals_ds.shp',driver='ESRI Shapefile')
ds['PATH'] = np.array(ds.PR.str.strip('p').str.split('r').tolist()).astype('int')[:,0]
ds['ROW'] = np.array(ds.PR.str.strip('p').str.split('r').tolist()).astype('int')[:,1]


bins = [-200000,-2000,-750,-400,-150,50,650,2100]
labels = ['archaic','hellenistic','early-roman','pyrrhic-wars','civil-war','roman-empire','post-roman']
places['label'] = pd.cut(places['minDate'],bins=bins,labels=labels,right=True,include_lowest=True)

wrs = wrs.loc[wrs.PATH >= ds.PATH.min()]
wrs = wrs.loc[wrs.ROW >= ds.ROW.min()]


geometry = places.extent.apply(json.loads).apply(shape)
locs = gpd.GeoDataFrame(places,crs=wrs.crs,geometry=geometry)
#locs = locs.loc[locs.geometry.geom_type == 'Point']

conts = {}
for i, poly in tqdm(wrs.geometry.iteritems()):
    conts[str(wrs.loc[i,'PR'])] = []
    for j,point in locs.geometry.iteritems():
        if isinstance(point,float):
            continue
        if point.within(poly):
            conts[str(wrs.loc[i,'PR'])].append(j)

cont = {x:y for x,y in conts.items() if y != []}
cont_lens = {y:len(x) for y,x in cont.items()}
cont_keys = {len(x):y for y,x in cont.items()}
prs = [cont_keys[x] for x in np.sort(list(cont_keys.keys()))[-15:]]
metadata = wrs.loc[wrs.PR.apply(lambda x: str(x) in prs)]
metadata = metadata.PR.astype('str')
prs = '/p' + metadata.str.slice(start=0,stop=3) + '/r' + metadata.str.slice(3)
prs = prs.tolist()

data_record = {x:y for x,y in conts.items() if x in metadata.tolist()}

locs['pathrow'] = None
total_records = pd.DataFrame()

for pr,locs_id in data_record.items():
    record = locs.loc[locs_id]
    record['pathrow'] = '/p{}r{}/'.format(pr[:3],pr[3:])
    total_records = pd.concat([total_records,record])

total_records = total_records.drop(['description','timePeriods','timePeriodsRange'],axis=1)  
total_records = gpd.GeoDataFrame(total_records,crs=wrs.crs)
total_records.geometry = total_records.geometry.representative_point()

total_records.to_file(driver='ESRI Shapefile',filename='last_locations.shp')

dlinks = get_dlinks(prs)