import os
import numpy as np
import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from shapely.geometry import Point
import geopandas as gpd

URLS = [
        'http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_rivers_europe.zip',
        'http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_rivers_lake_centerlines.zip',
        'http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip',
        'http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_geography_regions_elevation_points.zip'
        ]

def download_shapes(urls):

    print('Downloading shapefiles from naturalearthdata.com...')

    for url in tqdm(urls):
        zipfile = ZipFile(BytesIO(requests.get(url).content))
        zipfile.extractall('./data')
        
    return print('Saved at ./data folder')

def init_dataset():  

    pre_dataset = pd.DataFrame()
    print('Downloading from atlantides.org...')
    data = pd.read_csv('http://atlantides.org/downloads/pleiades/dumps/pleiades-locations-latest.csv.gz') 
    data2 = pd.read_csv('http://atlantides.org/downloads/pleiades/dumps/pleiades-places-latest.csv.gz')

    data = data[data['reprLatLong'].notnull()]
    data = data[data['timePeriodsRange'].notnull()]
    data.reset_index(drop=True,inplace=True)
    data2 = data2[data2['featureTypes'].notnull()]
    data2.reset_index(drop=True,inplace=True)

    pre_dataset['pid'] = data.pid
    pre_dataset['timePeriodsRange'] = data.timePeriodsRange
    pre_dataset['ftype'] = data.featureType
    pre_dataset['reprLatLong'] = data.reprLatLong

    print('Filling the gaps...')
    for i, reprLatLong in tqdm(data2.reprLatLong.iteritems()):
        pre_dataset.loc[pre_dataset['reprLatLong'] == reprLatLong,'ftype'] = data2['featureTypes'][i]

    pre_dataset.ftype = pre_dataset.ftype.str.split('-').str[0]
    pre_dataset.ftype = pre_dataset.ftype.str.split(',').str[0]
    pre_dataset.reset_index(drop=True,inplace=True)

    #  'A' (1000-550 BC), 'C' (550-330 BC), 'H' (330-30 BC), 'R' (AD 30-300), 'L' (AD 300-640)
    classes = ['A','C','H','R','L']
    for cls in classes:
        pre_dataset['label_{}'.format(cls)] = data.timePeriods.str.contains(cls)

    pre_dataset.to_csv('./data/dataset.csv')

    return print('Saved at ./data folder \n')

def compute_dists(shapefile,geometry):
    dists = np.empty((geometry.shape[0],shapefile.shape[0]))
    tmp = []
    for i in tqdm(range(geometry.shape[0])):
        tmp = shapefile.distance(geometry.iloc[i]).as_matrix()
        dists[i] = tmp
    return dists

def save_dms(dist_dict):
    for key,item in dist_dict.items():
        np.save('./data/{}_dm.npy'.format(key),item)
        
def get_features():

    dataset = pd.read_csv('./data/dataset.csv', index_col = 0)
    elevs = gpd.read_file('./data/ne_10m_geography_regions_elevation_points.shp').sort_values('elevation')
    coasts = gpd.read_file('./data/ne_10m_coastline.shp')
    eu_rivs = gpd.read_file('./data/ne_10m_rivers_europe.shp')
    rivs = gpd.read_file('./data/ne_10m_rivers_lake_centerlines.shp')

    crs = eu_rivs.crs

    rivs = gpd.GeoSeries((rivs.geometry.tolist() + eu_rivs.geometry.tolist()),crs = crs)


    latlist = list(map(float,dataset.reprLatLong.str.split(',').str[0].tolist()))
    lonlist = list(map(float,dataset.reprLatLong.str.split(',').str[1].tolist()))
    dataset['lat'] = latlist
    dataset['lon'] = lonlist

    mainframe = dataset[dataset.ftype.str.contains('settlement') == True].reset_index(drop=True)
    otherframe = dataset[dataset.ftype.str.contains('settlement') == False].reset_index(drop=True)

    dataset.to_csv('./data/dataset.csv')
    mainframe.to_csv('./data/settlements.csv')
    otherframe.to_csv('./data/not_settlements.csv')

    geoms = []
    for i in range(len(latlist)):
        geoms.append(Point(latlist[i],lonlist[i]))

    geometry=gpd.GeoSeries(geoms,crs=crs)
    mainframe = gpd.GeoDataFrame(mainframe,geometry = geometry,crs=crs)

    shapes = [rivs,coasts,elevs]
    names =  ['rivs','coasts','elevations']  

    distance_dict = {}
    print('Computing distances...\n')
    for i,shape in tqdm(enumerate(shapes)):
        distance_dict[names[i]] = compute_dists(shapefile=shape,geometry=geometry)

    save_dms(distance_dict)



if __name__ == '__main__':

    if 'data' not in os.listdir():
        os.makedirs('data')

    init_dataset()
    download_shapes(URLS)
    get_features()
