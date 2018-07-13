import subprocess
import numpy as np
import sh
from sh import gunzip,ErrorReturnCode
import geopandas as gpd
import json
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from tqdm import tqdm_notebook
import fiona
import struct
import matplotlib.image as img
import shutil
import rasterio
import os
import pandas as pd

## Набор утилит для работы с геопространственными снимками.

def unzip_all(dpath = '/home/elmirakh/sat_images/'):
    """Массовая распаковка"""
    zip_dirs = [dpath + x + '/' for x in os.listdir(dpath)]
    cnt = 0
    for zdir in tqdm(zip_dirs):
        for path_to_zip_file in os.listdir(zdir):
            if not path_to_zip_file.endswith('.gz'):
                continue
            path_to_zip_file = os.path.join(zdir,path_to_zip_file)
            try:
                gunzip(path_to_zip_file)
                cnt += 1
            except ErrorReturnCode:
                tqdm.write(path_to_zip_file)
                os.remove(path_to_zip_file)
    tqdm.write('Unzipped {} files succesfully'.format(cnt))
    return 

def check_completeness(idnts,dpath = '.'):
    """Проверка наличия всех необходимых файлов"""
    assert isinstance(idnts,list)
    dirs = [os.path.abspath(x) for x in os.listdir(dpath) if os.path.isdir(x)]
    
    for idn in idnts:
        for curdir in dirs:
            check = False
            if not any([name.endswith('.TIF') for name in os.listdir(curdir)]):
                continue
            if not any([idn in name for name in os.listdir(curdir)]):
                tqdm.write(curdir)
                shutil.rmtree(curdir,ignore_errors=True)
    return

def get_all_paths(ddir ='/home/elmirakh/sat_images/',include=['B80','B70','B50','B40','B30','B20','B10']):
    """Получения списка загруженных снимков"""
    paths = {}
    for dname in os.listdir(ddir):
        if not any([x.endswith('.TIF') for x in os.listdir(ddir + dname)]):
            continue
        paths[ddir + dname + '/'] = { \
            x.split('_')[-1].rstrip('.TIF'):x for x in os.listdir(ddir + dname) if  x.endswith('.TIF') \
            }
            
            
    return paths

def merge(paths,bands=['B30','B20','B10'],suffix='RGB'):
    """Построение композитного изображения из отдельных каналов"""
    for dname,files in tqdm(paths.items()):
        try:
            out_name = dname + files[bands[0]].split('_B')[0][3:] + '_{}.TIF'.format(suffix)
            if out_name in os.listdir():
                continue
            file_list = [dname + files[bands[0]],dname + files[bands[1]],dname + files[bands[2]]]
        except KeyError:
            continue
        merge = ["gdal_merge.py", "-o", out_name, '-co','PHOTOMETRIC=RGB',\
                '-tap', '-separate', file_list[0], file_list[1], file_list[2]]
        print(' '.join(merge))
        try:
            subprocess.check_call(merge)
        except subprocess.CalledProcessError:
            os.remove(out_name)
    return
        
def contrast_stretch(paths,suffix='RGB'):
    """Изменение контраста изображения"""
    for dname,files in tqdm_notebook(paths.items()):
        if not suffix in list(files.keys()):
            continue    
            
        target_raster = dname + files[suffix]
    
        cmd = ['gdal_contrast_stretch', '-ndv', '0', '-percentile-range', '0.01', '0.99',
               target_raster,target_raster]
        subprocess.check_output(cmd)
    return
     
    
def pansharpen_adv(paths,suffix='B321'):
    """Панхроматическое увеличение пространственного разрешения (паншарпенинг)"""
    for dname,files in tqdm(paths.items()):    
        pan_raster = dname + files['B80'] 
        target_raster = dname + files[suffix]
        out_raster = target_raster.split(suffix)[0] + 'pan{}.TIF'.format(suffix)
        
        cmd = ['gdal_landsat_pansharp',
            '-rgb', target_raster,
            '-lum', dname + files['B20'],'0.25',
            '-lum',dname + files['B30'], '0.23',
            '-lum', dname + files['B40'],'0.52',
            '-ndv','0',
            '-pan', pan_raster, '-o', out_raster]
        
        tqdm.write (' '.join(cmd))
        subprocess.check_output(cmd)
    return
        
        
def pansharpen(paths,suffix='B742'):
    """Паншарпенинг другим способом"""
    for dname,files in tqdm(paths.items()):
        if not suffix in list(files.keys()) or not 'B80' in list(files.keys()):
            continue
            
        pan_raster = dname + files['B80'] 
        target_raster = dname + files[suffix]
        out_raster = target_raster.split(suffix)[0] + 'pan{}.TIF'.format(suffix)

        cmd = ['gdal_landsat_pansharp',
               '-rgb', target_raster,
               '-pan', pan_raster, 
               '-ndv','0',
               '-o', out_raster]
        
        subprocess.check_output(cmd)
    return

def clean(paths,tags=['RGB','TOPO','IR','GBI']):
    """Удаление ненужных снимков"""
    for dname,files in tqdm(paths.items()):
        for suffix in tags:
            if not suffix in list(files.keys()):
                continue
            target_raster = dname + files[suffix]
            os.remove(target_raster)
    return
    
def downsample(paths,suffix='RGB'):
    """Даунсамплинг изображения"""
    for dname,files in tqdm(paths.items()):
        if not suffix in list(files.keys()):
            continue
            
        target_raster = dname + files[suffix]
        dst = dname + files[suffix].replace(suffix,suffix.lower())
        cmd = ["gdal_translate",'-strict','-r','cubic',
               '-outsize','12000','12000','-ot','Byte','-of','GTiff',
               target_raster,dst]
        tqdm.write (' '.join(cmd))
        subprocess.check_output(cmd)
    return