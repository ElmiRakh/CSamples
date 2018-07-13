import sys
import os
import numpy as np
import rasterio
import h5py
import warnings
import scipy.ndimage as ndimage
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_CRS = fiona.crs.from_epsg(4326)
imdir = os.path.join(dir_path,'sat_images')
paths = [os.path.join(imdir,x) for x in os.listdir(imdir)]

path_A = ''
path_B = ''
img_folder = ''
name_A = ''
name_B = ''

def get_image_names(records,combination_type='RGB'):
    """Подготовка списка путей до изображений"""
    records[combination_type] = np.NaN
    for r_id, record in records.iterrows():
        rname = check_path(record,combination_type)
        if not rname:
            continue
        records.loc[r_id, combination_type] = rname
        
    records['full_path'] = homedir + records['pathrow'] + records[combination_type]
    records.dropna(inplace=True,subset=['full_path',])
    return records

def check_path(record,combination_type='RGB'):
    """Проверка наличия изображения по указанному пути"""
    rdir = homedir + record['pathrow']
    im_list = [x for x in os.listdir(rdir) if '_' + combination_type in x]
    if im_list:
        return im_list[0]
    else:
        return None

def save_record(records,out_path):
    """Сохранение csv файла"""
    if not os.path.exists('./data'):
        os.mkdir('data')
    records.to_csv('./data/{}'.format(out_path))    
    return
        
def open_record(recordpath):
    """Открытие файла формата ESRI Shapefile"""
    records = gpd.read_file(recordpath,driver='ESRI Shapefile')  
    return records

def find_pixel_coord(records,combination_type='RGB'):
    """Поиск референсированного пространственными координатами участка спутникого изображения"""
    records['detections'] = np.NaN
    records['xsize'] = 0
    records['ysize'] = 0
    
    rasters = np.unique(records.loc[:,combination_type].tolist())
    
    for rname in rasters:
        tqdm.write(rname)
        geoms = records.loc[records[combination_type] == rname]
        
        dpath = geoms['full_path'].tolist()[0]

        xs,ys = geoms.geometry.x.tolist(),geoms.geometry.y.tolist()
        
        with rasterio.open(dpath) as src:
            xs_,ys_ = rasterio.warp.transform(BASE_CRS, src.crs,xs,ys)
            assert len(xs_) == len(ys_)
            transformed = [src.index(xs_[i],ys_[i]) for i in range(len(xs_))]
            transformed = ['{}:{}'.format(x,y) for x,y in transformed]
            records.loc[geoms.index,'detections'] = transformed
            records.loc[geoms.index,'xsize'] = src.shape[0]
            records.loc[geoms.index,'ysize'] = src.shape[1]
            
    return records

def create_labels(records,bins,labels,sizes):
    """Создание меток"""
    records['label'] = pd.cut(records['minDate'],bins=bins,labels=labels,right=True,include_lowest=True)
    records['class_id'] = pd.cut(records['minDate'],bins=bins,labels=range(1,len(labels)+1), right=True,include_lowest=True)
    records['size'] = 0

    for i,label in enumerate(labels):
        records.loc[records['label'] == label,'size'] = sizes[i]

    return records

def genpaths(dataset,suffix=['b742','b321']):
    first = list(sorted(os.listdir(os.path.join(dataset,suffix[0]))))
    second = list(sorted(os.listdir(os.path.join(dataset,suffix[1]))))
    assert len(first) == len(second)
    for i in range(len(first)):
        print('pair num {}'.format(i))
        yield first[i],second[i]

def create_img_folder_paired(path_A,path_B,img_folder,data_name_A,data_name_B):
    """
    Созданиие обучающей выборки путем разбивки парных изображений
    на соответствующие куски одинакого размера
    """
    from skimage.io import imread,imsave
    from sklearn.feature_extraction import image
    
    def _read_raster_image(impath):
        """Чтение изображения формата GeoTIFF"""
        return imread(impath,plugin='tifffile').astype(np.uint8)
    
    def _filter_patches(patches, min_mean=0.0, min_std=0.0,max_mean=255.):
        """
        Фильтрация по среднему значению и стандартному отклонению изображений.
        """
        patchdim = np.prod(patches.shape[1:])
        patchvectors = patches.reshape(patches.shape[0], patchdim)
        means =  patchvectors.mean(axis=1)
        stdevs = patchvectors.std(axis=1)
        indices = np.logical_and(np.logical_and((means > min_mean),(stdevs > min_std)),(means < max_mean))
        return indices
    
    def _save_patches(patches,path):
        patches = np.split(patches,patches.shape[0])
        for i,patch in enumerate(patches):
            imname = path.split('.TIF')[0] + '_' + str(i)+'.png'
            imsave(imname,np.squeeze(patch))
    
    if not os.path.exists(path_A):
        os.mkdir(path_A)
    if not os.path.exists(path_B):
        os.mkdir(path_B)  
    
    for img_name1, img_name2 in genpaths(img_folder,suffix=[data_name_A,data_name_B]):
        
        img1 = _read_raster_image(os.path.join(os.path.join(img_folder,data_name_A),img_name1))
        img1 = image.extract_patches_2d(img1,patch_size = (256,256),max_patches=250,random_state=0)
        
        img2 = _read_raster_image(os.path.join(os.path.join(img_folder,data_name_B),img_name2))
        print(img1.shape)
        img2 = image.extract_patches_2d(img2,patch_size = (256,256),max_patches=250,random_state=0)
        print(img2.shape)
        
        filter_inds1 = _filter_patches(img1,min_mean=48.,min_std=8.,max_mean=224.)
        filter_inds2 = _filter_patches(img2,min_mean=48.,min_std=8.,max_mean=224.)
        filter_inds = np.logical_and(filter_inds1,filter_inds2)
        
        img1 = img1[filter_inds]
        _save_patches(img1,os.path.join(path_A,img_name1))
        print('post-filtered shape A',img1.shape)
        img2 = img2[filter_inds]
        _save_patches(img2,os.path.join(path_B,img_name2))
        print('post-filtered shape B',img2.shape)