from lsst.meas.algorithms.htmIndexer import HtmIndexer
import lsst.geom as geom
import numpy as np
import pandas as pd
import wget
import os

def getShards(lon, lat, radius):
    '''
    lon : [float]
    lat : [float]
    radius : [float] Radius of the Region we want to study
    '''
    htm = HtmIndexer(depth=7)  
    shards, onBoundary = htm.getShardIds(geom.SpherePoint(lon*geom.degrees, lat*geom.degrees), radius*geom.degrees)
    return shards

radius = 1.7
field = pd.read_csv('~/testdata_hits/FieldHiTS_Radec.txt', sep=' ')

ra = field['ra']
dec = field['dec']

PATH_TO_DOWNLOAD = 'https://tigress-web.princeton.edu/~pprice/ps1_pv3_3pi_20170110/'

def Download_fields(ra,dec,radius,output_path,download=False):
    '''
    ======
    Input:
    ======
    ra : [ndarray] Right ascention coordinates
    dec : [ndarray] Declination coordinates 
    radius : [float] Radius of the region
    output_path: [string] Directory path for the images to be downloaded
    ======
    Output:
    ======
    Download the Field images and locate them in the folder output_path/
    
    '''
    list_shards = []
    for r, d in zip(ra,dec):
        shard = getShards(r,d,radius)
        for s in shard:
            list_shards.append(s)
    list_shards = np.unique(list_shards)
    print('List of Shards: ',list_shards)
    if download:
        for i in list_shards:
            if not os.path.isfile(output_path + str(i) + '.fits'):
                print('...downloading...')
                wget.download(PATH_TO_DOWNLOAD + str(i) + '.fits', out=output_path)
                print(i, ' succesfully downloaded c:')
            else:
                print(i, ' already downloaded')
                pass
    return

Download_fields(ra,dec,radius,'/home/jahumada/ps1_pv3_3pi_20170110/',download=False)


