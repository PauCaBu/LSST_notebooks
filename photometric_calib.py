from lsst.daf.butler import Butler
import lsst.geom
import sep
import numpy as np
import lsst_pyhelper as lp
import pandas as pd
import astropy.units as u
import sep
import pandas as pd
from scipy.optimize import curve_fit
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord
#from astropy import units as u
from astroquery.vizier import Vizier
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge


try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib 
except ImportError:  # Python 2.x
    import httplib 



pixel_to_arcsec = 0.2626 #arcsec/pixel 
detector_nomenclature= {'S29':1, 'S30':2, 'S31':3, 'S28':7, 'S27':6, 'S26':5, 'S25':4, 'S24':12, 'S23':11, 'S22':10, 'S21':9, 'S20':8, 'S19':18, 'S18':17, 'S17':16, 'S16':15, 'S15':14, 'S14':13, 'S13':24, 'S12':23, 'S11':22, 'S10':21, 'S9':20,'S8':19, 'S7':31, 'S6':30, 'S5':29, 'S4':28, 'S3':27, 'S2':26, 'S1':25, 'N29':60, 'N30':61, 'N31':62, 'N28':59, 'N27':58, 'N26':57, 'N25':56, 'N24':55, 'N23':54, 'N22':53, 'N21':52, 'N20':51, 'N19':50, 'N18':49, 'N17':48, 'N16':47, 'N15':46, 'N14':45, 'N13':44, 'N12':43, 'N11':42, 'N10':41, 'N9':40,'N8':39, 'N7':38, 'N6':37, 'N5':36, 'N4':35, 'N3':34, 'N2':33, 'N1':32 }
ccd_name = dict(zip(detector_nomenclature.values(), detector_nomenclature.keys()))
sibling_allcand = pd.read_csv('/home/jahumada/testdata_hits/SIBLING_allcand.csv', index_col=0)

##### this functions comes from the Panstarss DR1 API ###################################### 

def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a cone search of the PS1 catalog
    
    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)
    """
    
    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data)


def ps1search(table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius)
    
    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2).  Note this is required!
    """
    
    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    checklegal(table,release)
    if format not in ("csv","votable","json"):
        raise ValueError("Bad value for format")
    url = f"{baseurl}/{release}/{table}.{format}"
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

# either get or post works
#    r = requests.post(url, data=data)
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text
    
def checklegal(table,release):
    """Checks if this combination of table and release is acceptable
    
    Raises a VelueError exception if there is problem
    """
    
    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))


def ps1metadata(table="mean",release="dr1",
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table
    
    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    baseurl: base URL for the request
    
    Returns an astropy table with columns name, type, description
    """
    
    checklegal(table,release)
    url = f"{baseurl}/{release}/{table}/metadata"
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab


def mastQuery(request):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object

    Returns head,content where head is the response HTTP headers, and content is the returned data
    """
    
    server='mast.stsci.edu'

    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)
    
    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content


def resolve(name):
    """Get the RA and Dec for an object using the MAST name resolver
    
    Parameters
    ----------
    name (str): Name of object

    Returns RA, Dec tuple with position"""

    resolverRequest = {'service':'Mast.Name.Lookup',
                       'params':{'input':name,
                                 'format':'json'
                                },
                      }
    headers,resolvedObjectString = mastQuery(resolverRequest)
    resolvedObject = json.loads(resolvedObjectString)
    # The resolver returns a variety of information about the resolved object, 
    # however for our purposes all we need are the RA and Dec
    try:
        objRa = resolvedObject['resolvedCoordinate'][0]['ra']
        objDec = resolvedObject['resolvedCoordinate'][0]['decl']
    except IndexError as e:
        raise ValueError("Unknown object '{}'".format(name))
    return (objRa, objDec)


def mag_stars_calculation(repo, visit, ccdnum, collection_diff):
    """
    Calculates the magnitude of the stars used for photometric calibration and PSF measurement. 
    
    Input:
    -----
    repo : [str] 
    visit : [int] 
    ccdnum : [int] 
    collection_diff : [int]
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    pixel_to_arcsec = 0.2626 #arcsec/pixel 
    butler = Butler(repo)
    df = pd.DataFrame(columns=['ra', 'dec','Panstarss_dr1_mag', 'src_catalog_LSST_mag', 'calculated_byme_flx','calculated_byme_flx_coadd', 'calculated_byme_mag','seeing','m_inst', 'airmass', 'expoTime'])
    src = butler.get('src',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    table = src.asAstropy()
    mask = (table['calib_photometry_used'] == True) & (table['calib_psf_used']==True)
    phot_table = table[mask]
    phot_table['coord_ra_ddegrees'] = (phot_table['coord_ra']).to(u.degree)
    phot_table['coord_dec_ddegrees'] = (phot_table['coord_dec']).to(u.degree)
    calexp = butler.get('calexp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    coadd = butler.get('goodSeeingDiff_matchedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')


    phot_calexp = calexp.getPhotoCalib()
    
    for i in range(len(phot_table)):
        ra_star = phot_table[i]['coord_ra_ddegrees']
        dec_star = phot_table[i]['coord_dec_ddegrees']
        
        # from src catalog of LSST Science pipelines 
        obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        flux_star_src = phot_table['base_PsfFlux_instFlux'][i]
        src_mag = phot_calexp.instFluxToMagnitude(flux_star_src, obj_pos_2d)

        # from Panstarss catalog
        constraints = {'nDetections.gt':1}
        results = ps1cone(ra_star, dec_star, 0.00028, **constraints)
        tab = ascii.read(results)
        # improve the format
        for filter in 'grizy':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab[col][tab[col] == -999.0] = np.nan
            except KeyError:
                print("{} not found".format(col))
        ps1_mag = float(tab['gMeanPSFMag'][0])
        
        # calculated by me using Source Extractor 
        obj_pos_lsst = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
        obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        wcs = calexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        data = np.asarray(calexp.image.array, dtype='float')
        psf = calexp.getPsf()
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm * pixel_to_arcsec
        pixel_to_arcsec = 0.2626 #arcsec/pixel 
        r = seeing*2
        r/=pixel_to_arcsec
        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], r=r, var = np.asarray(calexp.variance.array, dtype='float'))
        calc_flx = flux[0]*1.05
        calc_mag = phot_calexp.instFluxToMagnitude(calc_flx, obj_pos_2d)

        # calculating in the coadd image
        data_coadd = np.asarray(coadd.image.array, dtype='float')
        flux_coadd, fluxerr_coadd, flagc = sep.sum_circle(data_coadd, [x_pix], [y_pix], r=r, var = np.asarray(coadd.variance.array, dtype='float'))
        calc_flx_coadd = flux_coadd[0]*1.05

        # m_instrumental 
        expoTime = float(calexp.getInfo().getVisitInfo().exposureTime)
        m_instrumental = -2.5*np.log10((flux[0]*1.05)/expoTime)
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        
        df.loc[len(df)] = [ra_star, dec_star, ps1_mag, src_mag, calc_flx, calc_flx_coadd, calc_mag, seeing, m_instrumental, airmass, expoTime]
    
    # doing the photometric calibration:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Panstarss_dr1_mag', 'm_inst'])
    
    #X = [df.m_inst, df.airmass]
    #alpha = curve_fit(Calibration, xdata = X, ydata = df.Panstarss_dr1_mag)[0]
    #print(alpha)
    #m_calib = Calibration(X, alpha[0], alpha[1])
    
    #df['m_calib'] = m_calib
    #df['Z_value'] = alpha[0]
    #df['k_value'] = alpha[1]
    df['Panstarss_counts'] = FromMagToCounts(np.array(df['Panstarss_dr1_mag']))
    pan_counts = np.mean(np.array(df.Panstarss_counts))
    strs_counts = np.mean(np.array(df.calculated_byme_flx))
    strs_counts_coadd = np.mean(np.array(df.calculated_byme_flx_coadd))
    print('stars counts in calexp: ', strs_counts)
    print('stars counts in coadd: ',strs_counts_coadd)
    
    return {'DataFrame' : df, 'panstarss_counts': pan_counts, 'calcbyme_counts' : strs_counts, 'calcbyme_counts_coadd' : strs_counts_coadd}

def mag_stars_calculation2(repo, visit, ccdnum, collection_diff):
    """
    Calculates the magnitude of the stars used for photometric calibration and PSF measurement
    using the coadd image 
    
    Input:
    -----
    repo : [str] 
    visit : [int] 
    ccdnum : [int] 
    collection_diff : [int]
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    pixel_to_arcsec = 0.2626 #arcsec/pixel 
    butler = Butler(repo)
    df = pd.DataFrame(columns=['ra', 'dec','Panstarss_dr1_mag', 'src_catalog_LSST_mag', 'calculated_byme_flx', 'calculated_byme_mag','seeing','m_inst', 'airmass', 'expoTime'])
    src = butler.get('src',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    table = src.asAstropy()
    mask = (table['calib_photometry_used'] == True) & (table['calib_psf_used']==True)
    phot_table = table[mask]
    phot_table['coord_ra_ddegrees'] = (phot_table['coord_ra']).to(u.degree)
    phot_table['coord_dec_ddegrees'] = (phot_table['coord_dec']).to(u.degree)
    calexp = butler.get('calexp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    coadd = butler.get('goodSeeingDiff_matchedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    phot_calexp = calexp.getPhotoCalib()
    
    for i in range(len(phot_table)):
        ra_star = phot_table[i]['coord_ra_ddegrees']
        dec_star = phot_table[i]['coord_dec_ddegrees']
        
        # from src catalog of LSST Science pipelines 
        obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        flux_star_src = phot_table['base_PsfFlux_instFlux'][i]
        src_mag = phot_calexp.instFluxToMagnitude(flux_star_src, obj_pos_2d)

        # from Panstarss catalog
        constraints = {'nDetections.gt':1}
        results = ps1cone(ra_star, dec_star, 0.00028, **constraints)
        tab = ascii.read(results)
        # improve the format
        for filter in 'grizy':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab[col][tab[col] == -999.0] = np.nan
            except KeyError:
                print("{} not found".format(col))
        ps1_mag = float(tab['gMeanPSFMag'][0])
        
        # calculated by me using Source Extractor 
        obj_pos_lsst = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
        obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        wcs = calexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        data = np.asarray(coadd.image.array, dtype='float')
        psf = calexp.getPsf()
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm * pixel_to_arcsec
        pixel_to_arcsec = 0.2626 #arcsec/pixel 
        r = seeing*2
        r/=pixel_to_arcsec
        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], r=r, var = np.asarray(coadd.variance.array, dtype='float'))
        calc_flx = flux[0]*1.05
        calc_mag = phot_calexp.instFluxToMagnitude(calc_flx, obj_pos_2d)
        
        # m_instrumental 
        expoTime = float(calexp.getInfo().getVisitInfo().exposureTime)
        m_instrumental = -2.5*np.log10((flux[0]*1.05)/expoTime)
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        
        df.loc[len(df)] = [ra_star, dec_star, ps1_mag, src_mag, calc_flx, calc_mag, seeing, m_instrumental, airmass, expoTime]
    
    # doing the photometric calibration:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Panstarss_dr1_mag', 'm_inst'])
    
    #X = [df.m_inst, df.airmass]
    #alpha = curve_fit(Calibration, xdata = X, ydata = df.Panstarss_dr1_mag)[0]
    #print(alpha)
    #m_calib = Calibration(X, alpha[0], alpha[1])
    
    #df['m_calib'] = m_calib
    #df['Z_value'] = alpha[0]
    #df['k_value'] = alpha[1]
    df['Panstarss_counts'] = FromMagToCounts(np.array(df['Panstarss_dr1_mag']))
    pan_counts = np.mean(np.array(df.Panstarss_counts))
    strs_counts = np.mean(np.array(df.calculated_byme_flx))
    
    return {'DataFrame' : df, 'panstarss_counts': pan_counts, 'calcbyme_counts' : strs_counts}


def get_mags_from_catalog_gaia(RA, DEC):
    """
    Gets magnitude from gaia catalog
    
    Input:
    -----
    RA
    DEC
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    df = pd.DataFrame(columns=['ra', 'dec','gaia_dr3_mag'])
    
    for i in range(len(RA)):
        ra_star = RA[i]
        dec_star = DEC[i]
       
        # from gaia catalog
        
        c = SkyCoord(ra_star * u.degree, dec_star * u.degree, frame='icrs')
        result = Vizier.query_region(c,
                             radius = 2 * u.arcsec,
                             catalog='I/355/gaiadr3',
                             column_filters={'Gmag': '<25','Var':"!=VARIABLE"})
        gaia_mag = float(result[0]['Gmag'])
        
        df.loc[len(df)] = [ra_star, dec_star, gaia_mag] 
    
    return df

def get_mags_from_catalog_ps1(RA, DEC):
    """
    Gets magnitude from panstarss catalog
    
    Input:
    -----
    RA
    DEC
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    df = pd.DataFrame(columns=['ra', 'dec','ps1_mag'])
    
    for i in range(len(RA)):
        ra_star = RA[i]
        dec_star = DEC[i]
       
        # from Panstarss catalog
        constraints = {'nDetections.gt':1}
        results = ps1cone(ra_star, dec_star, 0.00028, **constraints)
        tab = ascii.read(results)
        # improve the format
        for filter in 'grizy':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab[col][tab[col] == -999.0] = np.nan
            except KeyError:
                print("{} not found".format(col))
        ps1_mag = float(tab['gMeanPSFMag'][0])
        df.loc[len(df)] = [ra_star, dec_star, ps1_mag] 
    
    return df


def get_from_catalog_ps1(RA, DEC):
    """
    Gets magnitude from panstarss catalog
    
    Input:
    -----
    RA
    DEC
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    df = pd.DataFrame(columns=['ra', 'dec', 'gmag', 'g_r', 'g_i'])
    
    for i in range(len(RA)):
        ra_star = RA[i]
        dec_star = DEC[i]
       
        # from Panstarss catalog
        constraints = {'nDetections.gt':1}
        results = ps1cone(ra_star, dec_star, 0.00028, **constraints)
        tab = ascii.read(results)
        # improve the format
        for filter in 'grizy':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab[col][tab[col] == -999.0] = np.nan
            except KeyError:
                print("{} not found".format(col))
        gmag = float(tab['gMeanPSFMag'][0])
        rmag = float(tab['rMeanPSFMag'][0])
        imag = float(tab['iMeanPSFMag'][0])
        df.loc[len(df)] = [ra_star, dec_star, gmag, gmag - rmag, gmag - imag] 
    
    return df

def stars_scaling_calculation(stars_table, repo, visit, ccdnum, collection_diff):
    """
    Calculates the magnitude of the stars used for photometric calibration and PSF measurement. 
    
    Input:
    -----
    repo : [str] 
    visit : [int] 
    ccdnum : [int] 
    collection_diff : [int]
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    pixel_to_arcsec = 0.2626 #arcsec/pixel 
    butler = Butler(repo)
    df = pd.DataFrame(columns=['ra', 'dec','Panstarss_dr1_mag', 'calculated_byme_flx', 'calculated_byme_mag','seeing','m_inst', 'airmass', 'expoTime'])
    src = butler.get('src',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    #table = src.asAstropy()
    #mask = (table['calib_photometry_used'] == True) & (table['calib_psf_used']==True)
    #phot_table = table[mask]
    ra_icrs = np.array(stars_table['RA_ICRS'], dtype=float)#phot_table['coord_ra_ddegrees'] = (phot_table['coord_ra']).to(u.degree)
    dec_icrs = np.array(stars_table['DE_ICRS'], dtype=float) #phot_table['coord_dec_ddegrees'] = (phot_table['coord_dec']).to(u.degree)
    calexp = butler.get('calexp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    coadd = butler.get('goodSeeingDiff_matchedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    
    phot_calexp = calexp.getPhotoCalib()
    
    for i in range(len(stars_table)):
        ra_star = ra_icrs[i]#['coord_ra_ddegrees']
        dec_star = dec_icrs[i]#['coord_dec_ddegrees']
        
        # from src catalog of LSST Science pipelines 
        #obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        #flux_star_src = phot_table['base_PsfFlux_instFlux'][i]
        #src_mag = phot_calexp.instFluxToMagnitude(flux_star_src, obj_pos_2d)

        # from Panstarss catalog
        constraints = {'nDetections.gt':1}
        results = ps1cone(ra_star, dec_star, 0.00028, **constraints)
        tab = ascii.read(results)
        # improve the format
        for filter in 'grizy':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab[col][tab[col] == -999.0] = np.nan
            except KeyError:
                print("{} not found".format(col))
        ps1_mag = float(tab['gMeanPSFMag'][0])
        
        # calculated by me using Source Extractor 
        obj_pos_lsst = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
        obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        wcs = calexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        data = np.asarray(coadd.image.array, dtype='float')
        psf = calexp.getPsf()
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm * pixel_to_arcsec
        pixel_to_arcsec = 0.2626 #arcsec/pixel 
        r = seeing*2
        r/=pixel_to_arcsec
        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], r=r, var = np.asarray(calexp.variance.array, dtype='float'))
        calc_flx = flux[0]*1.05
        calc_mag = phot_calexp.instFluxToMagnitude(calc_flx, obj_pos_2d)
        
        # m_instrumental 
        expoTime = float(calexp.getInfo().getVisitInfo().exposureTime)
        m_instrumental = -2.5*np.log10((flux[0]*1.05)/expoTime)
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        
        df.loc[len(df)] = [ra_star, dec_star, ps1_mag, calc_flx, calc_mag, seeing, m_instrumental, airmass, expoTime]
    
    # doing the photometric calibration:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Panstarss_dr1_mag', 'm_inst'])
    
    #X = [df.m_inst, df.airmass]
    #alpha = curve_fit(Calibration, xdata = X, ydata = df.Panstarss_dr1_mag)[0]
    #print(alpha)
    #m_calib = Calibration(X, alpha[0], alpha[1])
    
    #df['m_calib'] = m_calib
    #df['Z_value'] = alpha[0]
    #df['k_value'] = alpha[1]
    df['Panstarss_counts'] = FromMagToCounts(np.array(df['Panstarss_dr1_mag']))
    pan_counts = np.mean(np.array(df.Panstarss_counts))
    strs_counts = np.mean(np.array(df.calculated_byme_flx))
    
    return {'DataFrame' : df, 'panstarss_counts': pan_counts, 'calcbyme_counts' : strs_counts}



def Calibration(x, Z, k):
    """
    Calibration function of instrumental magnitude to calibrated magnitude:
    
    Input:
    ------
    x : instrumental magnitude and airmass values
    Z : photometric zero point.
    k : atmospheric extinction coefficient
    
    Output:
    ------
    m_calib : calibrated magnitude
    
    """
    m_inst, X = x
    m_calib = m_inst + Z + k*X
    return m_calib


def MagAtOneCountFlux(repo, visit, ccdnum, collection_diff):
    """
    Returns magnitude equal to one count 

    Input:
    ------
    repo : [str] 
    visit : [int] 
    ccdnum : [int] 
    collection_diff : [int]
    
    Output:
    ------
    m_calib : calibrated magnitude
    
    """
    df =  mag_stars_calculation(repo, visit, ccdnum, collection_diff)
    nstars = len(df)
    Z = np.unique(np.array(df.Z_value))
    airmass = np.unique(np.array(df.airmass))
    k = np.unique(np.array(df.k_value))
    expoTime = np.unique(np.array(df.expoTime))
    m_inst = 1/expoTime
    m_calib = Calibration([m_inst, airmass], Z, k)
    print('Using {} stars for photometric calibration'.format(nstars))
    return m_calib[0]

def FromMagToCounts(mag, magzero=24.56):
    '''
    Returns flux counts 
    Mag zero from g band of panstarss : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99
    '''
    return 10**(-0.4*(mag - magzero))

def FluxToMag(flux, fluxerr, magzero=24.3):
    """
    Transform flux and fluxerr to magnitude 
    
    Input
    -----
    flux :
    fluxerr :
    magzero : 

    Output:
    ------
    mag, magerr
    """
    mag = -2.5*np.log10(flux) + magzero
    magerr = np.fabs(-2.5 * fluxerr/(flux * np.log(10)))
    return mag, magerr

def FluxJyToABMag(flux, fluxerr=None):
    """
    Transform flux and fluxerr in Jy to AB magnitude 
    
    Input
    -----
    flux :
    fluxerr :

    Output:
    ------
    magab, magab_err
    """
    magab = -2.5*np.log10(flux) + 8.9
    magab_err = None
    if type(fluxerr)==float:
        magab_err = np.fabs(-2.5 * fluxerr/(flux * np.log(10)))

    return magab, magab_err


def ABMagToFlux(mab):
    f = 10**(-0.4*(mab - 8.90))
    return f


def get_fluxes_from_stars(repo, visit, ccdnum, collection_diff):
    """
    Calculates the magnitude of the stars used for photometric calibration and PSF measurement
    using the coadd image 
    
    Input:
    -----
    repo : [str] 
    visit : [int] 
    ccdnum : [int] 
    collection_diff : [int]
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    pixel_to_arcsec = 0.2626 #arcsec/pixel 
    butler = Butler(repo)
    LSST_stars = lp.Select_table_from_one_exposure(repo, visit, ccdnum, collection_diff, well_subtracted=False)
    LSST_stars_to_pandas = LSST_stars.to_pandas()
    LSST_stars_to_pandas = LSST_stars_to_pandas.reset_index()
    calexp = butler.get('calexp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    coadd = butler.get('goodSeeingDiff_matchedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    df = pd.DataFrame(columns=['ra', 'dec','Panstarss_dr1_mag','Panstarss_dr1_flx', 'calculated_byme_flx', 'seeing','m_inst', 'airmass', 'expoTime'])
    #print('LSST stars df: ', LSST_stars)
    for i in range(len(LSST_stars_to_pandas)):
        ra_star =LSST_stars_to_pandas.loc[i]['coord_ra_ddegrees']
        dec_star = LSST_stars_to_pandas.loc[i]['coord_dec_ddegrees']

        # from Panstarss catalog
        constraints = {'nDetections.gt':1}
        results = ps1cone(ra_star, dec_star, 0.00028, **constraints)
        tab = ascii.read(results)
        # improve the format
        for filter in 'grizy':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab[col][tab[col] == -999.0] = np.nan
            except KeyError:
                print("{} not found".format(col))
        ps1_mag = float(tab['gMeanPSFMag'][0])
        ps1_flux = ABMagToFlux(ps1_mag)*1e9
        
        # calculated by me using Source Extractor 
        obj_pos_lsst = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
        obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        wcs = coadd.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        data = np.asarray(coadd.image.array, dtype='float')
        psf = coadd.getPsf()
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        pixel_to_arcsec = 0.2626 #arcsec/pixel 
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm * pixel_to_arcsec
        r = seeing*2
        r/=pixel_to_arcsec
        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], r=r, var = np.asarray(coadd.variance.array, dtype='float'))
        calc_flx = flux[0]*1.05 # here is to simulate a PSF type calculation of the flux

        # m_instrumental 
        expoTime = float(calexp.getInfo().getVisitInfo().exposureTime)
        m_instrumental = -2.5*np.log10(calc_flx/expoTime)
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        
        df.loc[len(df)] = [ra_star, dec_star, ps1_mag, ps1_flux, calc_flx, seeing, m_instrumental, airmass, expoTime]
    
    # making sure is clean:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Panstarss_dr1_mag', 'm_inst'])
 
    return df 

def DoCalibration(repo, visit, ccdnum, collection_diff):
    stars = get_fluxes_from_stars(repo, visit, ccdnum, collection_diff)
    print('Doing photometric calibration with {} stars'.format(len(stars)))
    
    #ps1_flux = np.array(stars.Panstarss_dr1_flx)#.reshape(-1, 1)
    #my_counts = np.array(stars.calculated_byme_flx)#.reshape(-1, 1)
    #print(ps1_flux)
    #print(my_counts)
    ps1_flux = np.array(stars.Panstarss_dr1_flx)#.reshape(-1, 1)
    my_counts = np.array(stars.calculated_byme_flx).reshape(-1, 1)
    X, y = my_counts, ps1_flux
    #print(ps1_flux)
    #print(my_counts)
    x = np.linspace(X.min(), X.max(), len(X))
    epsilon = 1.35
    huber = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, epsilon=epsilon)
    huber.fit(X, y)
    coef_ = huber.coef_ * x + huber.intercept_
    plt.figure(1, figsize=(10,6))
    plt.subplot(311)
    plt.plot(x, coef_, 'magenta', label="huber loss, %s" % epsilon)
    plt.plot(X, y, '*', color='blue')
    plt.xlim(0,100000)
    plt.xlabel('Counts from PSF aperture', fontsize=17)
    plt.ylabel('PS1 flux [nJy]', fontsize=17)
    plt.title('Calibration found : c {} + {}'.format(huber.coef_, huber.intercept_), fontsize=17)
    plt.legend()
    plt.subplot(313)
    plt.plot(X, coef_ - y, 'purple', label='residuals')
    plt.xlim(0,100000)
    plt.show()    
    return huber.coef_