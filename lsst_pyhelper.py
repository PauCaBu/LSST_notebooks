from platform import mac_ver
import platform
import plotly.express as px
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from pydoc import source_synopsis
from statistics import median
from lsst.pipe.tasks.imageDifference import ImageDifferenceTask
from lsst.ip.diffim import ZogyImagePsfMatchTask as zt
import lsst.afw.display as afwDisplay
from lsst.daf.butler import Butler
import matplotlib.pyplot as plt
import lsst.geom
from lsst.meas.algorithms.detection import *
import astropy.units as u 
import lsst.afw.image as afwImage
from astropy.coordinates import SkyCoord
import matplotlib
from astropy import units as u
from astropy.time import Time
import numpy as np
import pandas as pd
import fitsio
import random
import sep
import os
from astropy.io import fits
import sys 
from astroquery.vizier import Vizier
from photutils.centroids import centroid_sources
from photutils.centroids import centroid_com
from photutils.detection import find_peaks
from sklearn import *
import photometric_calib as pc
from astropy.table import Table, join, Column
import decimal
import seaborn as sns
from scipy import special
import scipy as sp
import scipy.spatial
from scipy.spatial.distance import cdist
from lsst.pipe.tasks.warpAndPsfMatch import WarpAndPsfMatchTask


import matplotlib as mpl
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve
import pickle
from photutils.centroids import centroid_2dg
from scipy import ndimage 
import scipy
from photutils.psf import create_matching_kernel 
from photutils.psf import HanningWindow, TopHatWindow, SplitCosineBellWindow
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.colors as mcolors
from astropy.convolution import AiryDisk2DKernel
from scipy.optimize import minimize
import pyvo as vo
from pathlib import Path
from astroquery.mast import Mast
from photutils.utils._convolution import _filter_data
import itertools as it

# To copy Jorges way
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from photutils import Background2D, MedianBackground
from photutils import aperture_photometry, CircularAperture
from scipy.spatial import cKDTree
#from kernel_jorge_decompiled import * 

sys.path.append('/home/pcaceres/kkernel/lib/')
sys.path.append('/home/pcaceres/kkernel/etc/')

from kkernel import *
#from sklearn.preprocessing import Normalize


bblue='#0827F5'
dark_purple = '#2B018E'
lilac='#a37ed4'
neon_green = '#00FF00'
arcsec_to_pixel=0.27#26
main_root = '/home/pcaceres'
files_to_sibling = 'SIBLING_sources_usingMPfilter_andPCB_comparison.csv'
sibling_dataset = pd.read_csv(main_root + '/LSST_notebooks/'+ files_to_sibling, index_col='internalID')


detector_nomenclature= {'S29':1, 'S30':2, 'S31':3, 'S28':7, 'S27':6, 'S26':5, 'S25':4, 'S24':12, 'S23':11, 'S22':10, 'S21':9, 'S20':8, 'S19':18, 'S18':17, 'S17':16, 'S16':15, 'S15':14, 'S14':13, 'S13':24, 'S12':23, 'S11':22, 'S10':21, 'S9':20,'S8':19, 'S7':31, 'S6':30, 'S5':29, 'S4':28, 'S3':27, 'S2':26, 'S1':25, 'N29':60, 'N30':61, 'N31':62, 'N28':59, 'N27':58, 'N26':57, 'N25':56, 'N24':55, 'N23':54, 'N22':53, 'N21':52, 'N20':51, 'N19':50, 'N18':49, 'N17':48, 'N16':47, 'N15':46, 'N14':45, 'N13':44, 'N12':43, 'N11':42, 'N10':41, 'N9':40,'N8':39, 'N7':38, 'N6':37, 'N5':36, 'N4':35, 'N3':34, 'N2':33, 'N1':32 }
ccd_name = dict(zip(detector_nomenclature.values(), detector_nomenclature.keys()))
sibling_allcand = pd.read_csv('{}/HiTS_data/SIBLING_allcand.csv'.format(main_root), index_col=0)
Blind15A_26_magzero_outputs = pd.read_csv('{}/LSST_notebooks/output_magzeros.csv'.format(main_root))
main_path = '{}/LSST_notebooks/'.format(main_root)


def product_exists(repo, field, ccd_num, collection, etype='calexp', instrument='DECam', btime = 20150219):
    '''
    check if product exists
    returns 1 if it exists, and 0 if not

    Input:
    -----
    repo [string] : 
    field [string] : 
    ccd_num : 
    collection : 
    etype : 
    instrument : 

    output:
    -------
    [int] : 1 if it exists, 0 if not

    '''

    butler = Butler(repo)
    
    try:
        data = get_all_exposures(repo, 'science')
        visit = list(data[(data['target_name']=='{}'.format(field)) & (data['day_obs']<btime)].exposure)[0]
        exp = butler.get(etype, detector=detector_nomenclature[ccd_num], visit=visit, collections=collection, instrument=instrument)
        return 1
    except: 
        return 0

 

def get_all_exposures(repo, obs_type, instrument='DECam'):
    """
    Gets all the calibrated exposure from a butler REPO as a pandas DataFrame
    
    -----
    Input
    -----
    repo : [string] directory of the butler repository
    obs_type : [string] Observation type of the image
    intrument : [string] Name of the instrument, by default 'DECam'
    -----
    Output
    -----
    data : [pd.DataFrame] Dataframe with columns: exposure, target_name, ra, dec, day_obs
    """
    butler = Butler(repo)
    registry = butler.registry
    exposures = []
    data=pd.DataFrame(columns =['exposure','target_name', 'ra', 'dec', 'day_obs'])
    for ref in registry.queryDimensionRecords('exposure',where="instrument='{}' AND exposure.observation_type='{}'".format(instrument,obs_type)):
        exposures.append(ref.dataId['exposure'])
        new_row = [ref.dataId['exposure'], ref.target_name, ref.tracking_ra, ref.tracking_dec, ref.day_obs]
        data.loc[len(data.index)] = new_row
    if data.empty:
        print("No exposure found in REPO:{} with observation type {}".format(repo, obs_type))

    return data

def Extract_information(repo, collection, visits, ccd_num, ra, dec, expotype='calexp', instrument='DECam', save=False, save_as =None):
    """
    extracts information of airmass, psf size (in arcsec), scaling calib parameter and its error
    onto a dataframe 

    Input
    -----
    repo
    collection 
    visits
    ccd_num
    expotype
    instrument

    output
    ------
    data
    
    """

    butler = Butler(repo)
    data=pd.DataFrame(columns =['airmass','psf', 'calib', 'calib_err', 'x_pix', 'y_pix', 'mjd', 'visit', 'zp', 'expTime'])
    for i in range(len(visits)):
        expo = butler.get(expotype, visit= visits[i], detector= ccd_num, instrument=instrument, collections=collection)
        psf = expo.getPsf() 
        arcsec_to_pixel = 0.27 #arcsec/pixel
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))  
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*arcsec_to_pixel*sigma2fwhm 
        airmass = float(expo.getInfo().getVisitInfo().boresightAirmass)
        ptcb_expo = expo.getPhotoCalib()
        calib = ptcb_expo.getCalibrationMean()
        calib_err = ptcb_expo.getCalibrationErr()
        zp = ptcb_expo.instFluxToMagnitude(1.0)
        wcs= expo.getWcs()
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        exp_visit_info = expo.getInfo().getVisitInfo()
        
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)        
        mjd= visit_date_astropy.mjd
        expTime = expo.getInfo().getVisitInfo().exposureTime 
      

        row = [airmass, seeing, calib, calib_err, x_pix, y_pix, mjd, visits[i], zp, expTime]

        data.loc[len(data.index)] = row
    if save:
        aux = collection.split('/')
        data.to_csv('info_{}.txt'.format(aux[-1]))
    return data

def radec_to_pixel(ra,dec,wcs):
    """

    Retrieves the x,y pixel position of a coordinate ra,dec given the wcs of the image.
    
    -----
    Input
    -----
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    wcs : [lsst.afw.geom.SkyWcs] Sky wcs of a product image from the LSST Science Pipelines
    -----
    Output
    -----
    x_pix : [float] x pixel position
    y_pix : [float] y pixel position
    """
    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
    return x_pix, y_pix

def Calib_and_Diff_plot(repo, collection_diff, collection_calexp, ra, dec, visits, ccd_num, s):
    """
    Plots the calibrated and difference-imaged exposure
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    ccd_num : [int] number of the DECam detector
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    butler = Butler(repo)
    for i in range(len(visits)):
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)

        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        diffexp_cat = butler.get('goodSeeingDiff_diaSrc',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        diffexp_im = diffexp.getMaskedImage()

        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        x_pix, y_pix = radec_to_pixel(ra, dec, diffexp.getWcs())

        fig = plt.figure(figsize=(16, 14))
        imag_display = []

        fig.add_subplot(1,2,1)
        imag_display.append(afwDisplay.Display(frame=fig))
        imag_display[0].scale('linear', 'zscale')
        imag_display[0].mtv(calexp_im)
        plt.title("Calibrated exposure", fontsize=17)
        imag_display[0].dot('o', x_pix, y_pix, ctype='green', size=25)

        with imag_display[0].Buffering():
            for j in calexp_cat[calexp_cat['calib_psf_used']]:
                 imag_display[0].dot("x", j.getX(), j.getY(), size=10, ctype="red")

        
        fig.add_subplot(1,2,2)
        imag_display.append(afwDisplay.Display(frame=fig))
        imag_display[1].scale('linear', 'zscale')
        imag_display[1].mtv(diffexp_im)
        plt.title("Subtracted exposure", fontsize=17)

        with imag_display[1].Buffering():
            for j in calexp_cat[calexp_cat['calib_psf_used']]:
                 imag_display[1].dot("x", j.getX(), j.getY(), size=10, ctype="red")
        

        imag_display[1].dot('o', x_pix, y_pix, ctype='green', size=25)

        for src in diffexp_cat:
            imag_display[1].dot('o', src.getX(), src.getY(), ctype='cyan', size=30)

        
        plt.tight_layout()
        plt.show()
    return

def Calib_plot(repo, collection_calexp, ra, dec, visits, ccd_num, s=20):
    """
    Plots the calibrated and difference-imaged exposure
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    ccd_num : [int] number of the DECam detector
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    butler = Butler(repo)
    for i in range(len(visits)):
        #i = 7 # index of visit
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)

        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')


        fig = plt.figure(figsize=(16, 14))
        imag_display = []

        fig.add_subplot(1,2,1)
        imag_display.append(afwDisplay.Display(frame=fig))
        imag_display[0].scale('linear', 'zscale')
        imag_display[0].mtv(calexp_im)
        plt.title("Calibrated exposure", fontsize=17)

        x_pix, y_pix = radec_to_pixel(ra, dec, calexp.getWcs())

        imag_display[0].dot('o', x_pix, y_pix, ctype='magenta', size=s)

        plt.tight_layout()
        plt.show()
    return

def Calib_cropped(repo, collection_calexp, ra, dec, visits, ccd_num, cutout=40, s=20):
    """
    Plots the calibrated and difference-imaged exposure cropped to the location of ra,dec
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    butler = Butler(repo)

    
    for i in range(len(visits)):
        obj_pos_lsst = lsst.geom.SpherePoint(ra[i], dec[i], lsst.geom.degrees)
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        
        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        wcs = calexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

        calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

        fig = plt.figure(figsize=(10, 5))
        stamp_display = []

        fig.add_subplot(1,2,1)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[0].scale('linear', 'zscale')
        print(visits[i])
        stamp_display[0].mtv(calexp_cutout.maskedImage)
        stamp_display[0].dot('o', x_pix, y_pix, ctype='magenta', size=s)
        for src in calexp_cat:
            stamp_display[0].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
        plt.title('Calexp Image and Source Catalog')
        plt.tight_layout()
        plt.show()
    return



def calib_crop(repo, collection_calexp, ra, dec, visits, ccd_num, cutout=40, s=20):
    """
    Plots the calibrated and difference-imaged exposure cropped to the location of ra,dec
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    butler = Butler(repo)

    
    for i in range(len(visits)):
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        
        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        wcs = calexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

        calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

        fig = plt.figure(figsize=(10, 5))
        stamp_display = []

        fig.add_subplot(1,2,1)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[0].scale('linear', 'zscale')
        print(visits[i])
        stamp_display[0].mtv(calexp_cutout.maskedImage)
        stamp_display[0].dot('o', x_pix, y_pix, ctype='magenta', size=s)
        for src in calexp_cat:
            stamp_display[0].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
        plt.title('Calexp Image and Source Catalog')
        plt.tight_layout()
        plt.show()
    return

def Calib_and_Diff_plot_cropped(repo, collection_diff, collection_calexp, ra, dec, visits, ccd_num, cutout=40, s=20):
    """
    Plots the calibrated and difference-imaged exposure cropped to the location of ra,dec
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    ccd_num : [int]
    cutout : [int]
    s : [int] circular display 
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    butler = Butler(repo)
    for i in range(len(visits)):
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        diffexp_cat = butler.get('goodSeeingDiff_diaSrc',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
                 
        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        wcs = diffexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

        calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

        fig = plt.figure(figsize=(10, 5))
        stamp_display = []

        fig.add_subplot(1,2,1)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[0].scale('asinh', 'zscale')
        stamp_display[0].mtv(calexp_cutout.maskedImage)
        stamp_display[0].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)

        with stamp_display[0].Buffering():
            for j in calexp_cat[calexp_cat['calib_psf_used']]:
                 stamp_display[0].dot("x", j.getX(), j.getY(), size=10, ctype="red")

        #for src in calexp_cat:
        #    stamp_display[0].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
        plt.title('Reduced Image')

        fig.add_subplot(1,2,2)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[1].scale('asinh', -10,10)
        stamp_display[1].mtv(diffexp_cutout.maskedImage)


        stamp_display[1].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)

        for src in diffexp_cat:
            stamp_display[1].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
            #if np.fabs(x_pix - src.getX())<cutout and np.fabs(y_pix - src.getY())<cutout:
            #    print('from catalog that is within the : {} {}'.format(src.getX(), src.getY()))
        plt.title('Difference Image')

        plt.tight_layout()
        plt.show()
        
        
    return



def Calib_Diff_and_Coadd_plot_cropped(repo, collection_diff, ra, dec, visits, ccd_num, cutout=40, s=20):
    """
    Plots the calibrated and difference-imaged exposure cropped to the location of ra,dec
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    ccd_num : [int] detector number 
    cutout : [int] half size in pixels of square cutout
    s : [int] area of circular display 
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    butler = Butler(repo)
    for i in range(len(visits)):
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        coadd = butler.get('goodSeeingDiff_matchedExp', visit=visits[i], detector=ccd_num, collections=collection_diff, instrument='DECam')
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        diffexp_cat = butler.get('goodSeeingDiff_diaSrc',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
                 
        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        wcs = diffexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

        calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        coadd_cutout = coadd.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

        fig = plt.figure(figsize=(16, 5))

        stamp_display = []

        fig.add_subplot(1,3,1)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[0].scale('asinh', -10, 10)
        stamp_display[0].mtv(calexp_cutout.maskedImage)
        stamp_display[0].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)

        with stamp_display[0].Buffering():
            for j in calexp_cat[calexp_cat['calib_psf_used']]:
                 stamp_display[0].dot("x", j.getX(), j.getY(), size=10, ctype="red")

        #for src in calexp_cat:
        #    stamp_display[0].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
        plt.title('Reduced Image', fontsize=15)

        fig.add_subplot(1,3,3)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[2].scale('asinh', -10,10)
        stamp_display[2].mtv(diffexp_cutout.maskedImage)


        stamp_display[2].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)

        for src in diffexp_cat:
            stamp_display[2].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
            if np.fabs(x_pix - src.getX())<cutout and np.fabs(y_pix - src.getY())<cutout:
                print('from catalog that is within the : {} {}'.format(src.getX(), src.getY()))
        plt.title('Difference Image', fontsize=15)

        fig.add_subplot(1,3,2)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[1].scale('asinh', -10, 10)
        stamp_display[1].mtv(coadd_cutout.maskedImage)
        stamp_display[1].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)
        
        plt.title('Template', fontsize=15)

        

        plt.tight_layout()
        plt.show()
        
        
    return



def raw_one_plot_cropped_astropy(repo, collection_raw, ra, dec, visit, ccd_num, cutout=40, s=20, field='', name=''):
    """
    Plots the raw exposure cropped to the location of ra,dec
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    ccd_num : [int] detector number 
    cutout : [int]
    s : [float] pixel radii for circular display in science and template image
    sd : [float] pixel radii for circular display in difference image
    field : [string] name of the field
    name : [string] name of the source (optional for saving purposes)
    -----
    Output
    -----
    None
    
    """

    cutout = 100
    butler = Butler(repo)
    #calexp = butler.get('calexp', visit= visit, detector= ccd_num, instrument='DECam', collections=collection)
    #calexp_im = calexp.getMaskedImage()
    #calexp_cat = butler.get('src', visit= visit, detector= ccd_num, instrument='DECam', collections=collection)
    collection_raw = 'DECam/raw/all'

    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)

    cross = butler.get('raw', exposure=visit, detector=ccd_num, collections=collection_raw, instrument='DECam') 
    cross_array = np.asarray(cross.image.array,dtype='float')
    m, s = np.mean(cross_array), np.std(cross_array)

    afwDisplay.setDefaultMaskTransparency(100)
    afwDisplay.setDefaultBackend('matplotlib')
    wcs = cross.getWcs()
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)

    if x_pix>=2000 or y_pix>=4000 or x_pix<50 or y_pix<50:
        return False
    else: 

        x_half_width = cutout
        y_half_width = cutout

        fig = plt.figure(figsize=(16, 5))
        stamp_display = []
        fig.add_subplot(1,1,1)  
        plt.imshow(cross_array, cmap='rocket', origin='lower', vmin = m-s, vmax=m+s)
        plt.colorbar()
        plt.scatter(x_pix,y_pix, color=neon_green, marker='x', linewidth=3)
        plt.xlim(x_pix - cutout, x_pix + cutout)
        plt.ylim(y_pix - cutout, y_pix + cutout)
        plt.title('Raw Image', fontsize=15)
        plt.show()
    
        return True


def Calib_one_plot_cropped_astropy(repo, collection, ra, dec, visit, ccd_num, cutout=40, s=20, field='', name=''):
    """
    Plots the calibrated and difference-imaged exposure cropped to the location of ra,dec
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    ccd_num : [int] detector number 
    cutout : [int]
    s : [float] pixel radii for circular display in science and template image
    sd : [float] pixel radii for circular display in difference image
    field : [string] name of the field
    name : [string] name of the source (optional for saving purposes)
    -----
    Output
    -----
    None
    
    """

    butler = Butler(repo)

    cutout_diff = 5
    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    calexp = butler.get('calexp', visit= visit, detector= ccd_num, instrument='DECam', collections=collection)
    calexp_im = calexp.getMaskedImage()
    calexp_cat = butler.get('src', visit= visit, detector= ccd_num, instrument='DECam', collections=collection)
         
    afwDisplay.setDefaultMaskTransparency(100)
    afwDisplay.setDefaultBackend('matplotlib')
    wcs = calexp.getWcs()
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
    x_half_width = cutout
    y_half_width = cutout
    bbox = lsst.geom.Box2I()
    bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
    bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

    calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

    calexp_cutout_forPeak = np.asarray(calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(10,10)).image.array, dtype = 'float')

    calexp_cutout_arr = np.asarray(calexp_cutout.image.array, dtype='float')

    fig = plt.figure(figsize=(16, 5))

    stamp_display = []

    
    fig.add_subplot(1,1,1)  
    plt.imshow(np.asarray(calexp.image.array,dtype='float'), cmap='rocket', origin='lower', vmin = 0, vmax=np.max(calexp_cutout_forPeak.flatten()))
    plt.colorbar()
    #levels=np.logspace(1.3, 2.5, 10),
    plt.contour(np.asarray(calexp.image.array,dtype='float'),  colors ='white', alpha=0.5)
    circle = plt.Circle((x_pix,y_pix), radius = s, color='red', fill = False, linewidth=4)
    plt.gca().add_patch(circle)
    plt.scatter(x_pix,y_pix, color=neon_green, marker='x', linewidth=3)
    plt.xlim(x_pix - cutout, x_pix + cutout)
    plt.ylim(y_pix - cutout, y_pix + cutout)

    plt.title('Reduced Image', fontsize=15)

    plt.show()
        
        
    return


def Calib_Diff_and_Coadd_plot_cropped_astropy(repo, collection_diff, ra, dec, visits, ccd_num, conv_image=None, cutout=20, s=20, sd=5, field='', name=''):
    """
    Plots the calibrated and difference-imaged exposure cropped to the location of ra,dec
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits
    ccd_num : [int] detector number 
    cutout : [int]
    s : [float] pixel radii for circular display in science and template image
    sd : [float] pixel radii for circular display in difference image
    field : [string] name of the field
    name : [string] name of the source (optional for saving purposes)
    -----
    Output
    -----
    None
    
    """
    
    factors = [0.5, 0.75, 1, 1.25, 1.5]
    
    if visits==[]:
        print("No visits submitted")
        return
    butler = Butler(repo)
    for i in range(len(visits)):
        cutout_diff = 5
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        try:
            coadd = butler.get('goodSeeingDiff_matchedExp', visit=visits[i], detector=ccd_num, collections=collection_diff, instrument='DECam')
        except:
            coadd = butler.get('goodSeeingDiff_warpedExp', visit=visits[i], detector=ccd_num, collections=collection_diff, instrument='DECam')
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        diffexp_cat = butler.get('goodSeeingDiff_diaSrc',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
                 
        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        wcs = diffexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

        calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        coadd_cutout = coadd.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

        calexp_cutout_forPeak = np.asarray(calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(10,10)).image.array, dtype = 'float')
        diffexp_cutout_forPeak = np.asarray(diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(10,10)).image.array, dtype = 'float')
        coadd_cutout_forPeak = np.asarray( coadd.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(10,10)).image.array, dtype = 'float')

        calexp_cutout_arr = np.asarray(calexp_cutout.image.array, dtype='float')
        diffexp_cutout_arr = np.asarray(diffexp_cutout.image.array, dtype='float')
        coadd_cutout_arr = np.asarray(coadd_cutout.image.array, dtype='float')
        #plt.rcParams['font.family'] = 'Trebuchet MS'
        fig = plt.figure(figsize=(16, 5))

        stamp_display = []

        #xPix = 
        
        fig.add_subplot(1,3,1)
        if conv_image is None:
            
            plt.imshow(np.asarray(calexp.image.array,dtype='float'), cmap='rocket', origin='lower', vmin = 0, vmax=np.max(calexp_cutout_forPeak.flatten()))
            plt.contour(np.asarray(calexp.image.array,dtype='float'),  colors ='white', alpha=0.5)
            plt.title('Reduced Image', fontsize=15)
            plt.colorbar()
        else:
            plt.imshow(conv_image, cmap='rocket', origin='lower', vmin = 0, vmax=np.max(calexp_cutout_forPeak.flatten()))
            plt.contour(conv_image,  colors ='white', alpha=0.5)
            plt.title('Convolved Image', fontsize=15)
        
            plt.colorbar()
        #levels=np.logspace(1.3, 2.5, 10),
        
        for fa in factors:
            circle = plt.Circle((x_pix,y_pix), radius = s * fa, color='red', fill = False, linewidth=2, alpha=0.6)
            plt.gca().add_patch(circle)
        #circle2 = plt.Circle((x_pix,y_pix), radius = sd, color=neon_green, fill = False, linewidth=4)
        #plt.scatter(x_pix,y_pix, color=neon_green, marker='x', linewidth=3)
        plt.xlim(x_pix - cutout, x_pix + cutout)
        plt.ylim(y_pix - cutout, y_pix + cutout)
        

        

        fig.add_subplot(1,3,2)
        plt.imshow(np.asarray(coadd.image.array,dtype='float'), cmap='rocket', origin='lower', vmin = 0 , vmax = np.max(coadd_cutout_forPeak.flatten()))
        plt.colorbar()
        #levels=np.logspace(1.3, 2.5, 10),
        plt.contour(np.asarray(coadd.image.array,dtype='float'),  colors ='white', alpha=0.5)
        plt.scatter(x_pix, y_pix, color=neon_green, marker='x', linewidth=3)
        #plt.scatter(x_half_width, y_half_width, s=np.pi*s**2, facecolors='none', edgecolors='red')
        #circle = plt.Circle((x_pix, y_pix), radius = s, color='red', fill = False, linewidth=4)
        #plt.gca().add_patch(circle)
        #circle2 = plt.Circle((x_pix, y_pix), radius = sd, color=neon_green, fill = False, linewidth=4)
        #plt.gca().add_patch(circle2)
        plt.title('Template', fontsize=15)
        plt.xlim(x_pix - cutout, x_pix + cutout)
        plt.ylim(y_pix - cutout, y_pix + cutout)

        fig.add_subplot(1,3,3)
        plt.imshow(np.asarray(diffexp.image.array,dtype='float'), vmin=np.min(diffexp_cutout_forPeak.flatten()), vmax=np.max(diffexp_cutout_forPeak.flatten()), cmap='rocket', origin='lower')
        plt.colorbar()
        # levels=np.logspace(1.3, 2.2, 10), 
        plt.contour(np.asarray(diffexp.image.array,dtype='float'), colors ='white', alpha=0.5)
        plt.scatter(x_pix, y_pix, color=neon_green, marker='x', linewidth=3)
        #plt.scatter(x_half_width, y_half_width, s=np.pi*s**2, facecolors='none', edgecolors='red')
        for fa in factors:
            circle = plt.Circle((x_pix, y_pix), radius = sd * fa, color=neon_green, fill = False, linewidth=2, alpha=0.6)
            plt.gca().add_patch(circle)
            
        plt.title('Difference Image', fontsize=15)
        plt.xlim(x_pix - cutout, x_pix + cutout)
        plt.ylim(y_pix - cutout, y_pix + cutout)

        plt.tight_layout()


        if i == 0:
            plt.savefig('{}/LSST_notebooks/light_curves/random_stamps/{}_{}_{}_one_stamp.jpeg'.format(main_root, field, ccd_name[ccd_num], name), bbox_inches='tight', dpi=300)
        plt.show()
        
        
    return


def stamps_and_LC_plot_forPaper(data_science, data_convol, data_diff, data_coadd, coords_datascience, coords_dataconvol, coords_coadd, Results_galaxy, Results_star, visits, kernel, seeing, SIBLING = '', cut_aux=40, r_diff =1, r_science=1,  field='', name='', first_mjd = 58810, name_to_save=''):
    """
    Plots the stamps and the light curves in a single figure.
    -----
    Input
    -----
    data_science [dict]: 
    data_convol [dict]
    data_diff [dict]
    data_coadd [dict]
    coords_datascience [dict]
    coords_dataconvol [dict]
    coords_coadd [dict]
    Results_galaxy [pd.DataFrame]
    Results_star [pd.DataFrame]
    visits [np.array]
    SIBLING [string]
    r_diff []

    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    columns = len(visits) 
    n_rows = 3#5
    n_cols = len(visits)

    # Create the figure and subplots using gridspec
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2), facecolor='k')
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    #print('Dictionary of the diffexp images: ' , data_diff)
    arcsec_to_pixel  = 0.27
    r_in_pixels = r_diff/arcsec_to_pixel
    rs_in_pixels = r_science/arcsec_to_pixel

    for col in range(n_cols):

        science_image = data_science[visits[col]]
        diff_image  = data_diff[visits[col]]

        x_pix, y_pix = coords_datascience[visits[col]]

        coadd_image = data_coadd[visits[col]]
        x_pix_coad, y_pix_coad = coords_coadd[visits[col]]

        convol_image = data_convol[visits[col]]
        x_pix_conv, y_pix_conv = coords_dataconvol[visits[col]]

        kernel_image = kernel[visits[col]]

        ax1 = axes[0,col] 
        #ax2 = axes[1,col] 
        #ax3 = axes[1,col]
        ax5 = axes[2,col] 
        ax4 = axes[1,col] 
        #print('ax4: ', ax4)

        ax1.set_xticklabels([])
        #ax2.set_xticklabels([])
        #ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        ax1.set_yticklabels([]) 
        #ax2.set_yticklabels([])
        #ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        ax5.set_yticklabels([])

        datascien_cut = science_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(datascien_cut).flatten()),  np.std(np.array(datascien_cut).flatten())
        vmin = m-s
        vmax = np.max(datascien_cut.flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=m-s, vmax=m+ 3*s)
        ax1.imshow(science_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax1.scatter(x_pix, y_pix, marker='x', color='k')
        ax1.add_patch(plt.Circle((x_pix, y_pix), radius=rs_in_pixels, color=neon_green, fill=False))
        ax1.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        ax1.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        
        dataconv_cut = convol_image[int(y_pix_conv-cut_aux): int(y_pix_conv+cut_aux),int(x_pix_conv-cut_aux): int(x_pix_conv+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(dataconv_cut).flatten()), np.std(np.array(dataconv_cut).flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)

        ax4.imshow(convol_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax3.scatter(x_pix_conv, y_pix_conv, marker='x', color='k')
        ax4.add_patch(plt.Circle((x_pix_conv, y_pix_conv), radius=rs_in_pixels, color=neon_green, fill=False))
        ax4.set_xlim(x_pix_conv - cut_aux, x_pix_conv + cut_aux)
        ax4.set_ylim(y_pix_conv - cut_aux, y_pix_conv + cut_aux)
        
        #datacoadd_cut = coadd_image[int(y_pix_coad-cut_aux): int(y_pix_coad+cut_aux),int(x_pix_coad-cut_aux): int(x_pix_coad+cut_aux)].copy(order='C')
        #m, s = np.mean(np.array(datacoadd_cut).flatten()), np.std(np.array(datacoadd_cut).flatten())
        ##log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)
        #ax2.imshow(coadd_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax2.add_patch(plt.Circle((x_pix_coad, y_pix_coad), radius=r_in_pixels, color=neon_green, fill=False))
        ##ax2.scatter(x_pix_coad, y_pix_coad, marker='x', color='k')
        #ax2.set_xlim(x_pix_coad - cut_aux, x_pix_coad + cut_aux)
        #ax2.set_ylim(y_pix_coad - cut_aux, y_pix_coad + cut_aux)

        datadiff_cut = diff_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        #m, s = np.mean(np.array(datadiff_cut).flatten()), np.std(np.array(datadiff_cut).flatten())
        ax5.imshow(diff_image, cmap='rocket', vmin=np.min(datadiff_cut.flatten()), vmax=np.max(datadiff_cut.flatten()))
        ax5.add_patch(plt.Circle((x_pix, y_pix), radius=r_in_pixels, color=neon_green, fill=False))
        #ax4.scatter(x_pix, y_pix, marker='x', color=neon_green)
        ax5.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        ax5.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        #cbar4 = fig.colorbar(img4, ax=ax4)
        #cbar4.set_label('Difference Data')

        #ax3.imshow(kernel_image, cmap='rocket', vmin=np.min(kernel_image.flatten()), vmax=np.max(kernel_image.flatten()))

        if col==0:
            ax1.set_ylabel('Science', fontsize=17)
            #ax2.set_ylabel('Template', fontsize=17)
            ax4.set_ylabel('Convolved', fontsize=17)
            #ax3.set_ylabel('my Kernel', fontsize=17)
            ax5.set_ylabel('Difference', fontsize=17)
    
    plt.subplots_adjust(hspace=0, wspace=0)

    plt.savefig('{}_forpaper_stamps.jpeg'.format(name_to_save), dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), facecolor='k')
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
    #ax1 = plt.subplot(gs[0, :])
    #ax1.plot(Results_galaxy.dates - min(Results_galaxy.dates), seeing, color='k')
    #ax1.set_ylabel('Seeing sigma [pixels]', fontsize=12)

    ax2 = plt.subplot(gs[0, :])
    ax2.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy['flux_ImagDiff_nJy_0.75sigmaPsf'], yerr = Results_galaxy['fluxErr_ImagDiff_nJy_0.75sigmaPsf'], capsize=4, fmt='s', color='w', ls ='dotted', label = 'diff image radii = 0.75 fwhm') # , label ='Fixed aperture of {}" * 1'.format(r_diff)
    #ax2.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.fluxConv_cal_nJy - np.mean(Results_galaxy.fluxConv_cal_nJy), yerr=Results_galaxy.fluxerrConv_cal_nJy, capsize=4, fmt='s', label='science after conv radii = {}"'.format(r_science), color='m', ls='dotted')
    #ax2.set_ylim(-2500,2500)
    ax2.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), Results_star.stars_diff_1sigmalow_byEpoch, Results_star.stars_diff_1sigmaupp_byEpoch, alpha=0.3, color='m', label = 'stars in diff image 1-2 $\sigma$ dev') #
    ax2.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), 2*Results_star.stars_diff_1sigmalow_byEpoch, 2*Results_star.stars_diff_1sigmaupp_byEpoch, alpha=0.3, color='m') #, label = 'stars 2-$\sigma$ dev'

    ax2.set_ylabel('Flux [nJy]', fontsize=18)
    #ax4.xlabel('MJD - {}'.format(int(min(dates_aux))), fontsize=15)
    
    if SIBLING!=None:
        x, y, yerr = compare_to(SIBLING, sfx='mag', factor=0.75)
        f, ferr = pc.ABMagToFlux(y, yerr)# in nJy
        #mfactor = 5e-10
        ax2.errorbar(x-min(x), f -  np.mean(f), yerr=ferr,  capsize=4, fmt='^', ecolor='blue', color='blue', label='Martinez-Palomera et al. 2020', ls ='dotted')
    
    ax2.legend(frameon=False, ncol=2, fontsize=12)

    ax3 =  plt.subplot(gs[1, :], sharex=ax2)

    ax3.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), Results_star.stars_science_1sigmalow_byEpoch,  Results_star.stars_science_1sigmaupp_byEpoch, alpha=0.3, color='blue', label = 'convolved stars 1-2 $\sigma$ dev') #
    ax3.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), 2*Results_star.stars_science_1sigmalow_byEpoch,  2*Results_star.stars_science_1sigmaupp_byEpoch, alpha=0.3, color='blue') #, label = 'stars 2-$\sigma$ dev'
    
    ax3.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy['flux_ConvDown_nJy_0.75wseeing'] - np.mean(Results_galaxy['flux_ConvDown_nJy_0.75wseeing']), yerr=Results_galaxy['fluxErr_ConvDown_nJy_0.75wseeing'], capsize=4, fmt='s', label='science after conv radii = 0.75 worst seeing'.format(r_science), color='m', ls='dotted')
    
    if SIBLING!=None:
        x, y, yerr = compare_to(SIBLING, sfx='mag', factor=0.75)
        f, ferr = pc.ABMagToFlux(y, yerr)# in nJy
        #mfactor = 5e-10
        ax3.errorbar(x-min(x), f -  np.mean(f), yerr=ferr,  capsize=4, fmt='^', ecolor='blue', color='blue', label='Martinez-Palomera et al. 2020', ls ='dotted')
    
    #ax6.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.flux_nJy_cal - np.mean(Results_galaxy.flux_nJy_cal), yerr = Results_galaxy.fluxerr_nJy_cal, capsize=4, fmt='d', label ='science before conv radii = {}'.format(r_diff), color='green', ls ='dotted')
    #ax3.set_ylim(-2500,2500)
    ax3.set_ylabel('Flux [nJy] - Median', fontsize=18)
    ax3.set_xlabel('MJD - {}'.format(first_mjd), fontsize=18)
    ax3.legend(frameon=False, ncol=2, fontsize=12)
    plt.subplots_adjust(hspace=0, wspace=0.2)
    plt.savefig('{}_forpaper_LCs.jpeg'.format(name_to_save), dpi=300, bbox_inches='tight')
    plt.show()

    return

def stamps(data_science, data_convol, data_diff, data_coadd, coords_datascience, coords_dataconvol, coords_coadd, Results_galaxy, Results_star, visits, kernel, seeing, SIBLING = '', cut_aux=40, r_diff =[1/0.27], r_science=[1/0.27],  field='', name='', first_mjd = 58810, name_to_save='', folder='./'):
    """
    Plots the stamps and the light curves in a single figure.
    -----
    Input
    -----
    data_science [dict]: 
    data_convol [dict]
    data_diff [dict]
    data_coadd [dict]
    coords_datascience [dict]
    coords_dataconvol [dict]
    coords_coadd [dict]
    Results_galaxy [pd.DataFrame]
    Results_star [pd.DataFrame]
    visits [np.array]
    SIBLING [string]
    r_diff []

    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    columns = len(visits) 
    n_rows = 5
    n_cols = len(visits)

    # Create the figure and subplots using gridspec
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, 5*2))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    arcsec_to_pixel  = 0.27
    #r_in_pixels = r_diff/arcsec_to_pixel
    #rs_in_pixels = r_science/arcsec_to_pixel

    for col in range(n_cols):

        science_image = data_science[visits[col]]
        diff_image  = data_diff[visits[col]]

        x_pix, y_pix = coords_datascience[visits[col]]

        coadd_image = data_coadd[visits[col]]
        x_pix_coad, y_pix_coad = coords_coadd[visits[col]]

        convol_image = data_convol[visits[col]]
        x_pix_conv, y_pix_conv = coords_dataconvol[visits[col]]

        kernel_image = kernel[visits[col]]

        ax1 = axes[0,col] 
        ax2 = axes[1,col] 
        ax3 = axes[2,col]
        ax5 = axes[4,col] 
        ax4 = axes[3,col] 

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        ax1.set_yticklabels([]) 
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        ax5.set_yticklabels([])
        
        for r in r_diff:
            ax1.add_patch(plt.Circle((x_pix, y_pix), radius=r, color=neon_green, fill=False, alpha=0.6))
            ax2.add_patch(plt.Circle((x_pix_coad, y_pix_coad), radius=r, color=neon_green, fill=False, alpha=0.6))
            ax5.add_patch(plt.Circle((x_pix, y_pix), radius=r, color=neon_green, fill=False, alpha=0.6))
            
        datascien_cut = science_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(datascien_cut).flatten()),  np.std(np.array(datascien_cut).flatten())
        vmin = m-s
        vmax = np.max(datascien_cut.flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=m-s, vmax=m+ 3*s)
        ax1.imshow(science_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax1.scatter(x_pix, y_pix, marker='x', color='k')
        
            
        ax1.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        ax1.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        
        dataconv_cut = convol_image[int(y_pix_conv-cut_aux): int(y_pix_conv+cut_aux),int(x_pix_conv-cut_aux): int(x_pix_conv+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(dataconv_cut).flatten()), np.std(np.array(dataconv_cut).flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)

        ax4.imshow(convol_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax3.scatter(x_pix_conv, y_pix_conv, marker='x', color='k')
        for r in r_science:
            ax4.add_patch(plt.Circle((x_pix_conv, y_pix_conv), radius=r, color='blue', fill=False, alpha=0.6))
        ax4.set_xlim(x_pix_conv - cut_aux, x_pix_conv + cut_aux)
        ax4.set_ylim(y_pix_conv - cut_aux, y_pix_conv + cut_aux)
        
        datacoadd_cut = coadd_image[int(y_pix_coad-cut_aux): int(y_pix_coad+cut_aux),int(x_pix_coad-cut_aux): int(x_pix_coad+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(datacoadd_cut).flatten()), np.std(np.array(datacoadd_cut).flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)
        ax2.imshow(coadd_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax2.scatter(x_pix_coad, y_pix_coad, marker='x', color='k')
        ax2.set_xlim(x_pix_coad - cut_aux, x_pix_coad + cut_aux)
        ax2.set_ylim(y_pix_coad - cut_aux, y_pix_coad + cut_aux)

        datadiff_cut = diff_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        #m, s = np.mean(np.array(datadiff_cut).flatten()), np.std(np.array(datadiff_cut).flatten())
        ax5.imshow(diff_image, cmap='rocket', vmin=np.min(datadiff_cut.flatten()), vmax=np.max(datadiff_cut.flatten()))
        #ax4.scatter(x_pix, y_pix, marker='x', color=neon_green)
        ax5.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        ax5.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        #cbar4 = fig.colorbar(img4, ax=ax4)
        #cbar4.set_label('Difference Data')
        
        #m, s = np.mean(kernel_image.flatten()), np.std(kernel_image.flatten())
        ax3.imshow(np.arcsinh(kernel_image/0.01), cmap='rocket')

        if col==0:
            ax1.set_ylabel('Science', fontsize=17)
            ax2.set_ylabel('Template', fontsize=17)
            ax4.set_ylabel('my Convolved', fontsize=17)
            ax3.set_ylabel('my Kernel', fontsize=17)
            ax5.set_ylabel('Difference', fontsize=17)
    
    plt.subplots_adjust(hspace=0, wspace=0)

    plt.savefig(folder / 'galaxy_stamps.jpeg', bbox_inches='tight')
    plt.show()


    return


def stamps_instcal(data_science, data_convol, coords_datascience, coords_dataconvol, Results_galaxy, Results_star, visits, kernel, seeing, SIBLING = '', cut_aux=40, r_diff =[1/0.27], r_science=[1/0.27],  field='', name='', first_mjd = 58810, name_to_save='', folder='./'):
    """
    Plots the stamps and the light curves in a single figure.
    -----
    Input
    -----
    data_science [dict]: 
    data_convol [dict]
    data_diff [dict]
    data_coadd [dict]
    coords_datascience [dict]
    coords_dataconvol [dict]
    coords_coadd [dict]
    Results_galaxy [pd.DataFrame]
    Results_star [pd.DataFrame]
    visits [np.array]
    SIBLING [string]
    r_diff []

    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    columns = len(visits) 
    n_rows = 3
    n_cols = len(visits)

    # Create the figure and subplots using gridspec
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, 5*2))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    arcsec_to_pixel  = 0.27
    #r_in_pixels = r_diff/arcsec_to_pixel
    #rs_in_pixels = r_science/arcsec_to_pixel

    for col in range(n_cols):

        science_image = data_science[visits[col]]

        x_pix, y_pix = coords_datascience[visits[col]]

        convol_image = data_convol[visits[col]]
        x_pix_conv, y_pix_conv = coords_dataconvol[visits[col]]

        kernel_image = kernel[visits[col]]

        ax1 = axes[0,col] 
        #ax2 = axes[1,col] 
        ax3 = axes[2,col]
        #ax5 = axes[4,col] 
        ax4 = axes[1,col] 

        ax1.set_xticklabels([])
        #ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        #ax5.set_xticklabels([])
        ax1.set_yticklabels([]) 
        #ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        #ax5.set_yticklabels([])
        
        for r in r_diff:
            ax1.add_patch(plt.Circle((x_pix, y_pix), radius=r, color=neon_green, fill=False, alpha=0.6))
            #ax2.add_patch(plt.Circle((x_pix_coad, y_pix_coad), radius=r, color=neon_green, fill=False, alpha=0.6))
            #ax5.add_patch(plt.Circle((x_pix, y_pix), radius=r, color=neon_green, fill=False, alpha=0.6))
            
        datascien_cut = science_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(datascien_cut).flatten()),  np.std(np.array(datascien_cut).flatten())
        vmin = m-s
        vmax = np.max(datascien_cut.flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=m-s, vmax=m+ 3*s)
        ax1.imshow(science_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax1.scatter(x_pix, y_pix, marker='x', color='k')
        
            
        ax1.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        ax1.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        
        dataconv_cut = convol_image[int(y_pix_conv-cut_aux): int(y_pix_conv+cut_aux),int(x_pix_conv-cut_aux): int(x_pix_conv+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(dataconv_cut).flatten()), np.std(np.array(dataconv_cut).flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)

        ax4.imshow(convol_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax3.scatter(x_pix_conv, y_pix_conv, marker='x', color='k')
        for r in r_science:
            ax4.add_patch(plt.Circle((x_pix_conv, y_pix_conv), radius=r, color='blue', fill=False, alpha=0.6))
        ax4.set_xlim(x_pix_conv - cut_aux, x_pix_conv + cut_aux)
        ax4.set_ylim(y_pix_conv - cut_aux, y_pix_conv + cut_aux)
        
        #datacoadd_cut = coadd_image[int(y_pix_coad-cut_aux): int(y_pix_coad+cut_aux),int(x_pix_coad-cut_aux): int(x_pix_coad+cut_aux)].copy(order='C')
        #m, s = np.mean(np.array(datacoadd_cut).flatten()), np.std(np.array(datacoadd_cut).flatten())
        ##log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)
        #ax2.imshow(coadd_image, cmap='rocket', vmin=vmin, vmax=vmax)
        ##ax2.scatter(x_pix_coad, y_pix_coad, marker='x', color='k')
        #ax2.set_xlim(x_pix_coad - cut_aux, x_pix_coad + cut_aux)
        #ax2.set_ylim(y_pix_coad - cut_aux, y_pix_coad + cut_aux)

        #datadiff_cut = diff_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        ##m, s = np.mean(np.array(datadiff_cut).flatten()), np.std(np.array(datadiff_cut).flatten())
        #ax5.imshow(diff_image, cmap='rocket', vmin=np.min(datadiff_cut.flatten()), vmax=np.max(datadiff_cut.flatten()))
        ##ax4.scatter(x_pix, y_pix, marker='x', color=neon_green)
        #ax5.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        #ax5.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        #cbar4 = fig.colorbar(img4, ax=ax4)
        #cbar4.set_label('Difference Data')
        
        #m, s = np.mean(kernel_image.flatten()), np.std(kernel_image.flatten())
        ax3.imshow(np.arcsinh(kernel_image/0.01), cmap='rocket')

        if col==0:
            ax1.set_ylabel('Science', fontsize=17)
            #ax2.set_ylabel('Template', fontsize=17)
            ax4.set_ylabel('my Convolved', fontsize=17)
            ax3.set_ylabel('my Kernel', fontsize=17)
            #ax5.set_ylabel('Difference', fontsize=17)
    
    plt.subplots_adjust(hspace=0, wspace=0)
    
    plt.savefig(folder / 'galaxy_stamps.jpeg', bbox_inches='tight')
    plt.show()


    return

def stamps_and_LC_plot(data_science, data_convol, data_diff, data_coadd, coords_datascience, coords_dataconvol, coords_coadd, Results_galaxy, Results_star, visits, kernel, seeing, SIBLING = '', cut_aux=40, r_diff =1, r_science=1,  field='', name='', first_mjd = 58810, name_to_save=''):
    """
    Plots the stamps and the light curves in a single figure.
    -----
    Input
    -----
    data_science [dict]: 
    data_convol [dict]
    data_diff [dict]
    data_coadd [dict]
    coords_datascience [dict]
    coords_dataconvol [dict]
    coords_coadd [dict]
    Results_galaxy [pd.DataFrame]
    Results_star [pd.DataFrame]
    visits [np.array]
    SIBLING [string]
    r_diff []

    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return
    columns = len(visits) 
    n_rows = 5
    n_cols = len(visits)

    # Create the figure and subplots using gridspec
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, 5*2))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    #print('Dictionary of the diffexp images: ' , data_diff)
    arcsec_to_pixel  = 0.27
    r_in_pixels = r_diff/arcsec_to_pixel
    rs_in_pixels = r_science/arcsec_to_pixel

    for col in range(n_cols):

        science_image = data_science[visits[col]]
        diff_image  = data_diff[visits[col]]

        x_pix, y_pix = coords_datascience[visits[col]]

        coadd_image = data_coadd[visits[col]]
        x_pix_coad, y_pix_coad = coords_coadd[visits[col]]

        convol_image = data_convol[visits[col]]
        x_pix_conv, y_pix_conv = coords_dataconvol[visits[col]]

        kernel_image = kernel[visits[col]]

        ax1 = axes[0,col] 
        ax2 = axes[1,col] 
        ax3 = axes[2,col]
        ax5 = axes[4,col] 
        ax4 = axes[3,col] 
        #print('ax4: ', ax4)

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        ax1.set_yticklabels([]) 
        ax2.set_yticklabels([])
        ax3.set_yticklabels([])
        ax4.set_yticklabels([])
        ax5.set_yticklabels([])

        datascien_cut = science_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(datascien_cut).flatten()),  np.std(np.array(datascien_cut).flatten())
        vmin = m-s
        vmax = np.max(datascien_cut.flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=m-s, vmax=m+ 3*s)
        ax1.imshow(science_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax1.scatter(x_pix, y_pix, marker='x', color='k')
        ax1.add_patch(plt.Circle((x_pix, y_pix), radius=rs_in_pixels, color=neon_green, fill=False))
        ax1.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        ax1.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        
        dataconv_cut = convol_image[int(y_pix_conv-cut_aux): int(y_pix_conv+cut_aux),int(x_pix_conv-cut_aux): int(x_pix_conv+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(dataconv_cut).flatten()), np.std(np.array(dataconv_cut).flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)

        ax4.imshow(convol_image, cmap='rocket', vmin=vmin, vmax=vmax)
        #ax3.scatter(x_pix_conv, y_pix_conv, marker='x', color='k')
        ax4.add_patch(plt.Circle((x_pix_conv, y_pix_conv), radius=rs_in_pixels, color=neon_green, fill=False))
        ax4.set_xlim(x_pix_conv - cut_aux, x_pix_conv + cut_aux)
        ax4.set_ylim(y_pix_conv - cut_aux, y_pix_conv + cut_aux)
        
        datacoadd_cut = coadd_image[int(y_pix_coad-cut_aux): int(y_pix_coad+cut_aux),int(x_pix_coad-cut_aux): int(x_pix_coad+cut_aux)].copy(order='C')
        m, s = np.mean(np.array(datacoadd_cut).flatten()), np.std(np.array(datacoadd_cut).flatten())
        #log_norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=m+ 3*s)
        ax2.imshow(coadd_image, cmap='rocket', vmin=vmin, vmax=vmax)
        ax2.add_patch(plt.Circle((x_pix_coad, y_pix_coad), radius=r_in_pixels, color=neon_green, fill=False))
        #ax2.scatter(x_pix_coad, y_pix_coad, marker='x', color='k')
        ax2.set_xlim(x_pix_coad - cut_aux, x_pix_coad + cut_aux)
        ax2.set_ylim(y_pix_coad - cut_aux, y_pix_coad + cut_aux)

        datadiff_cut = diff_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
        #m, s = np.mean(np.array(datadiff_cut).flatten()), np.std(np.array(datadiff_cut).flatten())
        ax5.imshow(diff_image, cmap='rocket', vmin=np.min(datadiff_cut.flatten()), vmax=np.max(datadiff_cut.flatten()))
        ax5.add_patch(plt.Circle((x_pix, y_pix), radius=r_in_pixels, color=neon_green, fill=False))
        #ax4.scatter(x_pix, y_pix, marker='x', color=neon_green)
        ax5.set_xlim(x_pix - cut_aux, x_pix + cut_aux)
        ax5.set_ylim(y_pix - cut_aux, y_pix + cut_aux)
        #cbar4 = fig.colorbar(img4, ax=ax4)
        #cbar4.set_label('Difference Data')

        ax3.imshow(kernel_image, cmap='rocket', vmin=np.min(kernel_image.flatten()), vmax=np.max(kernel_image.flatten()))

        if col==0:
            ax1.set_ylabel('Science', fontsize=17)
            ax2.set_ylabel('Template', fontsize=17)
            ax4.set_ylabel('my Convolved', fontsize=17)
            ax3.set_ylabel('my Kernel', fontsize=17)
            ax5.set_ylabel('Difference', fontsize=17)
    
    plt.subplots_adjust(hspace=0, wspace=0)

    plt.savefig('{}_stamps.jpeg'.format(name_to_save), dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1.5, 1.5])
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(Results_galaxy.dates - min(Results_galaxy.dates), seeing, color='k')
    ax1.set_ylabel('Seeing sigma [pixels]', fontsize=12)

    ax2 = plt.subplot(gs[1, :], sharex=ax1)
    ax2.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy['flux_ImagDiff_nJy_0.75sigmaPsf'], yerr = Results_galaxy['fluxErr_ImagDiff_nJy_0.75sigmaPsf'], capsize=4, fmt='s', color='k', ls ='dotted', label = 'diff image radii = {}"'.format(r_diff)) # , label ='Fixed aperture of {}" * 1'.format(r_diff)
    ax2.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy['flux_ConvDown_nJy_0.75wseeing'] - np.mean(Results_galaxy['flux_ConvDown_nJy_0.75wseeing']), yerr=Results_galaxy['fluxErr_ConvDown_nJy_0.75wseeing'], capsize=4, fmt='s', label='science after conv radii = {}"'.format(r_science), color='m', ls='dotted')
      
    ax2.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), Results_star.stars_diff_1sigmalow_byEpoch, Results_star.stars_diff_1sigmaupp_byEpoch, alpha=0.1, color='m', label = 'stars in diff image 1-2 $\sigma$ dev') #
    ax2.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), Results_star.stars_diff_2sigmalow_byEpoch, Results_star.stars_diff_2sigmaupp_byEpoch, alpha=0.1, color='m') #, label = 'stars 2-$\sigma$ dev'

    ax2.set_ylabel('Flux [nJy]', fontsize=12)
    #ax4.xlabel('MJD - {}'.format(int(min(dates_aux))), fontsize=15)
    
    if SIBLING!=None:
        x, y, yerr = compare_to(SIBLING, sfx='mag', factor=0.75)
        f, ferr = pc.ABMagToFlux(y, yerr)# in nJy
        #mfactor = 5e-10
        ax2.errorbar(x-min(x), f -  np.mean(f), yerr=ferr,  capsize=4, fmt='^', ecolor='orange', color='orange', label='Martinez-Palomera et al. 2020', ls ='dotted')
    ax2.legend(frameon=False, ncol=2)

    ax3 =  plt.subplot(gs[2, :], sharex=ax2)

    ax3.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), Results_star.stars_science_1sigmalow_byEpoch,  Results_star.stars_science_1sigmaupp_byEpoch, alpha=0.1, color='blue', label = 'convolved stars 1-2 $\sigma$ dev') #
    ax3.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), Results_star.stars_science_2sigmalow_byEpoch,  Results_star.stars_science_2sigmaupp_byEpoch, alpha=0.1, color='blue') #, label = 'stars 2-$\sigma$ dev'

    ax3.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy['flux_ConvDown_nJy_0.75wseeing'] - np.mean(Results_galaxy['flux_ConvDown_nJy_0.75wseeing']), yerr=Results_galaxy['fluxErr_ConvDown_nJy_0.75wseeing'], capsize=4, fmt='s', label='science after conv radii = {}"'.format(r_science), color='m', ls='dotted')
    
    #ax6.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.flux_nJy_cal - np.mean(Results_galaxy.flux_nJy_cal), yerr = Results_galaxy.fluxerr_nJy_cal, capsize=4, fmt='d', label ='science before conv radii = {}'.format(r_diff), color='green', ls ='dotted')
    ax3.set_ylabel('Flux [nJy] - Median', fontsize=12)

    ax3.set_xlabel('MJD - {}'.format(first_mjd), fontsize=12)
    ax3.legend(frameon=False, ncol=2)
    plt.subplots_adjust(hspace=0, wspace=0.2)
    plt.savefig('{}_LCs.jpeg'.format(name_to_save), dpi=300, bbox_inches='tight')
    plt.show()

    return





def Calib_and_Diff_one_plot_cropped(repo, collection_diff, collection_calexp, ra, dec, visits, ccd_num, cutout=40, s=20, save_stamps=False, save_as=''):
    """
    Plots all the calibrated and difference-imaged exposure cropped to the location of ra,dec in a single plot (figure).
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits.
    ccd_num : [int]
    cutout : [int]
    s : [int] circular display 
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return

    columns = len(visits) 
    fig = plt.figure(figsize=(columns*3, 2*3))
    butler = Butler(repo)
    #i=0
    for i in range(len(visits)):
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        diffexp_cat = butler.get('goodSeeingDiff_diaSrc',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
 
        exp_visit_info = calexp.getInfo().getVisitInfo()
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)

        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        wcs = diffexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

        calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

        stamp_display = []
        i+=1
        fig.add_subplot(2,columns,i)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[0].scale('asinh', -20, 50)
        stamp_display[0].mtv(calexp_cutout.maskedImage)
        
        stamp_display[0].dot('o', x_pix, y_pix, ctype='cyan', size=s)
        #stamp_display[0].dot('o', x_pix, y_pix, ctype='magenta', size=2*s)
        #stamp_display[0].dot('o', x_pix, y_pix, ctype='magenta', size=5*s)

        plt.axis('off')
        plt.title(visit_date_astropy.mjd)

        with stamp_display[0].Buffering():
            for j in calexp_cat[calexp_cat['calib_psf_used']]:
                 stamp_display[0].dot("x", j.getX(), j.getY(), size=10, ctype="red")
        



        fig.add_subplot(2, columns,i+columns)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[1].scale('asinh', -10,10)
        stamp_display[1].mtv(diffexp_cutout.maskedImage)
        stamp_display[1].dot('o', x_pix, y_pix, ctype='cyan', size=s)
        #stamp_display[1].dot('o', x_pix, y_pix, ctype='magenta', size=2*s)
        #stamp_display[1].dot('o', x_pix, y_pix, ctype='magenta', size=5*s)
        plt.axis('off')
        plt.title(diffexp.getInfo().getVisitInfo().id)
        #for src in diffexp_cat:
        #    stamp_display[1].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
        #plt.title('Diffexp Image and Source Catalog')

        #plt.tight_layout()
        #plt.show()
    if save_stamps and save_as!='':
        plt.savefig('{}/LSST_notebooks/light_curves/{}.jpeg'.format(main_root, save_as))

        
    return


def values_across_source(exposure, ra, dec , x_length, y_length, stat='median', title_plot = '', save_plot =False, field=None, name =None):
    """
    Returns an array of the values across a rectangular slit of a source,
    that is wider in the x-axis

    input:
    -----
    data
    xpix
    ypix
    x_length 
    y_length
    stat

    output 
    ------
    array
    """
    wcs = exposure.getWcs()
    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
    x_half_width = x_length
    y_half_width = y_length
    bbox = lsst.geom.Box2I()
    bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
    bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))
    exp_photocalib = exposure.getPhotoCalib()
    exp_cutout = exposure.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(x_length*2, y_length*2))
    exp_cutout_array = np.asarray(exp_cutout.image.array, dtype='float')
    obj_pos_2d = lsst.geom.Point2D(ra, dec)
   
    if stat == 'median':
        adu_values = np.median(exp_cutout_array, axis=0)
    fluxes = [exp_photocalib.instFluxToNanojansky(f, obj_pos_2d) for f in adu_values]

    ai, aip, bi, bip = special.airy(fluxes)


    f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, sharex=True, figsize=(10,6))
    ax1.set_title(title_plot, fontsize=17)
    ax1.imshow(exp_cutout_array)
    ax1.scatter(x_length, y_length, color='red', s=20)
    ax2.plot(range(len(adu_values)), ai)
    ax2.plot(range(len(adu_values)), bi)

    ax2.bar(range(len(adu_values)), fluxes, color = lilac)
    ax2.set_ylabel('Flux [nJy]', fontsize=17)
    ax2.set_xlabel('x-axis pixels', fontsize=17)
    ax1.set_ylabel('y-axis pixels', fontsize=17)
    f.subplots_adjust(hspace=0)
    if save_plot:
        #f.savefig('light_curves/{}/{}.jpeg'.format(field, name), bbox_inches='tight')
        pass
    plt.show()
    return fluxes

def Calib_Diff_and_Coadd_one_plot_cropped(repo, collection_diff, ra, dec, visits, ccd_num, cutout=40, s=20, save_stamps=False, save_as=''):
    """
    Plots all the calibrated and difference-imaged exposure cropped to the location of ra,dec in a single plot (figure).
    -----
    Input
    -----
    repo : [string] directory of the butler repository 
    collection_diff : [string] name of the difference imaging collection
    collection_calexp : [string] name of the calibrated exposures collection
    ra : [float] right ascention coordinate in decimal degrees
    dec : [float] declination coordinate in decimal degrees
    visits : [ndarray] list of visits.
    ccd_num : [int]
    cutout : [int]
    s : [int] circular display 
    -----
    Output
    -----
    None
    
    """
    if visits==[]:
        print("No visits submitted")
        return

    columns = len(visits) 
    fig = plt.figure(figsize=(columns*3, 3*3))
    butler = Butler(repo)
    #i=0
    for i in range(len(visits)):
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_diff)
        
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        diffexp_cat = butler.get('goodSeeingDiff_diaSrc',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        coadd = butler.get('goodSeeingDiff_matchedExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        exp_visit_info = calexp.getInfo().getVisitInfo()
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)

        afwDisplay.setDefaultMaskTransparency(100)
        afwDisplay.setDefaultBackend('matplotlib')
        wcs = diffexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))

        calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
        coadd_cutout = coadd.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))


        stamp_display = []
        i+=1
        fig.add_subplot(3,columns,i)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[0].scale('asinh', -20, 50)
        stamp_display[0].mtv(calexp_cutout.maskedImage)
        
        stamp_display[0].dot('o', x_pix, y_pix, ctype='cyan', size=s)
        if i==1:
            #print('hello i == 1')
            plt.ylabel('Science', fontsize=15)
        plt.axis('off')
        plt.title(visit_date_astropy.mjd)

        with stamp_display[0].Buffering():
            for j in calexp_cat[calexp_cat['calib_psf_used']]:
                 stamp_display[0].dot("x", j.getX(), j.getY(), size=10, ctype="red")

        fig.add_subplot(3, columns,i+columns)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[1].scale('asinh', -10,10)
        stamp_display[1].mtv(diffexp_cutout.maskedImage)
        stamp_display[1].dot('o', x_pix, y_pix, ctype='cyan', size=s)

        plt.axis('off')
        plt.title(diffexp.getInfo().getVisitInfo().id)
        if i==1:
            plt.ylabel('Difference', fontsize=15)

        fig.add_subplot(3, columns,i+2*columns)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[2].scale('asinh', -20,50)
        stamp_display[2].mtv(coadd_cutout.maskedImage)
        stamp_display[2].dot('o', x_pix, y_pix, ctype='cyan', size=s)
        plt.axis('off')
        if i==1:
            plt.ylabel('Template', fontsize=15)

    if save_stamps and save_as!='':
        plt.savefig('{}/LSST_notebooks/light_curves/{}.jpeg'.format(main_root, save_as), bbox_inches='tight')

        
    return





def Order_Visits_by_Date(repo, visits, ccd_num, collection_diff):
    '''
    Returns visits and dates sorted by the latter.
    
    input:
    repo [str] : Directory were butler repository is
    visits [array] : Array of integers 
    ccd_num [int] : number of the detector used
    collection_diff [str] : name of the collection data (here the from difference Imaging)

    '''
    butler = Butler(repo)
    dates = []
    for i in range(len(visits)):
        #diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        
        try:
            calexp = butler.get('calexp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 


            exp_visit_info = calexp.getInfo().getVisitInfo()

            visit_date_python = exp_visit_info.getDate().toPython()
            visit_date_astropy = Time(visit_date_python)        
            dates.append(visit_date_astropy.mjd)
        except LookupError:
            print('did not find the visit: ', visits[i])
            continue

    dates = dates #- min(dates)
    zipped = zip(dates, visits)
    res = sorted(zipped, key = lambda x: x[0])

    dates_aux, visits_aux = zip(*list(res))

    return dates_aux, visits_aux

def Find_worst_seeing(repo, visits, ccd_num, collection, arcsec_to_pixel = 0.27):
    '''
    Returns worst seeing (sigma) in pixel values 

    input
    ------
    repo [str] : directory were butler repository is
    visits [array of ints] : 
    ccd_num [int] : detector number
    collection [str] : name of collection were calexp type image is found

    output
    -------
    X, Y [float], [int] : maximum sigma seeing in pixels, visit number associated to it.
    '''
    butler = Butler(repo)
    Seeing = []
    
    for i in range(len(visits)):
        
        calexp = butler.get('calexp',visit=visits[i], detector=ccd_num , collections=collection, instrument='DECam') 
        psf = calexp.getPsf() 
        #sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))

        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()#* arcsec_to_pixel # to arcseconds 
        Seeing.append(seeing)
    
    j, = np.where(Seeing == np.max(Seeing))
    worst_visit = visits[j[0]]
    
    return np.max(Seeing), worst_visit


def centering_coords(data_cal, x_pix, y_pix, side_half_length_box, show_stamps, how='sep', minarea = np.pi * (0.5/arcsec_to_pixel)**2, flux_thresh=None):
    """
    Function that helps to find the centroid of the stamp of an image.
    
    data_cal [matrix np.array]
    x_pix [float] : x axis in pix
    y_pix [float] : y axis in pix
    wcs_cal [wcs lsst] : wcs object retrieved with lsst modules\
    side_half_length_box [int] : half length of the box we will do a stamp with
    show_stamps [bool] : If true, we plot the stamp with the new centroid
    how [str] : method of finding the centroid 
    
    """
    #print('before centering correction: xpix, y_pix: {} {} '.format(x_pix, y_pix))
    arcsec_to_pixel=0.27 #626
    # side_half_length_box = 20 # in pixels 
   
    #minarea = np.pi * (0.5/arcsec_to_pixel)**2 # area at which we more or less do aperture photometry (small)
            
    # We create a stamp surrounding the galaxy
    sub_data = data_cal[int(y_pix-side_half_length_box):int(side_half_length_box+y_pix),int(x_pix-side_half_length_box):int(side_half_length_box+x_pix)].copy(order='C')
    
    m, s = np.mean(sub_data), np.std(sub_data)
    
    sepThresh = m + 2*s # it was 3*s before
    peak_value = m + 3*s
    
    
    if flux_thresh is not None:
        sepThresh = flux_thresh
        
    #print('sepTresh: ', sepThresh)
    #print('minarea: ', minarea )
    
    if how=='sep':
        
        # We do an object detection 
        objects = sep.extract(sub_data, sepThresh, minarea=minarea)
        try:
            obj, j = Select_largest_flux(sub_data, objects)
            #print('obj: ',obj)
            x_pix_aux = obj['x']
            y_pix_aux = obj['y']
        except ValueError:
            print('Centering using {} failed... '.format(how))
            
            return np.array([x_pix]), np.array([y_pix])
            #x_pix_aux = np.array([side_half_length_box])
            #y_pix_aux = np.array([side_half_length_box])
            #print(x_pix_aux, y_pix_aux)
        
    elif how=='photutils':
        
        x_pix_aux, y_pix_aux = centroid_2dg(sub_data)
        x_pix_aux = np.array([x_pix_aux])
        y_pix_aux = np.array([y_pix_aux])
        
    x_pix_OgImage = x_pix_aux + int(x_pix - side_half_length_box)
    y_pix_OgImage = y_pix_aux + int(y_pix - side_half_length_box)
    
    #print('Updated pixels x_pix {} y_pix {}'.format(x_pix_OgImage, y_pix_OgImage))
    #print('New pixels after centering with {}: xpix = {}, y_pix = {}'.format(how, x_pix_OgImage, y_pix_OgImage))
    #ra, dec = wcs_cal.pixelToSkyArray([x_pix_OgImage], [y_pix_OgImage], degrees=True)
    #print('The ra dec after centering: ',ra,dec )
    
    #obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    #obj_pos_2d = lsst.geom.Point2D(ra, dec)
    #x_pix, y_pix = x_pix_OgImage, y_pix_OgImage
    
    
    #if show_stamps:
    #    fig, ax = plt.subplots()
    #    m, s = np.mean(sub_data), np.std(sub_data)
    #    plt.title('Corrected coordinate',  fontsize=17)
    #    plt.imshow(data_cal, cmap='rocket', vmin=m, vmax=m + 4*s)
    #    plt.colorbar()
    #    plt.scatter(x_pix_OgImage,y_pix_OgImage, marker = 'x', color=neon_green, label = 'updated centroid')
    #    plt.scatter(x_pix,y_pix, marker = 'o', color='r', label = 'old coords')
    #    circle = plt.Circle((x_pix_OgImage,y_pix_OgImage), radius = 1/arcsec_to_pixel, color=neon_green, fill = False, linewidth=4)
    #    plt.gca().add_patch(circle)
    #    plt.xlim(x_pix_OgImage - side_half_length_box, x_pix_OgImage + side_half_length_box)
    #    plt.ylim(y_pix_OgImage - side_half_length_box, y_pix_OgImage + side_half_length_box)
    #    plt.legend()
    #    plt.show()
    #    
    return x_pix_OgImage, y_pix_OgImage


def get_light_curve(repo, visits, collection_diff, collection_calexp, ccd_num, ra, dec,field='', cutout=20, save=False, title='', hist=False, sparse_obs=False, SIBLING=None, save_as='', do_lc_stars = False, nstars=10, seedstars=200, save_lc_stars = False, show_stamps=True, show_star_stamps=False, r_star = 6, correct_coord=False, correct_coord_after_conv=False, do_zogy=False, collection_coadd=None, plot_zogy_stamps=False, plot_coadd=False, instrument='DECam', sfx='flx', save_stamps=False, well_subtracted=False, verbose=False, tp='after_ID', area=None, thresh=None, mfactor=1, do_convolution=True, mode='Eridanus', name_to_save='', type_kernel = 'mine', show_coord_correction=False, stars_from='lsst_pipeline', how_centroid = 'sep', path_to_folder= '/home/pcaceres/LSST_notebooks/Results/HiTS/SIBLING/', check_convolution=True,minarea=np.pi * (0.5/arcsec_to_pixel)**2, flux_thresh=None, ap_radii = np.array([0.5, 0.75, 1, 1.25, 1.5]), jname=None, cutout_star=23):
    
    """
    Does aperture photometry of the source in ra,dec position and plots the light curve.
    
    Input: 
    ------
    repo : [string] directory of the butler repository
    visits : [list] list of the exposures 
    collection_diff : [string] collection name of the image difference exposures 
    collection_calexp: [string] collection name of the calibrated images 
    ccd_num : [int] detector number from DECam
    ra : [float] Rigth ascention position in decimal degrees 
    dec : [float] Declination position in decimal degrees
    r : [float] aperture radii in arc-seconds
    cutout : [int] half pixels centered at the ra,dec where the stamp is plotted 
    save : [bool] if True, the plot of the light curve is saved 
    title : [string] title of the plot
    hist : [bool] if True, it shows the histogram of the difference images 
    sparse_obs : [bool] if True, the xlabel of the light curve is in logarithmic scale
    SIBING : [string] directory of the file to Jorge's light curve 
    save_as : [string] name of the plot figure that will be saved, by default this are stored in the light_curve/ folder
    do_lc_stars : [bool]
    nstars : [int]
    seedstars : [int]
    save_lc_stars : [bool]
    show_stamps : [bool]
    show_star_stamps : [bool]
    correct_coord : [bool]
    
    Output: 
    ------
    source_of_interest : [pd.DataFrame]
    
    """

    ####### Setting empty dictionaries and arrays ##################################################
    Data_science = {}
    Data_convol = {}
    Data_coadd = {}
    Data_diff = {}
    wcs_images = {}
    
    coords_coadd = {}
    coords_science = {}
    coords_convol = {}

    profiles = {}
    profiles_stars = {}    

    my_calib = []
    calib_lsst = []
    my_calib_inter = []
    calib_lsst_err = []

    calib_relative = []
    calib_relative_intercept = []

    Fluxes_unscaled = []
    Fluxes_err_unscaled = []
    
    Fluxes = []
    Fluxes_err = []

    Fluxes_cal = []
    Fluxeserr_cal = []

    Fluxes_annuli = []
    Fluxeserr_annuli = []

    Mag = []
    Magerr = [] 

    Seeing = []
    ExpTimes = []
    KERNEL = {}

    Fluxes_scaled = []
    Fluxes_err_scaled = []
    
    magzero = []

    flux_coadd=0
    fluxerr_coadd=0

    Mag_coadd = []
    Magerr_coadd = []
    Airmass = []

    Fluxes_njsky = []
    Fluxeserr_njsky = []
    
    stars = pd.DataFrame()
    
    flags = []
    
    stats = {}
    arcsec_to_pixel = 0.27#626 #arcsec/pixel, value from Manual of NOAO - DECam
    px = 2048.0
    py = 4096.0
    #r_in_arcsec = r_diff 
    
    
    flux_reference = 0
    calib_reference = 0 
    magzero_firstImage = 0
    zero_set = 0
    coadd_photocalib = 0
    butler = Butler(repo)

    data_for_hist = []
    
    scaling = []
    scaling_coadd = []
    magzero_reference = 0
    TOTAL_counts = 0
    TOTAL_convolved_counts = 0 
    
    RA_source = []
    DEC_source = []
    
    distance_me_lsst = []
    kernel_stddev = []
    
    ##############################################################################################################
    
    
    if mode=='HiTS' or mode=='HITS':
        files_to_sibling = '/home/pcaceres/LSST_notebooks/SIBLING_sources_usingMPfilter_andPCB_comparison.csv'
        sibling_dataset = pd.read_csv(files_to_sibling, index_col='internalID')
    
    
    # build the folders and subfolders for a given source 
    
    path_to_results = Path(path_to_folder)
    if len(title)<2:
        folder_field = path_to_results.joinpath('{}/'.format(field))
        folder_field.mkdir(parents=True, exist_ok=True)
        subfolder_detector_number = folder_field.joinpath('{}/'.format(ccd_name[ccd_num]))
        subfolder_detector_number.mkdir(parents=True, exist_ok=True)
        subsubfolder_source = subfolder_detector_number.joinpath('ra_{}_dec_{}/'.format(round(ra,3), round(dec,3)))
        subsubfolder_source.mkdir(parents=True, exist_ok=True)
    else:
        subsubfolder_source = path_to_results.joinpath('{}/'.format(title))
        subsubfolder_source.mkdir(parents=True, exist_ok=True)
        #subfolder_detector_number = folder_field.joinpath('{}/'.format(ccd_name[ccd_num]))
        #subfolder_detector_number.mkdir(parents=True, exist_ok=True)
        #subsubfolder_source = subfolder_detector_number.joinpath('ra_{}_dec_{}/'.format(round(ra,3), round(dec,3)))
        #subsubfolder_source.mkdir(parents=True, exist_ok=True)

    #########################
    
    # Here we sort the visits by the date 
    dates_aux, visits_aux = Order_Visits_by_Date(repo, visits, ccd_num, collection_diff)
    
    # Here we find the image with the worst seeing, and the associated visit number
    worst_seeing, worst_seeing_visit = Find_worst_seeing(repo, visits, ccd_num, collection_diff) # sigma in pixels
    # We retrieve the image that has the worst seeing
    worst_cal = butler.get('calexp', visit=worst_seeing_visit, detector=ccd_num , collections=collection_diff, instrument='DECam')
    worst_cal_array = np.asarray(worst_cal.image.array, dtype='float')
    worst_psf = worst_cal.getPsf() 
    worst_wcs = worst_cal.getWcs()
    sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
    
    round_magnitudes =  np.array([16, 17, 18, 19, 20, 21, 22])
    
    ####### apertures we are working with: ##############

    #rs_aux = r_science/arcsec_to_pixel
    #rd_aux = r_diff/arcsec_to_pixel 
    #rd_dyn = 2*seeing*0.75
        
    # Here we query or retrieve the stars we will measure the flux from
    if do_lc_stars:
        
        # ra dec of the center of the image... still dont remember what for 
    
        ra_center, dec_center = worst_wcs.pixelToSkyArray([px/2], [py/2], degrees=True)
        ra_center = ra_center[0]
        dec_center = dec_center[0]
        
        ra_corner, dec_corner = worst_wcs.pixelToSkyArray([px - 40.0, 40.0], [py - 40.0, 40.0], degrees=True)
        
        if stars_from=='PS1_DR2_query':
            TAP_service = vo.dal.TAPService("https://mast.stsci.edu/vo-tap/api/v0.1/ps1dr2/")
            job = TAP_service.run_async("""
                SELECT objID, RAMean, DecMean, nDetections, ng, nr, ni, nz, ny, gMeanPSFMag, rMeanPSFMag, iMeanPSFMag, zMeanPSFMag, yMeanPSFMag, gMeanKronMag, iMeanKronMag
                FROM dbo.MeanObjectView
                WHERE
                CONTAINS(POINT('ICRS', RAMean, DecMean), CIRCLE('ICRS', {}, {}, .13))=1
                AND nDetections >= 3
                AND ng > 0 AND ni > 0
                AND iMeanPSFMag < 21 AND iMeanPSFMag - iMeanKronMag < 0.05
                  """.format(ra_center, dec_center))
            TAP_results = job.to_table()
            
            RA_stars = np.array(TAP_results['ramean'])
            DEC_stars = np.array(TAP_results['decmean'])
            
            idx = np.where((RA_stars>ra_corner.min()) & (RA_stars<ra_corner.max()) & (DEC_stars>dec_corner.min()) & (DEC_stars < dec_corner.max()))
            
            RA_stars = RA_stars[idx]
            DEC_stars = DEC_stars[idx]
            
            ra_inds_sort = RA_stars.argsort()
            RA_stars = RA_stars[ra_inds_sort[::-1]]
            DEC_stars = DEC_stars[ra_inds_sort[::-1]]
            
        if stars_from=='lsst_pipeline':
            
            stars_table = Inter_Join_Tables_from_LSST(repo, visits, ccd_num, collection_diff)
            
            print('taking stars from: ', stars_from, ' we have: ', len(stars_table), ' stars')
            
            xx_pix_stars = stars_table['base_SdssCentroid_x_{}'.format(worst_seeing_visit)]
            yy_pix_stars = stars_table['base_SdssCentroid_y_{}'.format(worst_seeing_visit)]
            
            idx, = np.where((xx_pix_stars>40) & (xx_pix_stars<px-40) & (yy_pix_stars>40) & (yy_pix_stars < py - 40))
            
            stars_table = stars_table.loc[idx].reset_index()
            
            mags_stars_lsst = np.array(stars_table['base_PsfFlux_mag_{}'.format(worst_seeing_visit)])
            
            stars_within_mags, = np.where((mags_stars_lsst >=16) & (mags_stars_lsst <= 22))
            print('Within magnitudes 16 and 22, there are: {} stars'.format(len(stars_within_mags)))
            
            stars_table = stars_table.loc[stars_within_mags]
            
            #closest_indices = np.array([np.argmin(np.abs(mags_stars_lsst - mag)) for mag in round_magnitudes])
            print('number of stars: ', len(stars_table))
            #print('stars we will see their profiles from: ', closest_indices+1)
            
            
            #nstars = len(stars_table)
    
    ###### Here we set the pandas dataframe were the lightcurves are stored
    # flux_ImagDiff_nJy_0.5wseeing
    print('worst seeing visit: ', worst_seeing_visit)
    name_columns_imagdiff = ['flux_ImagDiff_nJy_{}sigmaPsf'.format(f) for f in ap_radii]
    name_columns_imagdiffErr = ['fluxErr_ImagDiff_nJy_{}sigmaPsf'.format(f) for f in ap_radii]
    name_columns_convdown = ['flux_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_convdownErr = ['fluxErr_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_inst_convdown = ['instflux_ConvDown_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_inst_convdownErr = ['instfluxErr_ConvDown_{}_arcsec'.format(f) for f in ap_radii]
    
    name_columns_inst = ['instflux_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_instErr = ['instfluxErr_{}_arcsec'.format(f) for f in ap_radii]

    name_columns_mconvdown = ['mag_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_mconvdownErr = ['magErr_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    
    columns = name_columns_imagdiff + name_columns_imagdiffErr + name_columns_convdown + name_columns_convdownErr + name_columns_mconvdown + name_columns_mconvdownErr + name_columns_inst + name_columns_instErr
    
    source_of_interest = pd.DataFrame(columns = columns)
    source_of_interest['dates'] = dates_aux
    
    do_diff_lc = True
    do_conv_lc = True
    # Here we loop over the images 
    
    for i in range(len(visits_aux)):
        skip_observation = False
        zero_set_aux = zero_set
        
        # Difference Image 
        try:
            diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        except LookupError:
            do_diff_lc = False
            print('there is no difference image')
           
        # Calibrated Image 
        try:
            calexp = butler.get('calexp', visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 
        except LookupError:
            do_conv_lc = False
        
        if do_conv_lc==False and do_diff_lc==False:
            print('there is no images to do LCs from')
            return
        
        if do_diff_lc:      
            try:
                # Template image from the difference Image method
                coadd = butler.get('goodSeeingDiff_matchedExp',visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
                print('took coadd from: goodSeeingDiff_matchedExp')
            except:
                coadd = butler.get('goodSeeingDiff_templateExp',visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
                print('took coadd from: goodSeeingDiff_templateExp')

        # Background of calibrated image 
        calexpbkg = butler.get('calexpBackground', visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 

        # We retrieve the wcs from the images 
        if do_diff_lc:
            wcs_coadd = coadd.getWcs()
            wcs = diffexp.getWcs()
        
        wcs_cal = calexp.getWcs()
        wcs_images['wcs_{}'.format(visits_aux[i])] = wcs_cal
        ###############################
        
        # Here get the psf of the images
        
        psf_calexp = calexp.getPsf() 
        seeing_calexp = psf_calexp.computeShape(psf_calexp.getAveragePosition()).getDeterminantRadius()

        if do_diff_lc:
            psf = diffexp.getPsf() 
            seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()# sigma in pixels 
            Seeing.append(seeing * sigma2fwhm * arcsec_to_pixel) # append to Seeing array as fwhm in arcsec
        else:
            seeing = seeing_calexp
            Seeing.append(seeing_calexp * sigma2fwhm * arcsec_to_pixel)
               
        ########################
        
        
        obj_pos_2d = lsst.geom.Point2D(ra, dec)
        wimageKernel = worst_psf.computeKernelImage(obj_pos_2d)
        imageKernel = psf_calexp.computeKernelImage(obj_pos_2d)    
        #im = imageKernel.array 
        #wim = wimageKernel.array 
        
        
        ################################
        # Here we append the exposure times         
        
        exp_visit_info = calexp.getInfo().getVisitInfo()
        ExpTime = exp_visit_info.exposureTime 
        
        ExpTimes.append(ExpTime)
        
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)
        print(visit_date_astropy)
        
        ###############################   
        
        # Here we get the airmass 
        
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        Airmass.append(airmass)
        
        ##############################
        
        # We retrieve the images as a numpy matrix format.
        data_cal = np.asarray(calexp.image.array, dtype='float')
        var_data_cal = np.asarray(calexp.variance.array, dtype='float')
        Data_science[visits_aux[i]] = data_cal
        TOTAL_counts = np.sum(np.sum(data_cal))
        
        if do_diff_lc:
            data = np.asarray(diffexp.image.array, dtype='float')
            Data_diff[visits_aux[i]] = data
            
            data_coadd = np.asarray(coadd.image.array, dtype='float')
            Data_coadd[visits_aux[i]] = data_coadd
        else:
            Data_diff[visits_aux[i]] = np.zeros(data_cal.shape)
            Data_coadd[visits_aux[i]] = np.zeros(data_cal.shape)
        

        data_cal_bkg = np.asarray(calexpbkg.getImage().array,dtype='float')
        bkg_rms = np.sqrt(np.mean((data_cal_bkg.flatten()- np.mean(data_cal_bkg.flatten()))**2))

        #####################
        
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        x_pix, y_pix = wcs_cal.skyToPixel(obj_pos_lsst)
        
        
        # If we choose to correct coord, we update x_pix and y_pix
        if correct_coord:
            x_pix, y_pix = centering_coords(data_cal, x_pix, y_pix, cutout, show_coord_correction, how=how_centroid, minarea=minarea, flux_thresh=flux_thresh)
            ra, dec = wcs_cal.pixelToSkyArray(x_pix, y_pix, degrees=True)
        else:
            x_pix, y_pix = [x_pix], [y_pix] 
        
        # We get the x and y pixels corresponding to the ra dec of the images 
        
        obj_pos_2d = lsst.geom.Point2D(ra, dec)
        #obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        #x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)  # == to the wcs from the cal_image 
        coords_science[visits_aux[i]] = [x_pix, y_pix]
       
        if do_diff_lc:
            x_pix_coadd, y_pix_coadd = wcs_coadd.skyToPixel(obj_pos_lsst)
            coords_coadd[visits_aux[i]] = [x_pix_coadd, y_pix_coadd]
        else:
            coords_coadd[visits_aux[i]] = coords_science[visits_aux[i]]
        
        ############################################        

        print('xpix, y_pix: {} {} '.format(x_pix, y_pix))

        if do_convolution:
            
            stars_between_two_exp = Inter_Join_Tables_from_LSST(repo, [visits_aux[i],worst_seeing_visit], ccd_num, collection_diff)
            # mags_stars_lsst = stars_table['base_PsfFlux_mag_{}'.format(worst_seeing_visit)]
            
            stars_between_two_exp = stars_between_two_exp[(stars_between_two_exp['base_PsfFlux_mag_{}'.format(worst_seeing_visit)] >= 16) & (stars_between_two_exp['base_PsfFlux_mag_{}'.format(worst_seeing_visit)] <= 21)]
            
                        
            wim, _= check_psf_after_conv(repo, worst_seeing_visit, ccd_num, collection_diff, conv_image_array = None, plot=False, sn=20, cutout=30, isolated=True, dist_thresh=30)

            im, _= check_psf_after_conv(repo, visits_aux[i], ccd_num, collection_diff, conv_image_array = None, plot=False, sn=20, cutout=30, isolated=True, dist_thresh=30)
            
            print('Number of stars to make the conv: ', len(stars_between_two_exp))
            
            calConv_image, calConv_variance, kernel = do_convolution_image(data_cal, var_data_cal, im, wim, mode=mode, type_kernel=type_kernel, visit=visits_aux[i], worst_visit=worst_seeing_visit, stars_in_common=stars_between_two_exp, worst_calexp=worst_cal_array, calexp_exposure = calexp, worst_calexp_exposure=worst_cal)# (calexp, worst_cal, ra, dec, mode=mode, type_kernel=type_kernel, visit=visits_aux[i], worst_visit=worst_seeing_visit, stars_in_common=stars_table, im=im, wim=wim)
            
            TOTAL_convolved_counts = np.sum(np.sum(calConv_image))
            print('fraction of Flux lost after convolution: ',1-TOTAL_convolved_counts/TOTAL_counts)

            KERNEL[visits_aux[i]] = kernel
            Data_convol[visits_aux[i]] = calConv_image
            
            #np.save('kernel_for_exp{}_Blind15A_16_N24.npy'.format(visits_aux[i]), kernel)
            
            
            detectionTask = SourceDetectionTask()
            if correct_coord_after_conv:
                
                x_pix_conv, y_pix_conv = centering_coords(calConv_image, x_pix[0], y_pix[0], cutout, show_coord_correction,  how=how_centroid, minarea=minarea, flux_thresh=flux_thresh)
                coords_convol[visits_aux[i]] = [x_pix_conv, y_pix_conv]
            else:
                x_pix_conv = x_pix
                y_pix_conv = y_pix
                coords_convol[visits_aux[i]] = [x_pix, y_pix]
            print('-----------------')

            #if check_convolution:
                
                
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
        ############# Doing photometry step ###############################################

        if do_diff_lc:
            photocalib = diffexp.getPhotoCalib()
            photocalib_coadd = coadd.getPhotoCalib()
            diffexp_calibrated = photocalib.calibrateImage(diffexp.getMaskedImage())
            diffexp_calib_array = np.asarray(diffexp_calibrated.image.array, dtype='float')
        
        photocalib_cal = calexp.getPhotoCalib()
        

        calib_image = photocalib_cal.getCalibrationMean()
        calib_image_err = photocalib_cal.getCalibrationErr()
        calib_lsst.append(calib_image)
        calib_lsst_err.append(calib_image_err)
        
        
        stamp_gal = calexp.getCutout(lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees), size=lsst.geom.Extent2I(7,7))

        mask_stamp = np.asarray(stamp_gal.maskedImage.mask.array, dtype='int32')
        
        #number_of_sat_pixels = np.sum(np.array([["1" in str(element) for element in row] for row in mask_stamp])) # saturated 
        number_of_edge_pixels = np.sum(np.array([["4" in str(element) for element in row] for row in mask_stamp])) # EDGE
        number_of_ND_pixels = np.sum(np.array([["8" in str(element) for element in row] for row in mask_stamp])) # No DATA
        
        
        if number_of_edge_pixels + number_of_ND_pixels > 0:
            
            #print('number of sat pixels: ',number_of_sat_pixels ) 
            print('number of edge pixels: ', number_of_edge_pixels) 
            print('number of no data pixels: ', number_of_ND_pixels) 
            
            print('photometric point is not valid')
            pass
        
        else:
            
            if do_diff_lc:
                

                flux_diff, fluxerr_diff, flag_diff = sep.sum_circle(diffexp_calib_array, [x_pix], [y_pix], ap_radii / arcsec_to_pixel, var = np.asarray(diffexp_calibrated.variance.array, dtype='float')) # fixed aperture 

                source_of_interest.loc[i,name_columns_imagdiff] = flux_diff.flatten()
                source_of_interest.loc[i,name_columns_imagdiffErr] = fluxerr_diff.flatten()


            flux_sci, fluxerr_sci, flag_sci = sep.sum_circle(data_cal, [x_pix], [y_pix], ap_radii / arcsec_to_pixel, var = np.asarray(calexp.variance.array, dtype='float')) # fixed aperture 

            source_of_interest.loc[i,name_columns_inst] = flux_sci.flatten()
            source_of_interest.loc[i,name_columns_instErr] = fluxerr_sci.flatten()

            print('Aperture radii in px: ', ap_radii / arcsec_to_pixel)
            print('Aperture radii in arcsec: ', ap_radii )

            if do_convolution:

                if type_kernel == 'panchos':

                    dx_stamp = 75
                    if (x_pix_conv[0]<dx_stamp) or (x_pix_conv[0]>px-dx_stamp) or (y_pix_conv[0]<dx_stamp) or (y_pix_conv[0]> py - dx_stamp):
                        
                        dx_stamp = int(np.min(np.array([x_pix_conv[0], y_pix_conv[0], px - x_pix_conv[0], py - y_pix_conv[0]]))) - 1
                                            
                    
                    convolved_stamp = calConv_image[round(y_pix_conv[0])-dx_stamp:round(y_pix_conv[0])+dx_stamp+1,round(x_pix_conv[0])-dx_stamp:round(x_pix_conv[0])+dx_stamp+1].copy(order='C')
                    variance_stamp = calConv_variance[round(y_pix_conv[0])-dx_stamp:round(y_pix_conv[0])+dx_stamp+1,round(x_pix_conv[0])-dx_stamp:round(x_pix_conv[0])+dx_stamp+1].copy(order='C')
#

                    print('x pix and y pix: ', x_pix_conv[0], y_pix_conv[0])
                    print(np.shape(convolved_stamp))
                    try:
                        sigma_clip = SigmaClip(sigma=3., maxiters=2)
                        bkg_estimator = MedianBackground()
                        bkg = Background2D(convolved_stamp, (10, 10), sigma_clip=sigma_clip,
                               bkg_estimator=bkg_estimator)
                        data_image_wout_bkg = convolved_stamp - bkg.background
                    
                    except:
                        print('background subtraction failed.. so we skip this part')
                        data_image_wout_bkg = convolved_stamp
                    
                    x_pix_stamp = x_pix_conv[0] - round(x_pix_conv[0]) + dx_stamp
                    y_pix_stamp = y_pix_conv[0] - round(y_pix_conv[0]) + dx_stamp
                    
                    
                    flux_conv, fluxerr_conv, flux_convFlag = sep.sum_circle(data_image_wout_bkg, [x_pix_stamp], [y_pix_stamp], worst_seeing * sigma2fwhm * ap_radii, var=variance_stamp, gain=4.0)
                    flux_conv = [flux_conv]
                    fluxerr_conv = [fluxerr_conv]
                    #data_point = get_photometry(data_image_wout_bkg, bkg=bkg, pos=(dx_stamp,dx_stamp), radii= ap_radii / arcsec_to_pixel, sigma1 = bkg.background_rms_median, alpha=0.85, beta=1.133, centered=True, iter=i)

                    #flux_conv = [[data_point['aperture_sum_%i' % k][0] for k in range(len(ap_radii))]]
                    #fluxerr_conv = [[data_point['aperture_flx_err_%i' % k][0] for k in range(len(ap_radii))]]

                    #print(data_point)
                    #plt.imshow(data_image_wout_bkg)
                    #plt.colorbar()
                    #plt.scatter(dx_stamp, dx_stamp, color='red', marker='x')
                    #plt.title('background subtracted')
                    #
                    #plt.show()
                    #
                    #plt.imshow(bkg.background)
                    #plt.colorbar()
                    #plt.scatter(dx_stamp, dx_stamp, color='red', marker='x')
                    #plt.title('Background')
                    #plt.show()
                    #
                    #
                    #plt.imshow(convolved_stamp)
                    #plt.colorbar()
                    #plt.scatter(dx_stamp, dx_stamp, color='red', marker='x')
                    #plt.title('convolved image')
                    #plt.show()
                    #print('flux_conv shape : ',flux_conv.shape)                                
                else:
                    flux_conv, fluxerr_conv, flux_convFlag = sep.sum_circle(calConv_image, [x_pix_conv], [y_pix_conv], ap_radii / arcsec_to_pixel, var=calConv_variance, gain=4.0)
                    #print('flux_conv shape : ',flux_conv.shape)
                fluxesConv_to_nJy = [photocalib_cal.instFluxToNanojansky(f, ferr, obj_pos_2d) for f, ferr in zip(flux_conv[0], fluxerr_conv[0])]

                fluxConv_nJy = [f.value for f in fluxesConv_to_nJy]
                fluxerrConv_nJy = [f.error for f in fluxesConv_to_nJy]

                source_of_interest.loc[i,name_columns_convdown] = fluxConv_nJy
                source_of_interest.loc[i,name_columns_convdownErr] = fluxerrConv_nJy

                source_of_interest.loc[i,name_columns_inst_convdown] = np.array(flux_conv).flatten()
                source_of_interest.loc[i,name_columns_inst_convdownErr] = np.array(fluxerr_conv).flatten()

                magsConv_to_nJy = [photocalib_cal.instFluxToMagnitude(f, ferr, obj_pos_2d) for f, ferr in zip(flux_conv[0], fluxerr_conv[0])]

                magsConv_nJy = [f.value for f in magsConv_to_nJy]
                magserrConv_nJy = [f.error for f in magsConv_to_nJy]

                source_of_interest.loc[i,name_columns_mconvdown] = magsConv_nJy
                source_of_interest.loc[i,name_columns_mconvdownErr] = magserrConv_nJy
            
        ############################ end of galaxy photometry #########################################################
        
        if show_stamps:
            if do_diff_lc:
                Calib_Diff_and_Coadd_plot_cropped_astropy(repo, collection_diff, ra, dec, [visits_aux[i]], ccd_num, s=worst_seeing * sigma2fwhm, sd=seeing * sigma2fwhm, conv_image= calConv_image, cutout=cutout, field=field, name=title)
            else:
                dx_stamp = 50
                convolved_stamp = calConv_image[round(y_pix_conv[0])-dx_stamp:round(y_pix_conv[0])+dx_stamp+1,round(x_pix_conv[0])-dx_stamp:round(x_pix_conv[0])+dx_stamp+1]
                m, s = np.mean(convolved_stamp), np.std(convolved_stamp)
                plt.imshow(calConv_image, vmin=m, vmax = m+5*s)
                plt.scatter(x_pix_conv[0], y_pix_conv[0], marker='x', color='red')
                plt.xlim(x_pix_conv[0]-dx_stamp,x_pix_conv[0]+dx_stamp) 
                plt.ylim(y_pix_conv[0]-dx_stamp,y_pix_conv[0]+dx_stamp)
                plt.title('convolved image')
                plt.show()
                
                #m, s = np.mean(data_cal), np.std(data_cal)
                #lt.imshow(calConv_image, vmin=m, vmax = m+3*s)
                #lt.xlim(x_pix-20,x_pix+20) 
                #lt.ylim(y_pix-20,y_pix+20)
                #lt.title('science image')
                #lt.show()
                
        prof = flux_profile_array(calConv_image, x_pix_conv, y_pix_conv, 0.05, 6)
        
        profiles['{}'.format(visits_aux[i])] = prof/max(prof)
        
        
        if do_lc_stars == True:
            
            if stars_from == 'lsst_pipeline':
                RA_stars = np.array(stars_table['coord_ra_ddegrees_{}'.format(visits_aux[i])], dtype=float)
                DEC_stars = np.array(stars_table['coord_dec_ddegrees_{}'.format(visits_aux[i])], dtype=float)   
                
                #idx = np.where((RA_stars>ra_corner.min()) & (RA_stars<ra_corner.max()) & (DEC_stars>dec_corner.min()) & (DEC_stars < dec_corner.max()))
                #
                #RA_stars = RA_stars[idx]
                #DEC_stars = DEC_stars[idx]
                
                #ra_inds_sort = RA_stars.argsort()
                #RA_stars = RA_stars[ra_inds_sort[::-1]]
                #DEC_stars = DEC_stars[ra_inds_sort[::-1]]
                
            obj_pos_lsst_array = [lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees) for ra, dec in zip(RA_stars,DEC_stars)]
            pixel_stars_coords = [wcs_cal.skyToPixel(obj_pos_lsst) for obj_pos_lsst in obj_pos_lsst_array]
            x_pix_stars = np.array([p[0] for p in pixel_stars_coords])
            y_pix_stars = np.array([p[1] for p in pixel_stars_coords])

                
            nstars = len(x_pix_stars)
            print('Number of stars: ', nstars)
                
            # If its the first image loop, we create the pandas df
            
            star_aperture = r_star # arcseconds #r_star * fwhm/2
            star_aperture/=arcsec_to_pixel # transform it to pixel values 
            
            if i==0:
                
                
                
                columns_stars_convDown_instfluxes = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_instflx'.format(i+1) for i in range(nstars)])))
                columns_stars_convDown_instfluxesErr = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_instflx'.format(i+1) for i in range(nstars)])))
                
                columns_stars_convDown_fluxes = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_fnJy'.format(i+1) for i in range(nstars)])))
                columns_stars_convDown_fluxesErr = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_fnJy_err'.format(i+1) for i in range(nstars)])))
                
                columns_stars_convDown_mag = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_mag'.format(i+1) for i in range(nstars)])))
                columns_stars_convDown_magErr = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_magErr'.format(i+1) for i in range(nstars)])))
                
                if do_diff_lc:
                    columns_stars_imagDiff_fluxes = list(np.ndarray.flatten(np.array([ 'star_{}_ImagDiff_fnJy'.format(i+1) for i in range(nstars)])))
                    columns_stars_imagDiff_fluxesErr = list(np.ndarray.flatten(np.array([ 'star_{}_ImagDiff_fnJy_err'.format(i+1) for i in range(nstars)])))

                    stars_calc_byme = pd.DataFrame(columns=columns_stars_imagDiff_fluxes + columns_stars_imagDiff_fluxesErr + columns_stars_convDown_fluxes + columns_stars_convDown_fluxesErr + columns_stars_convDown_mag + columns_stars_convDown_magErr)
                else:
                    stars_calc_byme = pd.DataFrame(columns= columns_stars_convDown_fluxes + columns_stars_convDown_fluxesErr + columns_stars_convDown_mag + columns_stars_convDown_magErr)
                    
            
            #fs, fs_err, fg = sep.sum_circle(diffexp_calib_array, x_pix_stars, y_pix_stars, star_aperture, var = np.asarray(diffexp_calibrated.variance.array, dtype='float'))
            #stars_calc_byme.loc[i,columns_stars_imagDiff_fluxes] = fs
            #stars_calc_byme.loc[i,columns_stars_imagDiff_fluxesErr] = fs_err
            
            saturated_star = []
            
            if do_convolution:
                
                # the centering I do not do it, but maybe I will have tooooo!
                fluxConv_nJy = []
                fluxerrConv_nJy = []
                
                magsConv_nJy = []
                magserrConv_nJy = []
                
                if do_diff_lc:
                    fluxDiff_nJy = []
                    fluxerrDiff_nJy = []
                
                
                for k in range(nstars):
                    
                    x_pix_1star = x_pix_stars[k]
                    y_pix_1star = y_pix_stars[k]
                    
                    # Here we check for stars that are saturated, and skip them in case they are
                    
                    str_digit = '1' # number when the pixel is saturated
                    obj_pos_lsst_star = obj_pos_lsst_array[k]
                    calexp_star_cutout = calexp.getCutout(obj_pos_lsst_star, size=lsst.geom.Extent2I(star_aperture, star_aperture))
                    number_of_sat_pixels = np.sum(np.array([[str_digit in str(element) for element in row] for row in calexp_star_cutout.getMask().array]))
                    
                    if number_of_sat_pixels > 0:
                        print('saturated star, so I skip on doing their photometry')
                        fluxConv_nJy.append(np.nan)
                        fluxerrConv_nJy.append(np.nan)
                        magsConv_nJy.append(np.nan)
                        magserrConv_nJy.append(np.nan)
                        saturated_star.append(k)
                        
                        if do_diff_lc:
                            fluxDiff_nJy.append(np.nan)
                            fluxerrDiff_nJy.append(np.nan)
                        
                        
                        continue
                    
                    ##################### In difference image #######################################
                    
                    if do_diff_lc:
                        x_pix_new_s, y_pix_new_s = centering_coords(data_cal, x_pix_1star, y_pix_1star, cutout_star, show_stamps=False, how='sep', minarea=3)
                        fd, fd_err, fg = sep.sum_circle(diffexp_calib_array, x_pix_new_s, y_pix_new_s, star_aperture, var = np.asarray(diffexp_calibrated.variance.array, dtype='float'), gain=4.0)

                        fluxDiff_nJy.append(fd)
                        fluxerrDiff_nJy.append(fd_err)
                    
                    ##################### In convolved image #######################################
                    x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_1star, y_pix_1star, cutout_star, show_stamps=False, how='sep', minarea=3)
                    
                    f_conv, ferr_conv, f_convFlag = sep.sum_circle(calConv_image, x_pix_new, y_pix_new, star_aperture, var=calConv_variance, gain=4.0)
                    
                        
                    fluxesConv_to_nJy = photocalib_cal.instFluxToNanojansky(f_conv, ferr_conv, obj_pos_2d)

                    fluxConv_nJy.append(fluxesConv_to_nJy.value)
                    fluxerrConv_nJy.append(fluxesConv_to_nJy.error)
                    
                    magsConv_to_nJy = photocalib_cal.instFluxToMagnitude(f_conv, ferr_conv, obj_pos_2d) # for f, ferr in zip(f_conv, ferr_conv)]
                    
                    magsConv_nJy.append(magsConv_to_nJy.value)
                    magserrConv_nJy.append(magsConv_to_nJy.error)
                    
                    if show_star_stamps and magsConv_to_nJy.value<=np.median(magsConv_nJy)+0.5 and magsConv_to_nJy.value> np.median(magsConv_nJy)-0.5:
                        
                        plt.imshow(np.arcsinh(calConv_image))
                        plt.xlim(x_pix_new - cutout_star, x_pix_new + cutout_star)
                        plt.ylim(y_pix_new - cutout_star, y_pix_new + cutout_star)
                        
                        plt.title('star number {} in convolved image'.format(k+1))
                        plt.colorbar()
                        plt.show()
                        
                        plt.imshow(np.arcsinh(diffexp_calib_array))
                        plt.xlim(x_pix_1star - cutout_star, x_pix_1star + cutout_star)
                        plt.ylim(y_pix_1star - cutout_star, y_pix_1star + cutout_star)
                        
                        plt.title('star number {} in difference image'.format(k+1))
                        plt.colorbar()
                        plt.show()
                #print('now I center, and the shape of this array is : ',np.array(fluxConv_nJy).shape)
                
                stars_calc_byme.loc[i,columns_stars_convDown_fluxes] = fluxConv_nJy
                stars_calc_byme.loc[i,columns_stars_convDown_fluxesErr] = fluxerrConv_nJy
                
                # has a bad name...
                
                

                #magsConv_nJy = [f.value for f in magsConv_to_nJy]
                #magserrConv_nJy = [f.error for f in magsConv_to_nJy]

                stars_calc_byme.loc[i,columns_stars_convDown_mag] = magsConv_nJy
                stars_calc_byme.loc[i,columns_stars_convDown_magErr] = magserrConv_nJy
                if do_diff_lc:
                    stars_calc_byme.loc[i,columns_stars_imagDiff_fluxes] = fluxDiff_nJy
                    stars_calc_byme.loc[i,columns_stars_imagDiff_fluxesErr] = fluxerrDiff_nJy
                
            if show_star_stamps:
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 42))
                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')
                m, s = np.mean(data_cal), np.std(data_cal)
                im1 = ax1.imshow(data_cal, vmin = m-s, vmax = m+s)
                im2 = ax2.imshow(calConv_image, vmin = m-s, vmax = m+s)
                ax1.set_title('Calibrated Image')
                ax2.set_title('Convolved Image')
                
                md, sd = np.mean(diffexp_calib_array), np.std(diffexp_calib_array)
                im3 = ax3.imshow(diffexp_calib_array, vmin=md-sd, vmax = md+sd)
                ax3.set_title('Difference Image')
                
                for s in range(nstars):
                    x_star, y_star = x_pix_stars[s], y_pix_stars[s]
                    ax1.add_patch(plt.Circle((x_star, y_star), radius=star_aperture, color=neon_green, fill=False))
                    ax1.text(x_star+star_aperture, y_star, '{}'.format(s+1), color=neon_green)
                    
                    ax1.add_patch(plt.Circle((x_pix, y_pix), radius=star_aperture, color='m', fill=False))
                    ax1.text(x_pix+star_aperture, y_pix, 'Galaxy', color='m')
                    
                    
                    ax2.add_patch(plt.Circle((x_star, y_star), radius=star_aperture, color=neon_green, fill=False))
                    ax2.text(x_star+star_aperture, y_star, '{}'.format(s+1), color=neon_green)
                    
                    ax2.add_patch(plt.Circle((x_pix_conv, y_pix_conv), radius=star_aperture, color='m', fill=False))
                    ax2.text(x_pix_conv+star_aperture, y_pix_conv, 'Galaxy', color='m')

                    
                    ax3.add_patch(plt.Circle((x_star, y_star), radius=star_aperture, color=neon_green, fill=False))
                    ax3.text(x_star+star_aperture, y_star, '{}'.format(s+1), color=neon_green)
                    
                    ax3.add_patch(plt.Circle((x_pix, y_pix), radius=star_aperture, color='m', fill=False))
                    ax3.text(x_pix+star_aperture, y_pix, 'Galaxy', color='m')


                plt.show()
    
    
    #print(profiles_stars)

    # plotting airmas, seeing & calibration factor ############################################

    plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, calib_relative_intercept, '*', color='black', label='My calib', linestyle = '--')
    plt.errorbar(dates_aux, np.array(calib_lsst)/4, yerr = calib_lsst_err, capsize=3, fmt='o', color='blue', linestyle='--', label='Scaling factor for calibration / 4')
    
    #plt.xlabel('MJD', fontsize=17)
    #plt.ylabel('Calibration intercept', fontsize=17)
    #plt.title('Calibration scaling intercept relative to first visit', fontsize=17)
    #plt.legend()
    #plt.show()


    # Airmass plot
    #plt.figure(figsize=(10,6))
    plt.plot(dates_aux, np.array(Airmass), 'o', color='magenta', linestyle='--', label='Airmass')
    #plt.title('Airmass', fontsize=17)
    #plt.xlabel('MJD', fontsize=17)
    #plt.ylabel('Airmass', fontsize=17)
    #plt.show()


    # Seeing plot
    plt.plot(dates_aux, np.array(Seeing), 'o', color='black', linestyle='--', label='FWHM (arcsec)')
    #if correct_coord:
    #    #plt.plot(dates_aux, np.array(distance_me_lsst),'^', color='blue', linestyle = '--', label= 'distance LSST and sep')
    #if do_convolution:
    #    plt.plot(dates_aux, kernel_stddev, 'd', linestyle='dotted', color = 'm', label='sigma kernel')
    #plt.title('seeing sigma observation', fontsize=17)
    
    
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('In their given scale', fontsize=17)
    plt.legend(frameon=False)
    plt.show()
    
    ######################################################################################
    ######################################################################################
    ######################################################################################
    
    
    norm = matplotlib.colors.Normalize(vmin=0,vmax=nstars)
    c_m = matplotlib.cm.plasma

    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])


    fluxconv_star_median = np.array([np.median(stars_calc_byme['star_{}_convDown_fnJy'.format(i+1)]) for i in range(nstars)])
    fluxconv_star_min = np.array([np.min(stars_calc_byme['star_{}_convDown_fnJy'.format(i+1)]) for i in range(nstars)])
    fluxconv_star_max = np.array([np.max(stars_calc_byme['star_{}_convDown_fnJy'.format(i+1)]) for i in range(nstars)])
    
    if do_diff_lc:
        fluxdiff_star_mean = np.array([np.mean(stars_calc_byme['star_{}_ImagDiff_fnJy'.format(i+1)]) for i in range(nstars)])
    
    magsconv_star_mean = np.array([np.mean(stars_calc_byme['star_{}_convDown_mag'.format(i+1)]) for i in range(nstars)])
    magsconv_star_std = np.array([np.std(stars_calc_byme['star_{}_convDown_mag'.format(i+1)]) for i in range(nstars)])


    
    # We only take the stars that were correctly subtracted in the image difference 
    ##### filtering stars ! ¨good stars¨ do not increase their flux to more than half their median maximally
    
    keep_stars, = np.where(fluxconv_star_max/fluxconv_star_median < 1.5)
    
    #saturated_star
    
    idx_nan_mags, = np.where(np.isnan(magsconv_star_mean))
    idx_var_mags, = np.where(magsconv_star_std > 1.0)
    print('these stars are >1mag variable: ', idx_var_mags)
    id_good_stars = np.array([item for item in range(nstars) if item not in idx_nan_mags])
    id_good_stars = np.array([item for item in id_good_stars if item not in idx_var_mags])
    id_good_stars = np.array([item for item in id_good_stars if item not in saturated_star])

    print('id_good_stars: ', id_good_stars)
    print('saturated star: ', saturated_star)
    ##################################################################################
    
    mags_good_stars = magsconv_star_mean[id_good_stars]
    x_pix_stars = x_pix_stars[id_good_stars]
    y_pix_stars = y_pix_stars[id_good_stars]
    RA_stars = RA_stars[id_good_stars]
    DEC_stars = DEC_stars[id_good_stars]
    
    closest_indices = np.array([np.argmin(np.abs(mags_good_stars - mag)) for mag in round_magnitudes])
    
    column_w_mags = 'mag_ConvDown_nJy_{}_arcsec'.format(1.0)
    mean_conv_mag = np.mean(np.array(source_of_interest[column_w_mags]))
    ids_stars_within_gal_mag, = np.where((magsconv_star_mean >= mean_conv_mag-0.5) & (magsconv_star_mean < mean_conv_mag+0.5))
    ids_stars_within_gal_mag = np.array([item for item in ids_stars_within_gal_mag if item in id_good_stars])

    #ids_stars_within_gal_mag, = np.where((mags_good_stars >= mean_conv_mag-0.5) & (mags_good_stars < mean_conv_mag+0.5))
    n_stars_within_gal_mag = len(ids_stars_within_gal_mag)
    
    print('Number of stars within the gal mag: ', n_stars_within_gal_mag )
    
    #objpos_good_stars = np.array(obj_pos_lsst_array)[id_good_stars] 
    
    #obj_pos_lsst_array = [lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees) for ra, dec in zip(RA_stars,DEC_stars)]
    #pixel_good_stars_coords = [wcs_cal.skyToPixel(obj) for obj in objpos_good_stars]
    #x_pix_stars = np.array([p[0] for p in pixel_stars_coords])
    #y_pix_stars = np.array([p[1] for p in pixel_stars_coords])
    
    for ii in range(len(visits_aux)):
        # de 0 a 6
        #wcs_cal = wcs_im[ii]
        wcs = wcs_images['wcs_{}'.format(visits_aux[ii])]
        calConv_image = Data_convol[visits_aux[ii]]
 

        idx_star = closest_indices[0]
        ra, dec = RA_stars[idx_star], DEC_stars[idx_star]
        obj = lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees)
        pixel_coords = wcs.skyToPixel(obj)
        x_pix_star = pixel_coords[0]
        y_pix_star = pixel_coords[1]
        x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_star, y_pix_star, cutout_star, show_stamps=False, how='sep', minarea=3)        
        prof = flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[0], visits_aux[ii])] = prof/max(prof)
        
        idx_star = closest_indices[1] 
        ra, dec = RA_stars[idx_star], DEC_stars[idx_star]
        obj = lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees)
        pixel_coords = wcs.skyToPixel(obj)
        x_pix_star = pixel_coords[0]
        y_pix_star = pixel_coords[1]
        x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_star, y_pix_star, cutout_star, show_stamps=False, how='sep', minarea=3)
        prof = flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[1], visits_aux[ii])] = prof/max(prof)
        
        
        idx_star = closest_indices[2] 
        ra, dec = RA_stars[idx_star], DEC_stars[idx_star]
        obj = lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees)
        pixel_coords = wcs.skyToPixel(obj)
        x_pix_star = pixel_coords[0]
        y_pix_star = pixel_coords[1]
        x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_star, y_pix_star, cutout_star, show_stamps=False, how='sep', minarea=3)
        prof = flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[2], visits_aux[ii])] = prof/max(prof)
        
        
        idx_star = closest_indices[3]
        ra, dec = RA_stars[idx_star], DEC_stars[idx_star]
        obj = lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees)
        pixel_coords = wcs.skyToPixel(obj)
        x_pix_star = pixel_coords[0]
        y_pix_star = pixel_coords[1]
        x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_star, y_pix_star, cutout_star, show_stamps=False, how='sep', minarea=3)
        prof = flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[3], visits_aux[ii])] = prof/max(prof)
        
        idx_star = closest_indices[4]
        ra, dec = RA_stars[idx_star], DEC_stars[idx_star]
        obj = lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees)
        pixel_coords = wcs.skyToPixel(obj)
        x_pix_star = pixel_coords[0]
        y_pix_star = pixel_coords[1]
        x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_star, y_pix_star, cutout_star, show_stamps=False, how='sep', minarea=3)
        prof = flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[4], visits_aux[ii])] = prof/max(prof)
        
        
        idx_star = closest_indices[5]
        ra, dec = RA_stars[idx_star], DEC_stars[idx_star]
        obj = lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees)
        pixel_coords = wcs.skyToPixel(obj)
        x_pix_star = pixel_coords[0]
        y_pix_star = pixel_coords[1]
        x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_star, y_pix_star, cutout_star, show_stamps=False, how='sep', minarea=3)
        prof = flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[5], visits_aux[ii])] = prof/max(prof)
                
        idx_star = closest_indices[6]
        ra, dec = RA_stars[idx_star], DEC_stars[idx_star]
        obj = lsst.geom.SpherePoint(ra,dec, lsst.geom.degrees)
        pixel_coords = wcs.skyToPixel(obj)
        x_pix_star = pixel_coords[0]
        y_pix_star = pixel_coords[1]
        x_pix_new, y_pix_new = centering_coords(calConv_image, x_pix_star, y_pix_star, cutout_star, show_stamps=False, how='sep', minarea=3)
        prof = flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[6], visits_aux[ii])] = prof/max(prof)
    
    
    # To plot the variability of the fluxes of stars in the convolved image :
    Results_star = pd.DataFrame(columns = ['date', 'stars_science_1sigmalow_byEpoch', 'stars_science_1sigmaupp_byEpoch', 'stars_science_2sigmalow_byEpoch', 'stars_science_2sigmaupp_byEpoch', 'stars_diff_1sigmalow_byEpoch', 'stars_diff_1sigmaupp_byEpoch', 'stars_diff_2sigmalow_byEpoch', 'stars_diff_2sigmaupp_byEpoch', 'stars_scienceMag_1sigmaupp_byEpoch', 'stars_scienceMag_2sigmaupp_byEpoch', 'stars_scienceMag_1sigmalow_byEpoch', 'stars_scienceMag_2sigmalow_byEpoch'])
    
    #stars_calc_byme = stars_calc_byme.dropna(axis='columns')
    
    stars_science_mag_columns = ['star_{}_convDown_mag'.format(i+1) for i in ids_stars_within_gal_mag]
    stars_science_mag = stars_calc_byme[stars_science_mag_columns].dropna(axis=1, how='any')
    print('stars science magnitude df: ', stars_science_mag)
    stars_science_mag -= stars_science_mag.mean()
    stars_science_disp = np.array([np.std(np.array(stars_science_mag.loc[i])) for i in range(len(stars_science_mag))])
    print('stars science dispersion', stars_science_disp)
    
    stars_sciencemag_1sigmalow_byEpoch = - stars_science_disp
    stars_sciencemag_1sigmaupp_byEpoch = stars_science_disp
    stars_sciencemag_2sigmalow_byEpoch = - 2 * stars_science_disp
    stars_sciencemag_2sigmaupp_byEpoch = 2 * stars_science_disp
    
    Results_star['stars_scienceMag_1sigmalow_byEpoch'] = stars_sciencemag_1sigmalow_byEpoch
    Results_star['stars_scienceMag_1sigmaupp_byEpoch'] = stars_sciencemag_1sigmaupp_byEpoch
    Results_star['stars_scienceMag_2sigmalow_byEpoch'] = stars_sciencemag_2sigmalow_byEpoch
    Results_star['stars_scienceMag_2sigmaupp_byEpoch'] = stars_sciencemag_2sigmaupp_byEpoch
    
    #print(Results_star)
    
    #########################################
    
    stars_science_flux_columns = ['star_{}_convDown_fnJy'.format(i+1) for i in ids_stars_within_gal_mag]
    stars_science_flux = stars_calc_byme[stars_science_flux_columns]
    stars_science_flux -= stars_science_flux.mean()

    stars_science_1sigmalow_byEpoch = - np.array([np.std(np.array(stars_science_flux.loc[i])) for i in range(len(stars_science_flux))])
    stars_science_1sigmaupp_byEpoch = np.array([np.std(np.array(stars_science_flux.loc[i])) for i in range(len(stars_science_flux))])
    stars_science_2sigmalow_byEpoch = - np.array([2*np.std(np.array(stars_science_flux.loc[i])) for i in range(len(stars_science_flux))])
    stars_science_2sigmaupp_byEpoch = np.array([2*np.std(np.array(stars_science_flux.loc[i])) for i in range(len(stars_science_flux))])
                
    Results_star['date'] = dates_aux
    Results_star['stars_science_1sigmalow_byEpoch'] = stars_science_1sigmalow_byEpoch
    Results_star['stars_science_1sigmaupp_byEpoch'] = stars_science_1sigmaupp_byEpoch
    Results_star['stars_science_2sigmalow_byEpoch'] = stars_science_2sigmalow_byEpoch
    Results_star['stars_science_2sigmaupp_byEpoch'] = stars_science_2sigmaupp_byEpoch

    stars_science_mean_byEpoch = np.array([np.mean(np.array(stars_science_flux.loc[i])) for i in range(len(stars_science_flux))])

    # To plot the variability of the fluxes of stars in the difference image :
    
    # difference flux
    if do_diff_lc:
        stars_diff_flux_columns = ['star_{}_ImagDiff_fnJy'.format(i+1) for i in ids_stars_within_gal_mag]
        stars_diff_flux = stars_calc_byme[stars_diff_flux_columns]
        stars_diff_sigma_byEpoch = np.array(stars_diff_flux.std(axis=1)) #np.array([np.nanstd(np.array(stars_diff_flux.loc[i])) for i in range(len(stars_science_flux))])
        stars_diff_mean_byEpoch = np.mean(stars_diff_sigma_byEpoch) #np.array([np.nanmean(np.array(stars_diff_flux.loc[i])) for i in range(len(stars_science_flux))])

        stars_diff_1sigmalow_byEpoch = -stars_diff_sigma_byEpoch 
        stars_diff_1sigmaupp_byEpoch = stars_diff_sigma_byEpoch 
        stars_diff_2sigmalow_byEpoch = -2*stars_diff_sigma_byEpoch 
        stars_diff_2sigmaupp_byEpoch = 2*stars_diff_sigma_byEpoch 

        Results_star['stars_diff_1sigmalow_byEpoch'] = stars_diff_1sigmalow_byEpoch
        Results_star['stars_diff_1sigmaupp_byEpoch'] = stars_diff_1sigmaupp_byEpoch
        Results_star['stars_diff_2sigmalow_byEpoch'] = stars_diff_2sigmalow_byEpoch
        Results_star['stars_diff_2sigmaupp_byEpoch'] = stars_diff_2sigmaupp_byEpoch  

    #print(Results_star)
    ############################################################################################
    ####################### STARS PLOTS ########################################################
    ############################################################################################
    
    # Here we plot the fluxes of stars measured in the convolved science image 
    ##########################################################################
    
    plt.figure(figsize=(10,6))
    plt.title('stars LCs in convolved science image from {} and {} with Aperture radii of {}" '.format(field, ccd_name[ccd_num], star_aperture*arcsec_to_pixel))
    #print('ploting stars')
    #print(fluxconv_star_median[id_good_stars])
    
    #print(min(fluxconv_star_median[id_good_stars]), max(fluxconv_star_median[id_good_stars]))
    
    min_flux = np.nanmin(fluxconv_star_median[id_good_stars])
    max_flux = np.nanmax(fluxconv_star_median[id_good_stars])
    
    if min_flux<0:
        min_flux=1
    
    norm = matplotlib.colors.LogNorm(vmin=min_flux,vmax=max_flux)
    c_m = matplotlib.cm.plasma

    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    for i in id_good_stars:
        fs_star = (np.array(stars_calc_byme['star_{}_convDown_fnJy'.format(i+1)])).flatten()
        fs_star_err = np.ndarray.flatten(np.array(stars_calc_byme['star_{}_convDown_fnJy_err'.format(i+1)]))
                    
        stars_yarray = np.array(fs_star - np.median(fs_star))
        try:
            
            plt.errorbar(dates_aux, stars_yarray, yerr= fs_star_err, capsize=4, fmt='s', ls='solid', color = s_m.to_rgba(fluxconv_star_median[i]))
        except ValueError:
            continue
        marker_labels = np.ones(len(dates_aux))*(int(i+1))

        for w, label in enumerate(marker_labels):
            plt.annotate(str(int(label)), (dates_aux[w], stars_yarray[w]), color='k')

    plt.xlabel('MJD', fontsize=15)
    plt.ylabel('offset Flux [nJy] from median', fontsize=15)
    plt.colorbar(s_m, label = 'Median Flux [nJy]')
    plt.legend()
    
    if save:
        plt.savefig(subsubfolder_source / f'{title}_stars_flux_convDown_LCs.jpeg', bbox_inches='tight')
    
    plt.show()

    ######## Here we plot the light curves of stars from the difference images  #############################################
    
    if do_diff_lc:
        norm = matplotlib.colors.Normalize(vmin=min(fluxdiff_star_mean[id_good_stars]),vmax=max(fluxdiff_star_mean[id_good_stars]))
        c_m = matplotlib.cm.plasma

        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(min(fluxdiff_star_mean[id_good_stars]),max(fluxdiff_star_mean[id_good_stars]),nstars)

        plt.figure(figsize=(10,6))
        plt.title('stars LCs in difference image from {} and {} with Aperture radii of {}" '.format(field, ccd_name[ccd_num], round(star_aperture*arcsec_to_pixel,2)))

        for i in id_good_stars:
            f_star = np.array(stars_calc_byme['star_{}_ImagDiff_fnJy'.format(i+1)]) #* scaling
            f_star_err = np.array(stars_calc_byme['star_{}_ImagDiff_fnJy_err'.format(i+1)]) #* scaling

            plt.errorbar(dates_aux, f_star, yerr= f_star_err, capsize=4, fmt='s', ls='solid', color = s_m.to_rgba(np.mean(f_star)))

            marker_labels = np.ones(len(dates_aux))*(int(i+1))


            for w, label in enumerate(marker_labels):
                plt.annotate(str(int(label)), (dates_aux[w], f_star[w]), color=neon_green)

        plt.ylabel('Difference Flux [nJy]', fontsize=15)    
        plt.xlabel('MJD', fontsize=15)
        plt.colorbar(s_m, label = 'Median Flux [nJy]')
        plt.legend()

        if save:
            plt.savefig(subsubfolder_source / f'{title}_stars_flux_imagDiff_LCs.jpeg', bbox_inches='tight')

        plt.show()

    # Here we plot the mags light curves of stars 
    
    plt.figure(figsize=(10,6))
    plt.title('stars in convolved science image from {} and {} with Ap. radii of {}", within mag {} $\pm$ 0.5'.format(field, ccd_name[ccd_num], round(star_aperture*arcsec_to_pixel,2), round(mean_conv_mag, 2)))
    
    norm = matplotlib.colors.Normalize(vmin=min(magsconv_star_mean[id_good_stars]),vmax=max(magsconv_star_mean[id_good_stars]))
    c_m = matplotlib.cm.plasma

    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    for i in ids_stars_within_gal_mag:
        fs_star = (np.array(stars_calc_byme['star_{}_convDown_mag'.format(i+1)])).flatten()
        fs_star_err = np.ndarray.flatten(np.array(stars_calc_byme['star_{}_convDown_magErr'.format(i+1)]))
        
        #if np.mean(fs_star) <= 17.5:
        #    pass
        #else:
        #    continue
        
        stars_yarray = np.array(fs_star - np.median(fs_star))
        plt.errorbar(dates_aux, stars_yarray, yerr= fs_star_err, capsize=4, fmt='s', ls='solid', color = s_m.to_rgba(np.median(fs_star)))
        marker_labels = np.ones(len(dates_aux))*(int(i+1))

        for w, label in enumerate(marker_labels):
            plt.annotate(str(int(label)), (dates_aux[w], stars_yarray[w]), color='k')

    plt.xlabel('MJD', fontsize=15)
    plt.ylabel('$mag$ [AB] - mean(mag)', fontsize=15)
    plt.colorbar(s_m, label = 'Median mag [AB]')
    plt.legend()
    
    if save:
        plt.savefig(subsubfolder_source / f'{title}_stars_mags_convDown_LCs.jpeg', bbox_inches='tight')
    
    # Here we plot the std deviation of stars"
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Average dispersion of magnitude as a function of magnitude', fontsize=15)
    
    evaluate_mags = np.array([14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5])
    stars_at_given_mag = [id_good_stars[np.where((mags_good_stars>=evaluate_mags[k]) & (mags_good_stars<evaluate_mags[k+1]))] for k in range(len(evaluate_mags)-1)]
    
    # mags_id_good_stars
    std_at_given_mag = []
    eval_mag = []
    n_stars_mag = []
    for i in range(len(evaluate_mags)-1):
        eval_mag.append((evaluate_mags[i]+evaluate_mags[i+1])*0.5)
        stars_science_given_mag_columns = ['star_{}_convDown_mag'.format(j+1) for j in stars_at_given_mag[i]]
        n_stars_mag.append(len(stars_at_given_mag[i]))
        #print('stars between {} and {}'.format(evaluate_mags[i],evaluate_mags[i+1]))
        
        stars_science_at_given_mag = stars_calc_byme[stars_science_given_mag_columns]
        #print(stars_science_at_given_mag)
        #print('number of stars: ',n_stars_mag[-1])
        stars_science_at_given_mag -= stars_science_at_given_mag.mean()
        stars_sciencemag_1sigma_disp_byEpoch = np.array(stars_science_at_given_mag.std())
        #np.array([np.nanstd(np.array(stars_science_at_given_mag.loc[i])) for i in range(len(stars_science_at_given_mag))])
        
        std_at_given_mag.append(np.mean(stars_sciencemag_1sigma_disp_byEpoch))
        ax.text(eval_mag[i]+0.2, 0, str(len(stars_at_given_mag[i])))
    
    ax.errorbar(eval_mag, np.zeros(len(eval_mag)), yerr=std_at_given_mag, xerr=1.0, color='m', capsize=3, fmt='o', ls=' ')
    
    ax.set_xlabel('$m_g$', fontsize=15)
    ax.set_ylabel('$Average \Delta m_g$', fontsize=15)
    
    #field = collection_diff[13:24]
    #if save_lc_stars and well_subtracted:
    #    print('saving lcs stars as: ' +'light_curves/{}/{}_{}_random_stars.jpeg'.format(field, field, ccd_name[ccd_num]))
    #    plt.savefig('light_curves/{}/{}_{}_random_stars_sumwTemp.jpeg'.format(field, field, ccd_name[ccd_num]), bbox_inches='tight')
    #if save_lc_stars and not well_subtracted:
    #    print('saving lcs stars as: ' +'light_curves/{}/{}_{}_random_stars.jpeg'.format(field, field, ccd_name[ccd_num]))
    #    plt.savefig('light_curves/{}/{}_{}_random_stars_wdipoles_sumwTemp.jpeg'.format(field, field, ccd_name[ccd_num]), bbox_inches='tight')
    #plt.show()

    #if do_zogy:
    #    zogy = zogy_lc(repo, collection_calexp, collection_coadd, ra, dec, ccd_num, visits, rd_aux, instrument = 'DECam', plot_diffexp=plot_zogy_stamps, plot_coadd = plot_coadd, cutout=cutout)
    #    print(zogy)
    #    z_flux = zogy.flux
    #    z_ferr = zogy.flux_err
    #    plt.errorbar(zogy.dates, z_flux, yerr=z_ferr, capsize=4, fmt='s', label ='ZOGY Cáceres-Burgos', color='orange', ls ='dotted')
    
    # Plot LC of the galaxy
    
    plt.show()
    
    ########################################################################################
    #######################  Instrumental flux LCs   ##########################################
    ########################################################################################

    #fig = plt.figure(figsize=(10,6))
    #ax = fig.add_subplot(111)
    #
    #ax.set_title('Convolution downgrade Instrumental flux Light curves', fontsize=17)    
    #norm = mpl.colors.Normalize(vmin=0.5,vmax=1.5)
    #c_m = mpl.cm.magma
#
    ## create a ScalarMappable and initialize a data structure
    #s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
    #s_m.set_array([])
    ## 
    ##     name_columns_inst_convdown = ['instflux_ConvDown_{}wseeing'.format(f) for f in ap_radii]
    ## name_columns_inst_convdownErr = ['instfluxErr_ConvDown_{}wseeing'.format(f) for f in ap_radii]
#
#
    ## 
    #for fa in ap_radii:
    #    
    #    column_to_take = 'instflux_ConvDown_{}wseeing'.format(fa)
    #    column_err_to_take = 'instfluxErr_ConvDown_{}wseeing'.format(fa)
    #    
    #    column2_to_take = 'instflux_{}seeing'.format(fa)
    #    column2_err_to_take = 'instfluxErr_{}seeing'.format(fa)
    #    
    #    conv_flux = np.array(source_of_interest[column_to_take])
    #    conv_flux_err = np.array(source_of_interest[column_err_to_take])
    #    
    #    cal_flux = np.array(source_of_interest[column2_to_take])
    #    cal_flux_err = np.array(source_of_interest[column2_err_to_take])
    #    
    #    plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), cal_flux, yerr = cal_flux_err , capsize=4, fmt='s', color=s_m.to_rgba(fa), ls ='dashdot')
    #    
    #    plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), conv_flux, yerr = conv_flux_err , capsize=4, fmt='s', color=s_m.to_rgba(fa), ls ='-')
    #    
    #    #if SIBLING!=None:
    #    #    x, y, yerr = compare_to(SIBLING, sfx='mag', factor=fa)
    #    #    flx_adu = pc.FromMagToCounts(y, t=np.array(ExpTimes))
    #    #    ax.plot(x-min(x), flx_adu, '^', color=s_m.to_rgba(fa), ls ='dotted')
    #        
    #        #ax.errorbar(x-min(x), y, yerr=yerr,  capsize=4, fmt='^', ecolor=s_m.to_rgba(fa), color=s_m.to_rgba(fa), ls ='dotted', markerfacecolor=None)
    #
    ## This plots are for the labels
    #if SIBLING!=None:
    #    ax.errorbar([-4,-4],[0,0],capsize=4, fmt='^', color='k', markeredgewidth=3, markerfacecolor='None', ls ='dotted', label='Martinez-Palomera et al. 2020')
    #ax.errorbar([-4,-4],[0,0],capsize=4, fmt='o', color='k', markeredgewidth=3, ls ='-', label='Convolved R=(n)*{}" '.format(round(worst_seeing * sigma2fwhm * arcsec_to_pixel,3)))
    #ax.errorbar([-4,-4],[0,0],capsize=4, fmt='o', color='k', markeredgewidth=3, ls ='dashdot', label='Science R=(n)*{}"'.format(round(worst_seeing * sigma2fwhm * arcsec_to_pixel,3)))
    #
    #
#
    #ax.set_ylabel('Flux [counts]', fontsize=15 )
    #ax.set_xlabel('MJD - {}'.format(round(min(dates_aux),2)), fontsize=15)
    #ax.set_xlim(-0.05, max(source_of_interest.dates) - min(source_of_interest.dates) + 0.05)
#
    #
    #ax.legend(frameon=False, loc='upper left', ncol=2)
    #cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    #cbar = plt.colorbar(s_m, cax=cbar_ax,  boundaries =np.linspace(0.5,1.5,len(ap_radii)+1),
    #                    ticks = np.linspace(0.5,1.5,len(ap_radii)+1)[:-1] + np.diff(np.linspace(0.5,1.5,len(ap_radii)+1))/2)
    #cbar.set_ticklabels(ap_radii.astype('str'))
#
#
    #if save:
    #    plt.savefig(subsubfolder_source / 'convolution_downgrade_instflx_LCs.jpeg', bbox_inches='tight')
    #
    #plt.show()
    
    
    ########################################################################################
    #######################  Image Difference LCs ##########################################
    ########################################################################################
    if do_diff_lc:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        ax.set_title('Image Differencing Light curves', fontsize=17)    
        norm = mpl.colors.Normalize(vmin=0.5,vmax=1.5)
        c_m = mpl.cm.magma

        # create a ScalarMappable and initialize a data structure
        s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        for fa in ap_radii:

            column_to_take = 'flux_ImagDiff_nJy_{}sigmaPsf'.format(fa)
            column_err_to_take = 'fluxErr_ImagDiff_nJy_{}sigmaPsf'.format(fa)
            diff_flux = np.array(source_of_interest[column_to_take])
            diff_flux_err = np.array(source_of_interest[column_err_to_take])
            plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), diff_flux, yerr = diff_flux_err , capsize=4, fmt='s', color=s_m.to_rgba(fa), ls ='-')

            if SIBLING!=None and os.path.exists(SIBLING):
                x, y, yerr = compare_to(SIBLING, sfx='mag', factor=fa)
                f, ferr = pc.ABMagToFlux(y, yerr) # in Jy
                ax.errorbar(x-min(x), f/1e-9 - np.mean(f/1e-9), yerr=ferr/1e-9,  capsize=4, fmt='^', ecolor=s_m.to_rgba(fa), color=s_m.to_rgba(fa), ls ='dotted', markerfacecolor=None)

        # This plots are for the labels
        if SIBLING!=None and os.path.exists(SIBLING):
            ax.errorbar([-4,-4],[0,0],capsize=4, fmt='^', color='k', markeredgewidth=3, markerfacecolor='None', ls ='dotted', label='Martinez-Palomera et al. 2020')
        ax.errorbar([-4,-4],[0,0],capsize=4, fmt='o', color='k', markeredgewidth=3, ls ='-', label='Caceres-Burgos in prep')


        if do_lc_stars:
            ax.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_diff_1sigmalow_byEpoch, stars_diff_1sigmaupp_byEpoch, alpha=0.1, color='m', label = '1-2 $\sigma$ dev of {} $\pm$ 0.5 mag'.format(round(mean_conv_mag,2))) #
            ax.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_diff_2sigmalow_byEpoch, stars_diff_2sigmaupp_byEpoch, alpha=0.1, color='m')     

        ax.set_ylabel('Flux [nJy]', fontsize=15 )
        ax.set_xlabel('MJD - {}'.format(round(min(dates_aux),2)), fontsize=15)
        ax.set_xlim(-0.005, max(source_of_interest.dates) - min(source_of_interest.dates) + 0.005)


        ax.legend(frameon=False, loc='upper left', ncol=2)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        cbar = plt.colorbar(s_m, cax=cbar_ax,  boundaries =np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1),
                            ticks = np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1)[:-1] + np.diff(np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1))/2)
        cbar.set_ticklabels(ap_radii.astype('str'))
        cbar.set_label('radii in arcsec', fontsize=15)

        if save:
            plt.savefig(subsubfolder_source / f'{title}_Image_difference_LCs.jpeg', bbox_inches='tight')

        plt.show()

    
    ########################################################################################
    #######################  Conv downgrade LCs   ##########################################
    ########################################################################################

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    ax.set_title('Convolution downgrade flux Light curves', fontsize=17)    
    norm = mpl.colors.Normalize(vmin=0.5,vmax=1.5)
    c_m = mpl.cm.magma

    # create a ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    # 
    # name_columns_convdown = ['flux_ConvDown_nJy_{}wseeing'.format(f) for f in ap_radii]
    # name_columns_convdownErr = ['fluxErr_ConvDown_nJy_{}wseeing'.format(f) for f in ap_radii]

    # 
    min_flux = 0
    max_flux = 0
    
    for fa in ap_radii:
        
        column_to_take = 'flux_ConvDown_nJy_{}_arcsec'.format(fa)
        column_err_to_take = 'fluxErr_ConvDown_nJy_{}_arcsec'.format(fa)
        conv_flux = np.array(source_of_interest[column_to_take])
        #print('conv flux for fa {}: '.format(fa))
        #print(conv_flux)
        conv_flux_err = np.array(source_of_interest[column_err_to_take])
        plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), conv_flux , yerr = conv_flux_err , capsize=4, fmt='s', color=s_m.to_rgba(fa), ls ='-')
        
        if SIBLING!=None and os.path.exists(SIBLING):
            x, y, yerr = compare_to(SIBLING, sfx='mag', factor=fa)
            f, ferr = pc.ABMagToFlux(y, yerr) # in Jy
            ax.errorbar(x-min(x), f/1e-9 , yerr=ferr/1e-9,  capsize=4, fmt='^', ecolor=s_m.to_rgba(fa), color=s_m.to_rgba(fa), ls ='dotted', markerfacecolor=None)
            if fa==min(ap_radii):
                min_flux = np.min(f/1e-9)
            if fa ==max(ap_radii):
                max_flux = np.max(f/1e-9)
    
    # This plots are for the labels
    if SIBLING!=None and os.path.exists(SIBLING):
        ax.errorbar([-4,-4],[0,0],capsize=4, fmt='^', color='k', markeredgewidth=3, markerfacecolor='None', ls ='dotted', label='Martinez-Palomera et al. 2020')
    ax.errorbar([-4,-4],[0,0],capsize=4, fmt='o', color='k', markeredgewidth=3, ls ='-', label='Caceres-Burgos in prep')
    
    
    #if do_lc_stars:
    #    ax.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_science_1sigmalow_byEpoch, stars_science_1sigmaupp_byEpoch, alpha=0.1, color='m', label = 'stars 1-2 $\sigma$ dev') #
    #    ax.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_science_2sigmalow_byEpoch, stars_science_2sigmalow_byEpoch, alpha=0.1, color='m')     
    
    ax.set_ylabel('Flux [nJy]', fontsize=15 )
    ax.set_xlabel('MJD - {}'.format(round(min(dates_aux),2)), fontsize=15)
    ax.set_xlim(-0.005, max(source_of_interest.dates) - min(source_of_interest.dates) + 0.005)
    #if SIBLING!=None:
    #    ax.set_ylim(min_flux*0.9, max_flux*1.1)
    
    ax.legend(frameon=False, loc='upper left', ncol=2)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = plt.colorbar(s_m, cax=cbar_ax,  boundaries =np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1),
                        ticks = np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1)[:-1] + np.diff(np.linspace(0.5,1.5,len(ap_radii)+1))/2)
    cbar.set_ticklabels(ap_radii.astype('str'))
    cbar.set_label('radii in arcsec', fontsize=15)

    if save:
        plt.savefig(subsubfolder_source / f'{title}_convolution_downgrade_flx_LCs.jpeg', bbox_inches='tight')
    
    plt.show()
    
    
    ########################################################################################
    ####################### Conv downgrade LCs ############################################
    ########################################################################################
    
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Convolution downgrade (mags) Light curves', fontsize=17)  
    
    norm = mpl.colors.Normalize(vmin=0.5,vmax=1.5)
    c_m = mpl.cm.magma

    # create a ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for fa in ap_radii:
        if do_convolution:
            
            column_to_take = 'mag_ConvDown_nJy_{}_arcsec'.format(fa)
            column_err_to_take = 'magErr_ConvDown_nJy_{}_arcsec'.format(fa)
            
            conv_mag = np.array(source_of_interest[column_to_take])
            conv_magerr = np.array(source_of_interest[column_err_to_take])
            
            mean_mag = np.nanmean(conv_mag, dtype='float64')
            
            ax.errorbar(source_of_interest.dates - min(source_of_interest.dates), conv_mag - mean_mag, yerr=conv_magerr, capsize=4, fmt='o', color=s_m.to_rgba(fa), ls='-')
            
            if SIBLING!=None and os.path.exists(SIBLING):
                x, y, yerr = compare_to(SIBLING, sfx='mag', factor=fa)
                ax.errorbar(x-min(x), y -  np.mean(y), yerr=yerr,  capsize=4, fmt='^', color=s_m.to_rgba(fa), markeredgewidth=3, markerfacecolor='None', ls ='dotted')
    
    # This plots are for the labels
    if SIBLING!=None and os.path.exists(SIBLING):
        ax.errorbar([-4,-4],[0,0],capsize=4, fmt='^', color='k', markeredgewidth=3, markerfacecolor='None', ls ='dotted', label='Martinez-Palomera et al. 2020')
    ax.errorbar([-4,-4],[0,0],capsize=4, fmt='o', color='k', markeredgewidth=3, ls ='-', label='Caceres-Burgos in prep')
    
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = plt.colorbar(s_m, cax=cbar_ax,  boundaries =np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1),
                        ticks = np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1)[:-1] + np.diff(np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1))/2)
    cbar.set_ticklabels(ap_radii.astype('str'))
    cbar.set_label('radii in arcsec', fontsize=15)
    
    if do_lc_stars:
        ax.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_sciencemag_1sigmalow_byEpoch, stars_sciencemag_1sigmaupp_byEpoch, alpha=0.1, color='b', label = '1-2 $\sigma$ dev of {} $\pm$ 0.5 mag'.format(round(mean_conv_mag,2))) #
        ax.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_sciencemag_2sigmalow_byEpoch, stars_sciencemag_2sigmaupp_byEpoch, alpha=0.1, color='b')  
        
        #print(source_of_interest.dates - min(source_of_interest.dates))
        #print(stars_sciencemag_1sigmalow_byEpoch)
        #print(stars_sciencemag_1sigmaupp_byEpoch)
        source_of_interest['stars_dev_within_0p5_galmag'] = stars_sciencemag_1sigmaupp_byEpoch
    
    ax.axhline(0, ls='--', color='gray')
    
    ax.set_xlabel('MJD - {}'.format(round(min(dates_aux),2)), fontsize=15)
    ax.set_ylabel('$m_g$ - $\hat{m_g}$', fontsize=15)
    ax.set_xlim(-0.005, max(source_of_interest.dates) - min(source_of_interest.dates) + 0.005)
    #ax.set_ylim()
    ax.legend(frameon=False, ncol=2, loc='upper left')
    if save:
        plt.savefig(subsubfolder_source / f'{title}_Convolution_downgrade_LCs.jpeg', bbox_inches='tight')
    
    
    ##### Calculate Excess Variance ###########
    
    
    sigma_rms_sq, errsigma_rms_sq, sigma_rms_subtracted = Excess_variance(np.array(source_of_interest['mag_ConvDown_nJy_1.0_arcsec']), np.array(source_of_interest['magErr_ConvDown_nJy_1.0_arcsec']))
    
    print('sigma_rms_sq, errsigma_rms_sq, sigma_rms_subtracted: ', sigma_rms_sq, errsigma_rms_sq, sigma_rms_subtracted)
    
    if mode=='HiTS' or mode=='HITS' and SIBLING is not None and jname is not None and os.path.exists(SIBLING):
        
        x, m, merr = compare_to(SIBLING, sfx='mag', factor=0.75)
        
        
        sibling_dataset.loc[jname, 'Excess_variance_pcb_wkernel_{}'.format(type_kernel)] = sigma_rms_sq
        sibling_dataset.loc[jname, 'Excess_variance_e_pcb_wkernel_{}'.format(type_kernel)] = errsigma_rms_sq
        sibling_dataset.loc[jname, 'Excess_variance_cor_pcb_wkernel_{}'.format(type_kernel)] = sigma_rms_subtracted
        #sibling_dataset.to_csv('SIBLING_sources_usingMPfilter_andPCB_comparison.csv')
    
        sigma_rms_sq_jge, errsigma_rms_sq_jge, sigma_rms_subtracted_jge = Excess_variance(y, yerr)
        
        sibling_dataset.loc[jname, 'Excess_variance_jge'] = sigma_rms_sq_jge
        sibling_dataset.loc[jname, 'Excess_variance_e_jge'] = errsigma_rms_sq_jge
        sibling_dataset.loc[jname, 'Excess_variance_cor_jge'] = sigma_rms_subtracted_jge
        #sibling_dataset.to_csv('SIBLING_sources_usingMPfilter_andPCB_comparison.csv')
    
    ########################################################################################
    ###############################  Profiles ##############################################
    ########################################################################################
    #
    #fig = plt.figure(figsize=(10,6))
    #ax = fig.add_subplot(111)

    #norm = mpl.colors.Normalize(vmin=min(Seeing),vmax=max(Seeing))
    #c_m = mpl.cm.magma

    ## create a ScalarMappable and initialize a data structure
    #s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
    #s_m.set_array([])
    #
    #for i, v in enumerate(visits_aux):
    #    ax.plot(np.linspace(0.05, 6 , 15), profiles['{}'.format(v)]/profiles['{}'.format(worst_seeing_visit)], label=str(round(min(dates_aux),2)) + ' + {}'.format(dates_aux[i] - min(dates_aux)), color=s_m.to_rgba(Seeing[i]))
    #    
    #cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    #cbar = plt.colorbar(s_m, cax=cbar_ax)
    #cbar.set_label('FWHM in arcsec')
    #
    ##plt.colorbar(s_m)
    #ax.set_yscale('log')
    #ax.set_xlabel('arcseconds', fontsize=17)
    #ax.set_ylabel('Growth curve / Growth curve worst visit', fontsize=17)
    #ax.legend(frameon=False)
   
    #if save:
    #    plt.savefig(subsubfolder_source / f'{title}_curve_growth_of_galaxy.jpeg', bbox_inches='tight')

    #plt.show()
    
    plot_star_profiles(profiles_stars, round_magnitudes, visits_aux,  np.linspace(0.05, 6, 15), worst_seeing_visit, Seeing, save_as = subsubfolder_source / 'curve_growth_stars.jpeg' )

    stamps(Data_science, Data_convol, Data_diff, Data_coadd, coords_science, coords_convol, coords_coadd, source_of_interest, Results_star, visits, KERNEL, Seeing, SIBLING = SIBLING, cut_aux=cutout, r_diff = seeing * sigma2fwhm * np.array([0.75]), r_science=worst_seeing * sigma2fwhm * np.array([0.75]),  field='', name='', first_mjd = 58810, folder=subsubfolder_source)        
    #stamps_and_LC_plot_forPaper(Data_science, Data_convol, Data_diff, Data_coadd, coords_science, coords_convol, coords_coadd, source_of_interest, Results_star, visits, KERNEL, Seeing, r_diff=r_diff, r_science=r_science, SIBLING = SIBLING, cut_aux=cutout, first_mjd=int(min(dates_aux)), name_to_save=name_to_save)
    if save:
        source_of_interest.to_csv(subsubfolder_source / f'{title}_galaxy_LCs_dps.csv')
        
    return source_of_interest


def get_photometry(data, mask=None, bkg = None, gain=4., pos=(20, 20),
                   radii=10., sigma1=None, alpha=None, beta=None, iter=0, centered=True):

    if centered:
        back_mean, back_median, back_std = sigma_clipped_stats(data, mask,
                                                               sigma=3,
                                                               maxiters=3,
                                                               cenfunc=np.median)
        print('\tBackground stats: %f, %f' % (back_median, back_std))
        tbl = find_peaks(data,
                         np.minimum(back_std, bkg.background_rms_median) * 3,
                         box_size=5) # subpixel=True
        if len(tbl) == 0:
            print('\tNo detection...')
            return None
        #print(tbl)
        tree_XY = cKDTree(np.array([tbl['x_peak'], tbl['y_peak']]).T)
        if iter == 0:
            d = 9
        else:
            d = 5
        dist, indx = tree_XY.query(pos, k=2, distance_upper_bound=d)
        print(tbl)
        print(dist, indx)

        if np.isinf(dist).all():
            print('\tNo source found in the asked position... ')
            print('using given position...')
            position = pos
            # return None
        else:
            if len(tbl) >= 2 and not np.isinf(dist[1]):
                if tbl[indx[1]]['peak_value'] > \
                    tbl[indx[0]]['peak_value']:
                    indx = indx[1]
                else:
                    indx = indx[0]
            else:
                indx = indx[0]
                
            position = [tbl[indx]['x_peak'], tbl[indx]['y_peak']]
    else:
        position = pos

    print('\tObject position: ', position)

    apertures = [CircularAperture(position, r=r) for r in radii]
    try:
        phot_table = aperture_photometry(data, apertures, mask=mask,
                                         method='subpixel', subpixels=5)
    except IndexError:
        phot_table = aperture_photometry(data, apertures,
                                         method='subpixel', subpixels=5)
    for k, r in enumerate(radii):
        area = np.pi * r ** 2
        phot_table['aperture_flx_err_%i' %
                   k] = np.sqrt(sigma1**2 * alpha**2 * area**beta +
                                phot_table['aperture_sum_%i' % k][0] / gain)
    phot_table.remove_columns(['xcenter', 'ycenter'])
    phot_table['xcenter'] = position[0]
    phot_table['ycenter'] = position[1]
    return phot_table




def plot_star_profiles(profiles, round_magnitudes, visits, apertures, worst_visit, seeing, save_as=None):
    
    norm = mpl.colors.Normalize(vmin=min(seeing),vmax=max(seeing))
    c_m = mpl.cm.magma

    # create a ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    n_rows = 2
    n_cols = len(round_magnitudes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    row1_axes = [axes[0, i] for i in range(n_cols)]
    row2_axes = [axes[1, i] for i in range(n_cols)]

    for j in range(n_rows):
        for i in range(n_cols):            
            mag = round_magnitudes[i]
            
            ax = axes[j, i]
            
            if j ==0 and i==0:
                ax.set_ylabel('Flux / maximum(Flux)')
            if j == 1 and i==0:
                ax.set_ylabel('prof - worst profile')
            if j==1:
                ax.set_xlabel('Apertures in arcsec')
            if j==0:
                ax.set_title('Star of psf mag {}'.format(mag))    
                
            for k,v in enumerate(visits):
                
                try:
                    worst_prof = profiles['mag{}_{}'.format(mag, worst_visit)]
                    prof = profiles['mag{}_{}'.format(mag, v)]
                    if j==0:
                        ax.plot(apertures, prof, color = s_m.to_rgba(seeing[k]))
                        ax.sharey(row1_axes[0])
                    if j==1:
                        ax.plot(apertures, prof - worst_prof, color = s_m.to_rgba(seeing[k]))
                        ax.sharey(row2_axes[0])
                except KeyError:
                    print('failed to check star of mag {}'.format(mag))
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = plt.colorbar(s_m, cax=cbar_ax)
    cbar.set_label('FWHM in arcsec')
    
    plt.subplots_adjust(hspace=0, wspace=0)
    
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')
    
    plt.show()
    return 


def all_ccds(repo, field, collection_calexp, collection_diff, collection_coadd, sfx='flx', show_star_stamps = False, show_stamps=False, factor=1, well_subtracted=False, r_star=2):
    """
    plots all sources located in a field

    """
    Dict = {}
    #repo = "/home/pcaceres/data_hits"
    #field = 'Blind15A_04'
    cands = Find_sources(sibling_allcand, field)
    index = cands.index

    ccds = [f.split('_')[2] for f in cands.internalID]
    ra_agn = cands.raMedian
    dec_agn = cands.decMedian

    #collection_diff = "imagDiff_AGN/{}".format(field)
    #collection_calexp = "processCcdOutputs/{}".format(field)
    #collection_coadd = 'Coadd_AGN/{}'.format(field)
    #len(ccds)

    ccds_used = []
    for i in range(len(ccds)):
        try:
            print('Trying ccd: ', ccds[i])
            data = get_all_exposures(repo, 'science')
            visits = list(data[(data['target_name']=='{}'.format(field)) & (data['day_obs']<20150219)].exposure)
            ra, dec = ra_agn[index[i]], dec_agn[index[i]]
            ccdnum = detector_nomenclature[ccds[i]]
            title = '{}'.format(cands.SDSS12[index[i]])
            folder = '{}/'.format(field)
            name_file = (field + '_' + ccds[i] + '_ra_' + str(ra_agn[index[i]]) + '_dec_' + str(dec_agn[index[i]]) + '_calibLsst').replace('.', '_')
            df = get_light_curve(repo, visits, collection_diff, collection_calexp, ccdnum, ra, dec, r=10, factor=factor, save=True, save_as = name_file, SIBLING = '/home/pcaceres/Jorge_LCs/'+cands.internalID.loc[index[i]] +'_g_psf_ff.csv', title=title, show_stamps=show_stamps, do_zogy=False, collection_coadd=collection_coadd, plot_coadd=False, save_stamps=True, do_lc_stars=True, save_lc_stars=True, show_star_stamps=show_star_stamps, sfx=sfx, nstars=10, well_subtracted=well_subtracted, field=field, r_star = r_star)
            
            if len(df) != 0 or type(df)!=None:

                Dict['{}_{}'.format(field, ccds[i])] = df
                ccds_used.append(ccds[i])

        except:
            pass

    if len(Dict) == 0:
        print('couldnt obtain the dataframe for the ccds: ', ccds)
        return ccds_used
    
    ccds_used = np.unique(ccds_used)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=32)
    c_m = matplotlib.cm.plasma

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    T = np.linspace(0,27,len(ccds_used))


    plt.figure(figsize=(10,6))
    i=0
    for key in Dict:
        source_of_interest = Dict[key]
        if type(source_of_interest) == type(None):
            continue
        plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy, yerr=source_of_interest.fluxerr_nJy, capsize=4, fmt='s', label ='AL Cáceres-Burgos {}'.format(key), ls ='dotted',  color = s_m.to_rgba(T[i]))
        plt.xlabel('MJD', fontsize=15)
        plt.ylabel('Flux [nJy]', fontsize=15)
        plt.title('Difference flux', fontsize=15)
        i+=1
    plt.legend()
    plt.savefig('{}/LSST_notebooks/light_curves/{}/{}_all_ccds_diference.png'.format(main_root, field,field))
    plt.show()

    plt.figure(figsize=(10,6))

    i=0
    for key in Dict:
        source_of_interest = Dict[key]
        if type(source_of_interest) == type(None):
            continue
        plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy_coadd, yerr=source_of_interest.fluxerr_nJy_coadd, capsize=4, fmt='s', label ='AL Cáceres-Burgos {}'.format(key), ls ='dotted',  color = s_m.to_rgba(T[i]))
        plt.xlabel('MJD', fontsize=15)
        plt.ylabel('Flux [nJy]', fontsize=15)
        plt.title('Coadd/Template flux', fontsize=15)
        i+=1
    plt.legend()
    plt.savefig('{}/LSST_notebooks/light_curves/{}/{}_all_ccds_template.png'.format(main_root, field,field))
    plt.show()

    #i=0
    #for key in Dict:
    #    source_of_interest = Dict[key]
    #    if type(source_of_interest) == type(None):
    #        continue
    #    plt.errorbar(source_of_interest.fdates, source_of_interest.Mg - source_of_interest.Mg_coadd, yerr = np.sqrt(np.array(source_of_interest.Mg_err)**2 + np.array(source_of_interest.Mg_err_coadd)**2) , capsize=4, fmt='s', label ='AL Cáceres-Burgos in prep {}'.format(key), ls ='dotted',  color = s_m.to_rgba(T[i]))
    #    plt.xlabel('MJD', fontsize=15)
    #    plt.ylabel('offset Mag AB difference', fontsize=15)
    #    plt.title('Magnitude AB - Magnitude AB coadd', fontsize=15)
    #    i+=1
    ##plt.savefig('/home/pcaceres/testdata_hits/LSST_notebooks/light_curves/{}/{}_mag_all_ccds_difference.png'.format(field, field))
    #plt.legend()
    ##plt.savefig('/home/pcaceres/testdata_hits/LSST_notebooks/light_curves/{}/{}_all_ccds_ab_magnitude.png'.format(field, field))
    #plt.show()
    #plt.figure(figsize=(10,6))

    #i=0
    #for key in Dict:
    #    source_of_interest = Dict[key]
    #    if type(source_of_interest) == type(None):
    #        continue
    #    plt.errorbar(source_of_interest.dates, source_of_interest.Mg - np.median(source_of_interest.Mg), yerr = source_of_interest.Mg_err, capsize=4, fmt='s', label ='AL Cáceres-Burgos in prep {}'.format(key), ls ='dotted',  color = s_m.to_rgba(T[i]))
    #    plt.xlabel('MJD', fontsize=15)
    #    plt.ylabel('offset Mag AB', fontsize=15)
    #    plt.title('Magnitude AB - median AB mag', fontsize=15)
    #    i+=1
    #plt.savefig('/home/pcaceres/testdata_hits/LSST_notebooks/light_curves/{}/{}_mag_all_ccds_difference.png'.format(field, field))
    
    
    return ccds_used

def make_kernel(repo, visit, ccdnum, collection_calexp, ns=10, plot=False, cut=10):
    '''
    
    input
    -----
    repo [string] : directory were data is processed
    visit []
    ccdnum [int] : number of the detector
    collection_calexp [string] : directory were the reduced image is 
    ns [int] : number of star 
    plot [bool] : if True, the kernel is plotted 
    cut [int] : 

    output
    ------
    stars_sum_norm

    '''   
    field = collection_calexp.split('/')[-1]
    directory = '{}/LSST_notebooks/kernels/Kernel_{}_{}_{}'.format(main_root, visit, field, ccdnum)
    file = '{}.pickle'.format(directory)
    print(visit)

    try:
        with open(file, 'rb') as handle:
            stars_sum_norm = pickle.load(handle)
    except:
        butler = Butler(repo)
        stars_table = Select_table_from_one_calib_exposure(repo, visit, ccdnum, collection_calexp, stars=True, s_to_n_star=10).to_pandas()
        sn_stars = np.array(stars_table['base_PsfFlux_instFlux']/stars_table['base_PsfFlux_instFluxErr']) # instrumental signal to noise 
        stars_table['inst_s_to_n'] = sn_stars
        stars_table = stars_table.sort_values(by=['inst_s_to_n'], ascending=False)
        stars_table = stars_table.drop_duplicates(subset = ['coord_ra_ddegrees'], keep='first')

        ra_stars =  np.array(stars_table['coord_ra_ddegrees'])
        dec_stars = np.array(stars_table['coord_dec_ddegrees'])
        arcsec_to_pixel = 0.27 #arcsec/pixel
        calexp = butler.get('calexp', visit= visit, detector= ccdnum, instrument='DECam', collections=collection_calexp)
        wcs = calexp.getWcs()

        visits = [visit for i in range(len(stars_table))]
        number_stars = 0
        cutout = int(cut/arcsec_to_pixel) 
        thresh = 60
        print('number of available stars: ', len(stars_table))
        x_half_width = cutout
        y_half_width = cutout
        
        RAs = []
        DECs = []
        starn = 0
        collected_stars = {}
        centroid_stars = {}

        
        for ra, dec in zip(ra_stars, dec_stars):
            
            starn+=1
            obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
            
            afwDisplay.setDefaultMaskTransparency(100)
            afwDisplay.setDefaultBackend('matplotlib')
            
            x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
            
            calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
            
            objects = sep.extract(calexp_cutout.image.array, thresh, minarea=30)

            if len(objects)!=1:
                continue
            else:
                RAs.append(ra)
                DECs.append(dec)
                
                x_pix_cen = round(objects[0]['x'],0)
                y_pix_cen = round(objects[0]['y'],0)
                second_cutout = cutout-5
                calexp_cutout_to_use = calexp_cutout.image.array[int(y_pix_cen-second_cutout):int(y_pix_cen+second_cutout), int(x_pix_cen-second_cutout):int(x_pix_cen+second_cutout)]
                collected_stars['star_{}'.format(number_stars)] = calexp_cutout_to_use
                centroid_stars['star_{}'.format(number_stars)] = [x_pix_cen, y_pix_cen]

            number_stars+=1

            if number_stars==1:
                sum_stars = calexp_cutout_to_use
            else:
                sum_stars += calexp_cutout_to_use
            
            if plot:
                #second_cutout = cutout-5

                fig = plt.figure(figsize=(10, 5))
                plt.imshow(calexp_cutout_to_use, cmap='rocket', origin='lower', vmin=0, vmax=100)
                plt.colorbar()
                plt.scatter(second_cutout, second_cutout, color=neon_green, marker='x', linewidth=3, label='new center')
                
                circle = plt.Circle((second_cutout, second_cutout), radius = 1/arcsec_to_pixel, color=neon_green, fill = False, linewidth=4)
                plt.gca().add_patch(circle)
                plt.title('star number {}'.format(number_stars), fontsize=15)
                plt.tight_layout()
                plt.legend()
                plt.show()
            if not number_stars < ns:
                break

        stars_sum_norm = sum_stars/np.sum(sum_stars)

        

        with open(file, 'wb') as handle:        
            pickle.dump(stars_sum_norm, handle, protocol = pickle.HIGHEST_PROTOCOL)
    if plot:
            butler = Butler(repo)
            stars_table = Select_table_from_one_calib_exposure(repo, visit, ccdnum, collection_calexp, stars=True, s_to_n_star=10).to_pandas()
            sn_stars = np.array(stars_table['base_PsfFlux_instFlux']/stars_table['base_PsfFlux_instFluxErr'])
            stars_table['inst_s_to_n'] = sn_stars
            stars_table = stars_table.sort_values(by=['inst_s_to_n'], ascending=False)
            stars_table = stars_table.drop_duplicates(subset = ['coord_ra_ddegrees'], keep='first')
            #stars_table_eff = stars_table.head(ns)
            calexp = butler.get('calexp', visit=visit, detector=ccdnum, collections=collection_calexp, instrument='DECam')
            psf =  calexp.getPsf()
            
            ra_stars =  np.array(stars_table['coord_ra_ddegrees'])
            dec_stars = np.array(stars_table['coord_dec_ddegrees'])
            ra = ra_stars[0]
            dec = dec_stars[0]

            obj_pos_2d = lsst.geom.Point2D(ra, dec)
            imageKernel = psf.computeKernelImage(obj_pos_2d)
            fig = plt.figure(figsize=(10, 5))
            #print(imageKernel)
            plt.imshow(np.asarray(imageKernel.array,dtype='float'), cmap='rocket', origin='lower', vmin=np.min(imageKernel.array), vmax=np.max(imageKernel.array))
            plt.colorbar()

            fig = plt.figure(figsize=(10, 5))
            arcsec_to_pixel = 0.27 #arcsec/pixel
            cutout = int(cut/arcsec_to_pixel) 
            second_cutout = cutout-5
            plt.imshow(stars_sum_norm, cmap='rocket', origin='lower', vmin=np.min(stars_sum_norm), vmax=np.max(stars_sum_norm))
            plt.colorbar()
            plt.scatter(second_cutout, second_cutout, color=neon_green, marker='x', linewidth=3)
            circle = plt.Circle((second_cutout, second_cutout), radius = 1/arcsec_to_pixel, color=neon_green, fill = False, linewidth=4)
            plt.gca().add_patch(circle)
            plt.title('Kernel from stars', fontsize=15)
            plt.tight_layout()
            #plt.legend()
            plt.show()
            plt.plot(np.linspace(0,second_cutout*2, second_cutout*2),stars_sum_norm[second_cutout-1:second_cutout+1, :][0], label='my kernel', color='m')
            plt.plot(np.linspace(0,len(imageKernel.array), len(imageKernel.array)),imageKernel.array[int(len(imageKernel.array)/2)-1:int(len(imageKernel.array)/2)+1, :][0], label='lsst kernel', color='k')
            plt.legend()

    return stars_sum_norm


def isolated_coords(ra, dec, thresh=6/3600):
    """
    function that finds sources that are isolated within a certain distance
    
    If isolated = 1, then is True, if its 0 then is False
    
    """
    
    
    df = pd.DataFrame(columns=['coord_ra', 'coord_dec', 'isolated'])
    df['coord_ra'] = ra
    df['coord_dec'] = dec
    df['isolated'] = True
    
    coords = np.column_stack((ra,dec))
    check_isolation = pd.DataFrame(columns=['coords', 'isolated'])
    
    for pair in it.combinations(coords, 2):
        ra1, dec1 = pair[0][0],  pair[0][1]
        ra2, dec2 = pair[1][0],  pair[1][1]
        dist = np.sqrt((ra1-ra2)**2 + (dec1-dec2)**2)
        if dist>thresh:
            df.loc[(df['coord_ra']==ra1) & (df['coord_dec']==dec1), 'isolated']=df.loc[(df['coord_ra']==ra1) & (df['coord_dec']==dec1), 'isolated'] * True
            df.loc[(df['coord_ra']==ra2) & (df['coord_dec']==dec2), 'isolated']=df.loc[(df['coord_ra']==ra2) & (df['coord_dec']==dec2), 'isolated'] * True
            
        else:
            df.loc[(df['coord_ra']==ra1) & (df['coord_dec']==dec1), 'isolated']= df.loc[(df['coord_ra']==ra1) & (df['coord_dec']==dec1), 'isolated'] * False
            df.loc[(df['coord_ra']==ra2) & (df['coord_dec']==dec2), 'isolated']=df.loc[(df['coord_ra']==ra2) & (df['coord_dec']==dec2), 'isolated'] * False
            
    
    return df


def return_isolated(x, y, thresh=40):
    """
    
    Function that returns indexes of isolated sources
    
    """
    a = np.array(list(zip(x, y)))
    len_srcs = len(a)
    
    distances = sp.spatial.distance.cdist(a,a)

    np.fill_diagonal(distances, sp.inf)
    mins = distances.argmin(0)

    mins_values = distances.min(axis=0)

    idxs, = np.where(mins_values>thresh)


    return idxs


def check_psf_after_conv(repo, visit, ccdnum, collection_calexp, conv_image_array = None, plot=False, sn=5, cutout=24, cutout_for_psf=12, isolated=True, dist_thresh=40, flux_thresh=None):
    '''

    Finds the empirical psf for an image processed with the LSST Science Pipelines
    
    input
    -----
    repo [string] : directory were data is processed
    visit []
    ccdnum [int] : number of the detector
    collection_calexp [string] : directory were the reduced image is 
    ns [int] : number of star 
    plot [bool] : if True, the kernel is plotted 
    cut [int] : 

    output
    ------
    stars_sum_norm

    '''   
    field = collection_calexp.split('/')[-1]
    arcsec_to_pixel = 0.27 #arcsec/pixel
    
    butler = Butler(repo)
    ## take the stars of an image 
    
    stars_table = Select_table_from_one_calib_exposure(repo, visit, ccdnum, collection_calexp, stars=True, s_to_n_star=sn).to_pandas()
    
    stars_table = stars_table[stars_table['calib_photometry_used']==True]
    
    sn_stars = np.array(stars_table['base_PsfFlux_instFlux']/stars_table['base_PsfFlux_instFluxErr']) # instrumental signal to noise 
    inst_star_flux = stars_table['base_PsfFlux_instFlux']
    stars_table['inst_s_to_n'] = sn_stars
    # add signal to noise in the table of stars 
    stars_table = stars_table.sort_values(by=['inst_s_to_n'], ascending=False)
    
    # discard stars that are too close together 
    x_pix = np.array(stars_table['base_SdssCentroid_x'])
    y_pix = np.array(stars_table['base_SdssCentroid_y'])
    
    #if isolated:
    #    idx_isolated = return_isolated(x_pix, y_pix, thresh=dist_thresh)
    #    stars_table.loc[idx_isolated, 'isolated'] = 1 # test_isolation['isolated']
    #    stars_table = stars_table[stars_table['isolated']==1]
    
    print('Number of stars to construct empirical Kernel: ', len(stars_table))
    #ra_stars =  np.array(stars_table['coord_ra_ddegrees'])
    #dec_stars = np.array(stars_table['coord_dec_ddegrees'])
    
    calexp = butler.get('calexp', visit = visit, detector = ccdnum, instrument='DECam', collections=collection_calexp)
    
    data_cal = np.asarray(calexp.image.array, dtype='float64')
    wcs = calexp.getWcs()

    visits = [visit for i in range(len(stars_table))]
    print('number of available stars: ', len(stars_table))
    x_half_width = cutout
    y_half_width = cutout
    
    RAs = []
    DECs = []
    starn = 0
    collected_stars = {}
    centroid_stars = {}
    sum_stars = np.zeros((cutout_for_psf*2+1, cutout_for_psf*2+1))
    sum_stars_conv = np.zeros((cutout_for_psf*2+1, cutout_for_psf*2+1))
    
    number_stars = 0
    
    for x, y in zip(x_pix, y_pix):

        #obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        #x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        
        x_pix_new, y_pix_new = centering_coords(data_cal, x, y, int(cutout/2), show_stamps=plot, how='sep', minarea=3)
        calexp_cutout_to_use = data_cal[round(y_pix_new[0])-cutout_for_psf:round(y_pix_new[0])+cutout_for_psf+1, round(x_pix_new[0])-cutout_for_psf:round(x_pix_new[0])+cutout_for_psf+1]
        
        if conv_image_array is not None:
            x_pix_conv, y_pix_conv = centering_coords(conv_image_array, x_pix_new, y_pix_new, int(cutout/2), show_stamps=plot, how='sep', minarea=3)
            conv_cutout_to_use = conv_image_array[round(y_pix_conv[0])-cutout_for_psf:round(y_pix_conv[0])+cutout_for_psf+1, round(x_pix_conv[0])-cutout_for_psf:round(x_pix_conv[0])+cutout_for_psf+1]
            sum_stars_conv += conv_cutout_to_use
            
        sum_stars += calexp_cutout_to_use
        number_stars+=1
        
        if plot:

            fig = plt.figure(figsize=(8, 5))
            m, s = np.mean(calexp_cutout_to_use), np.std(calexp_cutout_to_use)
            plt.imshow(calexp_cutout_to_use, cmap='rocket', origin='lower', vmin=m, vmax=m+2*s)
            plt.colorbar()
            plt.scatter(cutout_for_psf, cutout_for_psf, color=neon_green, marker='x', linewidth=3, label='rounded center')
            plt.title('star number {} in sicence after centering'.format(number_stars), fontsize=15)
            plt.tight_layout()
            plt.legend()
            plt.show()
    
    stars_sum_norm = sum_stars/np.sum(np.sum(sum_stars))

    if conv_image_array is not None:
        stars_sum_conv_norm = sum_stars_conv/np.sum(np.sum(sum_stars_conv))
    else:
        stars_sum_conv_norm = sum_stars_conv
        
    return stars_sum_norm, stars_sum_conv_norm





def empirical_kernel(repo, visit, ccdnum, collection_calexp, plot=False, sn=10, cut=10):
    '''

    Construct empirical kernel for image processed with the LSST Science Pipelines
    
    input
    -----
    repo [string] : directory were data is processed
    visit []
    ccdnum [int] : number of the detector
    collection_calexp [string] : directory were the reduced image is 
    ns [int] : number of star 
    plot [bool] : if True, the kernel is plotted 
    cut [int] : 

    output
    ------
    stars_sum_norm

    '''   
    field = collection_calexp.split('/')[-1]
    directory = '{}/LSST_notebooks/empirical_kernel/Kernel_{}_{}_{}'.format(main_root, visit, field, ccdnum)
    file = '{}.pickle'.format(directory)
    print(visit)

    try:
        with open(file, 'rb') as handle:
            stars_sum_norm = pickle.load(handle)
    except:
        butler = Butler(repo)
        # We retrieve table of stars found in the image 
        stars_table = Select_table_from_one_calib_exposure(repo, visit, ccdnum, collection_calexp, stars=True, s_to_n_star=sn).to_pandas()
        sn_stars = np.array(stars_table['base_PsfFlux_instFlux']/stars_table['base_PsfFlux_instFluxErr']) # instrumental signal to noise 
        inst_star_flux = stars_table['base_PsfFlux_instFlux']
        stars_table['inst_s_to_n'] = sn_stars
        stars_table = stars_table.sort_values(by=['inst_s_to_n'], ascending=False)
        stars_table = stars_table.drop_duplicates(subset = ['coord_ra_ddegrees'], keep='first')
        print('Number of stars to construct empirical Kernel: ', len(stars_table))
        ra_stars =  np.array(stars_table['coord_ra_ddegrees'])
        dec_stars = np.array(stars_table['coord_dec_ddegrees'])
        arcsec_to_pixel = 0.27 #arcsec/pixel
        calexp = butler.get('calexp', visit= visit, detector= ccdnum, instrument='DECam', collections=collection_calexp)
        wcs = calexp.getWcs()

        visits = [visit for i in range(len(stars_table))]
        number_stars = 0
        cutout = int(cut/arcsec_to_pixel) 
        thresh = 60
        print('number of available stars: ', len(stars_table))
        x_half_width = cutout
        y_half_width = cutout
        
        RAs = []
        DECs = []
        starn = 0
        collected_stars = {}
        centroid_stars = {}
        second_cutout = cutout-5
        sum_stars = np.zeros((2*second_cutout, 2*second_cutout))

        for ra, dec, fl in zip(ra_stars, dec_stars, inst_star_flux):

            obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
            afwDisplay.setDefaultMaskTransparency(100)
            afwDisplay.setDefaultBackend('matplotlib')
            x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
            calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
            thresh = fl*0.6

            objects = sep.extract(calexp_cutout.image.array, thresh, minarea=30)

            if len(objects)!=1:
                continue
            else:
                RAs.append(ra)
                DECs.append(dec)
                
                x_pix_cen = round(objects[0]['x'],0)
                y_pix_cen = round(objects[0]['y'],0)
                
                calexp_cutout_to_use = calexp_cutout.image.array[int(y_pix_cen-second_cutout):int(y_pix_cen+second_cutout), int(x_pix_cen-second_cutout):int(x_pix_cen+second_cutout)]
                collected_stars['star_{}'.format(number_stars)] = calexp_cutout_to_use
                centroid_stars['star_{}'.format(number_stars)] = [x_pix_cen, y_pix_cen]
                sum_stars += calexp_cutout_to_use
            
            if plot:

                fig = plt.figure(figsize=(10, 5))
                plt.imshow(calexp_cutout_to_use, cmap='rocket', origin='lower', vmin=0, vmax=100)
                plt.colorbar()
                plt.scatter(second_cutout, second_cutout, color=neon_green, marker='x', linewidth=3, label='new center')
                plt.title('star number {}'.format(number_stars), fontsize=15)
                plt.tight_layout()
                plt.legend()
                plt.show()

        stars_sum_norm = sum_stars/np.sum(np.sum(sum_stars))

        with open(file, 'wb') as handle:        
            pickle.dump(stars_sum_norm, handle, protocol = pickle.HIGHEST_PROTOCOL)

    return stars_sum_norm

def all_ccds_Jorge(field, ccds, folder, sfx='mag'):
    '''
    plots and saves the LCs from Martinez-Palomera's aperture photometry

    input:
    -----
    field : [str] 
    ccds : [list]
    folder : [str]

    output:
    ------
    None

    '''
    if len(ccds)==0:
        print('No ccds where found to be used in the LC difference construction')
        return

    cands = Find_sources(sibling_allcand, field)
    index = cands.index
    
    #ccds = [f.split('_')[2] for f in cands.internalID]
    plt.figure(figsize=(10,6))
    norm = matplotlib.colors.Normalize(vmin=0,vmax=32)
    c_m = matplotlib.cm.plasma

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    T = np.linspace(0,27,len(ccds))
    i=0
    ccds = np.unique(ccds)

    for i in range(len(ccds)):
        SIBLING = '/home/pcaceres/Jorge_LCs/'+cands.internalID.loc[index[i]] +'_g_psf_ff.csv'
        #sfx = 'flx'
        factor = 0.75
        x,y,yerr = compare_to(SIBLING, sfx, factor, beforeDate=57072)
        print(ccds[i])
        plt.errorbar(x-min(x),y , yerr=yerr,  capsize=4, fmt='o', label='Martinez-Palomera et al. 2020 {}'.format(ccds[i]), ls ='dotted', color = s_m.to_rgba(T[i]))
        i+=1
    plt.legend()
    plt.title('Aperture Fotometry {} by Martinez-Palomera for {}'.format(sfx, field))
    plt.savefig('{}/LSST_notebooks/light_curves/{}/{}_all_ccds_{}_Jorge_aperture.png'.format(main_root, folder, field, sfx))
    plt.show()
    
    return

def Find_ccd_for_ra_dec(repo, field, collection_calexp, ra, dec):
    '''
    Find the ccd that corresponds to a certain ra, dec position

    Input:
    -----
    repo
    collection_calexp
    ra
    dec

    Output:
    ------
    ccd : [int] detector number 
    '''
    CCDS = []
    data = get_all_exposures(repo, 'science')
    visits = data[data['target_name']=='{}'.format(field)].exposure
    index_visits = visits.index
    for key in detector_nomenclature:
        try:
            #print(detector_nomenclature[key])
            Calib_cropped(repo, collection_calexp, ra, dec, [visits[visits.index[0]]],  detector_nomenclature[key], cutout=40, s=20)
            #lp.Calib_plot(repo, collection, ra, dec, [visits[visits.index[0]]], detector_nomenclature[key], s=20)
            CCDS.append(detector_nomenclature[key])
            #lp.Calib_cropped(repo, collection ra, dec, [visits[0]],  detector_nomenclature[key], cutout=40, s=20)
        except:
            continue

    
    return CCDS 

def Find_sources(sibling_allcand, field):
    """
    Finds sources in sibling_allcand with field 
    
    Input:
    -----
    sibling_allcand : [pd.DataFrame]
    field : [str]
    
    Output:
    ------
    selection : [pd.DataFrame]
    """
    ccds = 0 
    px = 0 
    py = 0
    
    fs = []
    for intID in sibling_allcand.internalID:
        if type(intID)!=str:
            #print(intID)
            pass
        else:
            fs.append(intID[:11])
    #print(fs)
    if len(fs)==0:
        "No fields found uwu"
    j, = np.where(np.array(fs) == field)
    if len(j)==0:
        print('no ccds found for that field uwu')
    selection = sibling_allcand.loc[j]
    return selection[['internalID', 'SDSS12', 'raMedian', 'decMedian']]

def values_under_aperture(matrix, x, y, r):
    """
    Finds array values under circular aperture in a non efficient way...
    
    Inputs:
    ------
    matrix: [ndarray matrix]
    x : [float]
    y : [float]
    r : [float]
    
    Outputs:
    -------
    values : [np.array]
    """
    values = []
    
    row, col = np.shape(matrix)
    for i in range(row):
        for j in range(col):
            
            if np.sqrt((x-j)**2 + (y-i)**2) <= r:
                values.append(matrix[i][j])
            else:
                pass
    return np.array(values) 


def do_convolution_image2(repo, main_root, visit, ccd_num, collection_calexp, xpix, ypix, worst_seeing_visit=None, worst_seeing = None, seeing_image=None):   
    '''
    does convolution of an image 

    input
    -----
    main_root
    visit
    ccd_num
    collection_calexp

    output
    ------
    convoled_image


    '''
    #repo = '/home/pcaceres/data_hits'
    butler = Butler(repo)
    field = collection_calexp.split('/')[-1]
    #worst_seeing = max(Seeing)
    #sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
    #arcsec_to_pixel = 0.27 #arcsec/pixel
    stddev_kernel = 0
    if type(worst_seeing) != type(None) and type(seeing_image)!=type(None):
        stddev_im = seeing_image
        stddev_wi = worst_seeing
        stddev_kernel = np.sqrt(stddev_wi**2 - stddev_im**2)
        #print('stddev Kernel !!!!!!!!!: ', stddev_kernel)
        directory = '{}/LSST_notebooks/convolved_images/convolved_image_{}_{}_{}_sigmaKernel_{}'.format(main_root, visit, field, ccd_num, stddev_kernel)

        if stddev_kernel == 0 :
            kernel = None
        else:
            kernel =  create_matching_kernel(im, wim)
            #Gaussian2DKernel(x_stddev=stddev_kernel)

        print('DOING gaussian convolution .... computed kernel')
        kernel = Gaussian2DKernel(x_stddev=stddev_kernel)  
    #if type(worst_seeing_visit) != type(None):
    #    kernel = make_kernel(repo, worst_seeing_visit, ccd_num, collection_calexp, ns=10, cut=10)
    #    directory = '{}/LSST_notebooks/convolved_images/convolved_image_myKernel_{}_{}_{}_seeing_visit_{}'.format(main_root, visit, field, ccd_num, worst_seeing_visit)
    
    file = '{}.pickle'.format(directory)
    convolved_image = {}

    try:
        with open(file, 'rb') as handle:
            convolved_image = pickle.load(handle)
    except:
    #if not os.path.exists(file):
        calexp = butler.get('calexp', visit=visit, detector=ccd_num, collections=collection_calexp, instrument='DECam') 
        calexp_array = np.asarray(calexp.image.array, dtype='float')
        if type(worst_seeing) != type(None):  
            if stddev_kernel == 0:
                conv = calexp_array
            else:
                conv = scipy.signal.convolve2d(calexp_array, kernel)
            convolved_image[visit] = conv
            print(' I did the convolution !!!! ')

        #if type(worst_seeing_visit) != type(None):
        #    astropy_conv = ndimage.convolve(calexp_array, kernel, mode='constant', cval=0.0)
        #    convolved_image[visit] = astropy_conv


        with open(file, 'wb') as handle:        
            pickle.dump(convolved_image, handle, protocol = pickle.HIGHEST_PROTOCOL)
    #else:
    #    with open(file, 'rb') as handle:
    #        convolved_image = pickle.load(handle)
    return convolved_image, stddev_kernel


def custom_convolve(input_array, kernel):
    '''
    input
    ------
    input_array [np.darray]
    kernel [np.darray]

    output
    ------
    result [np.darray]
    '''
    # Get the dimensions of the kernel
    kernel_height, kernel_width = kernel.shape
    
    # Perform the convolution without padding
    result = scipy.signal.convolve2d(input_array, kernel, mode='same', boundary='symm')
    
    return result

def custom_convolve2(input_array, kernel):

    # Calculate the amount of padding needed for each dimension
    pad_height = kernel.shape[0] - 1
    pad_width = kernel.shape[1] - 1
    
    # Pad the input array
    padded_array = np.pad(input_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Perform convolution
    result = scipy.signal.convolve(padded_array, kernel, mode='valid')  # 'valid' mode ensures output size matches
    
    # Calculate the shift needed for alignment
    vertical_shift = pad_height // 2
    horizontal_shift = pad_width // 2
    
    # Shift the result to match the position of the original matrix
    shifted_result = np.roll(result, (-vertical_shift, -horizontal_shift), axis=(0, 1))
    
    return shifted_result


def objective_function(alpha, image_kernel, worst_kernel, resize=False):
    
    airy_kernel = AiryDisk2DKernel(alpha).array
    
    target_kernel = custom_convolve(image_kernel, airy_kernel) # convolve
    
    if resize:
        max_size = target_kernel.shape[0]
        worst_kernel_resized = np.zeros((max_size, max_size))
        worst_kernel_resized[:worst_kernel.shape[0], :worst_kernel.shape[1]] = worst_kernel#airy_kernel.array
    else:
        worst_kernel_resized = worst_kernel
        
    # Normalize both resized kernels
    target_kernel_normalized = target_kernel / np.sum(target_kernel)
    worst_kernel_normalized = worst_kernel_resized / np.sum(worst_kernel_resized)
    
    # Calculate MSE between the two kernels
    mse = np.mean((target_kernel_normalized - worst_kernel_normalized)**2)
    return mse

def do_convolution_image_bla(repo, main_root, visit, ccd_num, collection_calexp, ra, dec, worst_seeing_visit=None, mode='Eridanus'):   
    '''
    does convolution of an image 

    input
    -----
    main_root
    visit
    ccd_num
    collection_calexp

    output
    ------
    convoled_image


    '''
    #repo = '/home/pcaceres/data_hits'
    print('Entering the function to convolve...')
    butler = Butler(repo)
    #rint(butler)
    field = collection_calexp.split('/')[-1]
    calexp = butler.get('calexp', visit=visit, detector=ccd_num, collections=collection_calexp, instrument='DECam') 
    calexp_array = np.asarray(calexp.image.array, dtype='float')
    #print('calexp array: ', calexp_array)
    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    wcs = calexp.getWcs()
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
    directory = '{}/LSST_notebooks/convolved_images/convolved_image_{}_{}_{}_{}_{}_photutils_kernel'.format(main_root, visit, field, ccd_num, x_pix, y_pix)
    #print('directory to search for convolved images: ', directory)
    file = '{}.pickle'.format(directory)
    convolved_image = {}

    try:
        #print('trying to open the file...')
        with open(file, 'rb') as handle:
            convolved_image = pickle.load(handle)

    except:
        #print('no file found, so doing the convolution')
        #print('is the visit equal to the worst one?', visit == worst_seeing_visit)
        
        if visit == worst_seeing_visit:
            
            convolved_image[visit] = calexp_array
            return convolved_image, 0 
        #print('lets look at the worst image...')
        worst_calexp = butler.get('calexp', visit=worst_seeing_visit, detector=ccd_num, collections=collection_calexp, instrument='DECam')
        #print(worst_calexp)
        worst_psf =  worst_calexp.getPsf()
        worst_seeing = worst_psf.computeShape(worst_psf.getAveragePosition()).getDeterminantRadius()
        #print('worst psf: ', worst_psf)
        
        obj_pos_2d = lsst.geom.Point2D(ra, dec)
        wimageKernel = worst_psf.computeKernelImage(obj_pos_2d)
        
        psf = calexp.getPsf()
        imageKernel = psf.computeKernelImage(obj_pos_2d)

        im = imageKernel.array
        wim = wimageKernel.array


        #print('about to convolve...')

        if (mode == 'Eridanus'):
            alpha = 0.3
            beta = 0.25
            kernel = create_matching_kernel(im, wim, SplitCosineBellWindow(alpha=alpha, beta=beta))

        if (mode == 'HITS'):
            
            initial_alpha = 1.5 
        
            result = minimize(objective_function, initial_alpha, args=(im, wim), method='Nelder-Mead')

            # Extract the optimized alpha
            optimized_alpha = result.x[0]

            print('alpha = ', optimized_alpha)
            kernel = AiryDisk2DKernel(optimized_alpha).array

        print('using the configuration for: ', mode)

        #
        #, window = TopHatWindow(0.6))

        

        #print('')
        conv = custom_convolve(calexp_array, kernel) #scipy.signal.convolve(kernel, calexp_array)
        #conv = np.clip(conv, 0, np.inf)  # Clip negative values to zero
        #conv = gaussian_filter(conv, sigma=worst_seeing)  # Apply Gaussian blur
        
        #conv[conv < 0] = 0

        convolved_image[visit] = conv
        print(' I did the convolution !!!! ')

        with open(file, 'wb') as handle:        
            print('ups I did it again... ')
            #pickle.dump(convolved_image, handle, protocol = pickle.HIGHEST_PROTOCOL)

    return convolved_image, 0



def do_convolution_image(calexp_array, var_calexp_array, im, wim, mode='Eridanus', type_kernel='mine', visit=None, worst_visit=None, worst_calexp=None, stars_in_common=None, calexp_exposure = None, worst_calexp_exposure=None):   
    '''
    does convolution of an image 

    input
    -----
    repo [string] : directory were processed images are
    main_root [string] : home directory
    visit [int] : number of exposure
    ccd_num [int] : number of detector
    collection_calexp [string] : directory of reduced images 
    ra [float] : Right ascention in radians
    dec [float] : Declination in radians 
    worst_seeing_visit [int/None] : 
    mode [string] :
    type_kernel [string]: 
    
    output
    ------
    convoled_image


    '''
    print('About to do the convolution...') 
    
    print('shape of kernel: ', im.shape)
    print('shape of worst kernel: ', wim.shape)
    
    
    if visit == worst_visit :
        
        conv_image = calexp_array
        conv_variance = var_calexp_array
        kernel = wim
        return conv_image, conv_variance, kernel
    
    
    if (type_kernel=='mine'):

        if (mode == 'Eridanus'):
            alpha = 0.3
            beta = 0.25
            kernel = create_matching_kernel(im, wim, SplitCosineBellWindow(alpha=alpha, beta=beta))   

        if (mode == 'HITS' or mode=='HiTS'):
            
            initial_alpha = 1.5 
            result = minimize(objective_function, initial_alpha, args=(im, wim), method='Nelder-Mead', bounds=[(0, None)])    
            optimized_alpha = result.x[0]
            print('alpha = ', optimized_alpha)
            kernel = AiryDisk2DKernel(optimized_alpha).array
    
    elif (type_kernel=='lsst'):
        
        warpTask = WarpAndPsfMatchTask(name="warp_and_psf_match")
        
        wcs_cal = worst_calexp_exposure.getWcs()
        model_psf = worst_calexp_exposure.getPsf()

        result = warpTask.run(
                exposure=calexp_exposure,
                wcs=wcs_cal,
                modelPsf = model_psf,
                makePsfMatched=True)
        
        conv_image = np.asarray(result.psfMatched.image.array, dtype='float64')
        
        conv_variance = np.zeros(conv_image.shape) 
        kernel = np.zeros(wim.shape)
        
        return conv_image, conv_variance, kernel
        
        
        
    else:
        star_pairs = get_starpairs(calexp_array, worst_calexp, visit, worst_visit, stars_in_common)
        kernel = get_Panchos_matching_kernel(star_pairs)
        #kernel/=kernel.sum() # Here we normalize the Kernel
    
    print('this kernel type {} has shape: '.format(type_kernel), kernel.shape)    
    print('using the configuration for: ', mode)    

    # doing convolution 
    
    if (type_kernel=='mine'):
        conv_image = custom_convolve(calexp_array, kernel) 
        conv_variance = custom_convolve(var_calexp_array, kernel**2)
    else:
        conv_image = _filter_data(calexp_array, kernel, mode='nearest')
        conv_variance = _filter_data(var_calexp_array, kernel**2, mode='nearest')
        
    return conv_image, conv_variance, kernel

def get_starpairs(calexp, worst_calexp, visit, worst_visit, stars_in_common):
    
    visits = [visit, worst_visit]
    cal_images = [calexp, worst_calexp]
    stars_dict = {}
    nstars = len(stars_in_common)
    
    for j, v in enumerate(visits):
    
        coords_stars = {}
        
        X_pix_stars = np.array(stars_in_common['base_SdssCentroid_x_{}'.format(v)], dtype='float64')
        Y_pix_stars = np.array(stars_in_common['base_SdssCentroid_y_{}'.format(v)], dtype='float64')
        
        cal_array = cal_images[j]
        #cal_array = np.asarray(cal_image.image.array, dtype='float64')
 
        array_of_stars = np.zeros((nstars, 46, 46))
        cutout = 23
        bad_stars = []
        
        for i in range(nstars):        
            xpix = X_pix_stars[i]
            ypix = Y_pix_stars[i]

            try:
                x_pix_new, y_pix_new = centering_coords(cal_array, xpix, ypix, int(cutout/2)+1, show_stamps=True, how='sep', minarea=3)
                if x_pix_new == xpix and y_pix_new == ypix:
                    
                    print('if centering is not working, we skip this star')
                    
                    bad_stars.append(i)
                    array_of_stars[i] = np.zeros((46,46))
                    continue
                    
                calexp_cutout_to_use = cal_array[round(y_pix_new[0])-cutout:round(y_pix_new[0])+cutout, round(x_pix_new[0])-cutout:round(x_pix_new[0])+cutout].copy(order='C')
                

                calexp_bkg = cal_array[round(ypix)-2*cutout:round(ypix)+2*cutout, round(xpix)-2*cutout:round(xpix)+2*cutout].copy(order='C')

                sigma_clip = SigmaClip(sigma=3.,maxiters=2)
                bkg_estimator = MedianBackground()

                bkg = Background2D(calexp_cutout_to_use, (cutout * 2, cutout * 2), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

                calexp_cutout_to_use -= bkg.background
                array_of_stars[i] = calexp_cutout_to_use
            
            except ValueError:
                bad_stars.append(i)
                array_of_stars[i] = np.zeros((46,46))
                
        stars_dict[v] = array_of_stars 
        
        
    bad_stars = np.array(bad_stars)
    bad_stars = np.unique(bad_stars)
    print('bad stars: ', bad_stars)
    print('stars array: ', stars_dict[visit].shape)
    if len(bad_stars) > 0:
        stars_image = np.delete(stars_dict[visit], bad_stars, axis=0)
        stars_wimage = np.delete(stars_dict[worst_visit], bad_stars, axis=0)
    
        print('stars array after: ', stars_image.shape)
    
    else:
        stars_image = stars_dict[visit]
        stars_wimage = stars_dict[worst_visit]
   
    starpairs = np.stack([stars_wimage, stars_image])
    
    return starpairs


def get_Panchos_matching_kernel(starpairs):
    
    kern = kkernel(81)
    
    _, nstars, nside, _ = np.shape(starpairs)
    npsf = nside - kern.nf
    nfh = int(kern.nf / 2)
    
    # create array with pairs of stars
    pairs = []
    for i in range(nstars):
        star1 = starpairs[1][i]
        star2 = starpairs[0][i]
        pairs.append([star1, star2])
        
    #sol = kern.solve(npsf, pairs)#.solfilter
    kern.solve(npsf, pairs)
    sol = kern.solfilter
    #print('shape of sol from kernel pancho: ',sol.shape)
    
    sol /= sol.sum()
    
    print('sum of kernel: ', sol.sum())
    
    return sol



def Panchos_kernel(repo, collection_calexp, ccd_num, visit, worst_seeing_visit):
    
    visits = [visit, worst_seeing_visit]
    stars_dict = {}
    stars_in_common = Inter_Join_Tables_from_LSST(repo, visits, ccd_num, collection_calexp)
    nstars = len(stars_in_common)
    butler = Butler(repo)

    for j in range(len(visits)):
    
        coords_stars = {}
        RA = np.array(stars_in_common['coord_ra_ddegrees_{}'.format(visits[j])], dtype=float)
        DEC = np.array(stars_in_common['coord_dec_ddegrees_{}'.format(visits[j])], dtype=float)
        calexp = butler.get('calexp',visit=visits[j], detector=ccd_num , collections=collection_calexp, instrument='DECam')
        photocalib_calexp = calexp.getPhotoCalib()
        wcs = calexp.getWcs()
        data_calexp = np.asarray(calexp.image.array, dtype='float')
        psf = calexp.getPsf()
        exp_visit_info = calexp.getInfo().getVisitInfo()
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)        
        d = visit_date_astropy.mjd    
        array_of_stars = np.zeros((nstars, 46, 46))
        
        for i in range(nstars):        
            ra_star = RA[i]
            dec_star = DEC[i]
            obj_pos_lsst = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
            cutout = 46
            calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout, cutout))
            array_of_stars[i] = np.array(calexp_cutout.image.array)
        
        stars_dict[visits[j]] = array_of_stars 
    
    kern = kkernel(81)            
    starpairs = np.stack([stars_dict[worst_seeing_visit], stars_dict[visit]])
    _, nstars, nside, _ = np.shape(starpairs)
    npsf = nside - kern.nf
    nfh = int(kern.nf / 2)
    
    # create array with pairs of stars
    pairs = []
    for i in range(nstars):
        star1 = starpairs[1][i]
        star2 = starpairs[0][i]
        pairs.append([star1, star2])
    sol = kern.solve(npsf, pairs)

    return kern

def Find_stars(ra, dec, width, height, n, seed=[True, 200]):
    """
    Finds n stars in a rectangular aperture, which I intend to be the ccd size.
    
    Inputs:
    ------
    ra : [float] right ascention position in degrees
    dec : [float] declination position in degrees 
    width : [float] width of rectangular aperture in arcsec 
    height : [float] height of reclangular aperture in arcsec
    n : [int] number of stars we wan to find
    seed : [tuple (bool, int)] if bool is True, we set a random seed equal to int
    
    Outputs:
    -------
    Table : [astropy.table] table with the selected stars 
    
    """
    
    c = SkyCoord(ra * u.degree, dec * u.degree, frame='icrs')
    result = Vizier.query_region(c,
                                 width=width * u.arcsec, height = height * u.arcsec,
                                 catalog='I/345/gaia2',
                                 column_filters={'Gmag': '>15', 'Gmag': '<25','Var':"!=VARIABLE"})
    if seed[0]:
        random.seed(seed[1])
    
    len_table = len(result[0])
    random_indexes = random.sample(range(len_table), n)
    
    table = result[0][random_indexes]


    return table

def Select_table_from_one_calib_exposure(repo, visit, ccdnum, collection_calexp, stars=True, s_to_n_star = 5, isolated=True, distance_isolation=15):# 20
    """
    Selects stars or all detected sources from calibrated exposures 

    input
    ------
    repo [string] : 
    visit [int] : 
    ccdnum [int] :
    collection_calexp [string] : 
    stars [bool] : 

    output
    -------
    phot_table [Table]: astropy table with the sources

    """
    
    butler = Butler(repo)
    calexp = butler.get('calexp',visit=visit, detector=ccdnum , collections=collection_calexp, instrument='DECam')
    photocalib_calexp = calexp.getPhotoCalib()

    src = butler.get('src',visit=visit, detector=ccdnum , collections=collection_calexp, instrument='DECam')
    src = photocalib_calexp.calibrateCatalog(src)
    src_pandas = src.asAstropy().to_pandas()
    src_pandas['coord_ra_trunc'] = [Truncate(f, 5) for f in np.array(src['coord_ra'])]
    src_pandas['coord_dec_trunc'] = [Truncate(f, 5) for f in np.array(src['coord_dec'])]
        
    if isolated:
        
        x_pix = src_pandas['base_SdssCentroid_x']
        y_pix = src_pandas['base_SdssCentroid_y']
        idx_isolated = return_isolated(x_pix, y_pix, thresh=distance_isolation)
        src_pandas.loc[idx_isolated, 'isolated'] = 1
        mask = src_pandas['isolated'] == 1
        src_pandas = src_pandas[mask]
    
    src_pandas = src_pandas[src_pandas['base_PixelFlags_flag_saturated']==False]
    
    if stars:
        mask = (src_pandas['base_ClassificationExtendedness_value'] == 0.0) & (src_pandas['base_PsfFlux_instFlux']/src_pandas['base_PsfFlux_instFluxErr'] > s_to_n_star)
        
        #mask = (src_pandas['calib_photometry_used'] == True) & (src_pandas['base_PsfFlux_instFlux']/src_pandas['base_PsfFlux_instFluxErr'] > s_to_n_star)
        stars_photometry = src_pandas[mask]
    else:
        stars_photometry = src_pandas
        
    sources_masked = stars_photometry.dropna(subset=['coord_ra', 'coord_dec'])
    phot_table = Table.from_pandas(sources_masked)
    phot_table['coord_ra_ddegrees'] = (phot_table['coord_ra'] * u.rad).to(u.degree)
    phot_table['coord_dec_ddegrees'] = (phot_table['coord_dec'] * u.rad).to(u.degree)
    photom_mean = photocalib_calexp.getCalibrationMean()
    phot_table['phot_calib_mean'] = photom_mean * np.ones(len(phot_table))
    phot_table['src_id'] = phot_table['id']

    return phot_table


def Select_table_from_one_exposure(repo, visit, ccdnum, collection_diff, well_subtracted=True):
    """
    Selects star used for photometry from a src table, it also can ensure that it is well subtracted in the 
    difference image 

    Input:
    ------
    repo : [string]
    visit : [ndarray]
    ccdnum : [int]
    collection_diff : [string]
    well_subtracted : [bool] True by default
    
    Output:
    -------
    phot_table : [Astropy Table]
    """

    butler = Butler(repo)
    diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    try:
        coadd = butler.get('goodSeeingDiff_matchedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    except:
        coadd = butler.get('goodSeeingDiff_warpedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')

    photocalib_coadd = coadd.getPhotoCalib()
    photom_mean = photocalib_coadd.getCalibrationMean()

    wcs = diffexp.getWcs()
    
    src = butler.get('src',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    src = photocalib_coadd.calibrateCatalog(src)
    diaSrcTable = butler.get('goodSeeingDiff_diaSrc', visit=visit, detector=ccdnum, collections=collection_diff, instrument='DECam') 
    diaSrcTable = photocalib_coadd.calibrateCatalog(diaSrcTable)
    diaSrcTable_pandas = diaSrcTable.asAstropy().to_pandas()
    coord_ra_diff = np.array(diaSrcTable_pandas['coord_ra'])
    coord_dec_diff = np.array(diaSrcTable_pandas['coord_dec'])

    src_pandas = src.asAstropy().to_pandas()
    src_pandas['coord_ra_trunc'] = [Truncate(f, 5) for f in np.array(src['coord_ra'])]
    src_pandas['coord_dec_trunc'] = [Truncate(f, 5) for f in np.array(src['coord_dec'])]

    srcMatchFull = butler.get('srcMatchFull', visit=visit, detector=ccdnum, collections=collection_diff, instrument='DECam')
    srcMatchFull_pandas = srcMatchFull.asAstropy().to_pandas()
    srcMatchFull_pandas['coord_ra_trunc'] = [Truncate(f, 5) for f in np.array(srcMatchFull['ref_coord_ra'])]
    srcMatchFull_pandas['coord_dec_trunc'] = [Truncate(f, 5) for f in np.array(srcMatchFull['ref_coord_dec'])]

    x_pix_stars = []
    y_pix_stars = []

    for src in diaSrcTable:
        x_pix_stars.append(src.getX())
        y_pix_stars.append(src.getY())
 
    x_pix_stars = np.array(x_pix_stars)
    y_pix_stars = np.array(y_pix_stars)

    pdimx = 2048 
    pdimy = 4096 

    sources = pd.merge(src_pandas, srcMatchFull_pandas, on=['coord_ra_trunc', 'coord_dec_trunc'], how='outer')
    mask = (sources['base_PsfFlux_instFlux']/sources['base_PsfFlux_instFluxErr'] > 3)
    sources_masked = sources[mask]
    mask = (sources['src_calib_photometry_used'] == True) | (sources['src_calib_astrometry_used'] == True)
    sources_masked = sources[mask]

    if well_subtracted:
        bad_indexes = []
        for index, row in sources_masked.iterrows():
            ra = row['coord_ra']
            dec = row['coord_dec']
            distance = np.sqrt((coord_ra_diff - ra)**2 + (coord_dec_diff - dec)**2)
            j, = np.where(distance < 5e-6)
            if len(j)>0:
                bad_indexes.append(index)
        sources_masked = sources_masked.drop(np.unique(bad_indexes))
        sources_masked = sources_masked.reset_index()

    sources_masked = sources_masked.dropna(subset=['coord_ra', 'coord_dec'])
    phot_table = Table.from_pandas(sources_masked)
    phot_table['coord_ra_ddegrees'] = (phot_table['coord_ra'] * u.rad).to(u.degree)
    phot_table['coord_dec_ddegrees'] = (phot_table['coord_dec'] * u.rad).to(u.degree)
    phot_table['phot_calib_mean'] = photom_mean * np.ones(len(phot_table))
  
    return phot_table

def Gather_Tables_from_LSST(repo, visits, ccdnum, collection_diff, well_subtracted = True, tp='after_ID'):
    """
    From the src tables of LSST for each exposure, we select the sources that were used for photometry,
    which are the stars. We add them all in a dictionary whose keys are the visit number, and the 
    content is an astropy table with the stars.

    Input: 
    ------
    repo : [string]
    visits : [ndarray]
    ccdnum : [int]
    collection_diff : [string]
    well_subtracted : [bool]
    tp : [string]

    Output: 
    ------
    Dict_tables : [dict]

    """
    Dict_tables = {}
    for i in range(len(visits)):
        Dict_tables['{}'.format(visits[i])] = Select_table_from_one_calib_exposure(repo, visits[i], ccdnum, collection_diff)
    
    return Dict_tables

def Join_Tables_from_LSST(repo, visits, ccdnum, collection_diff, well_subtracted = True, tp ='after_ID'):
    """
    Joins src tables of LSST on the ra, dec truncated of the visits 
    """

    butler = Butler(repo)
    #Dict_tables = {}
    if type(visits) == int:
        visits = [visits]

    dictio =  Gather_Tables_from_LSST(repo, visits, ccdnum, collection_diff, well_subtracted = well_subtracted, tp=tp)
    big_table = 0
    i=0
    columns_picked = ['src_id', 'coord_ra', 'coord_dec', 'coord_ra_ddegrees', 'coord_dec_ddegrees', 'base_CircularApertureFlux_3_0_instFlux', 'base_PsfFlux_instFlux', 'base_PsfFlux_mag', 'base_PsfFlux_magErr','slot_PsfFlux_mag', 'phot_calib_mean', 'base_PixelFlags_flag_saturated', 'base_SdssCentroid_x', 'base_SdssCentroid_y']
    for key in dictio:
        if key == '{}'.format(visits[0]):
            big_table = dictio[key].to_pandas()[columns_picked]
            new_column_names = ['{}_{}'.format(c, key) for c in columns_picked]
            big_table.columns = new_column_names
            # antes estaba como: Truncate(f,4)
            big_table['coord_ra_trunc'] = [Truncate(f, 3) for f in np.array(big_table['coord_ra_ddegrees_{}'.format(key)])]
            big_table['coord_dec_trunc'] = [Truncate(f, 3) for f in np.array(big_table['coord_dec_ddegrees_{}'.format(key)])]
            #big_table['photo_calib_{}'.format(visits[0])] = photom_calib[0]
            #big_table = big_table.rename(columns = {'coord_ra_trunc_{}'.format():'src_id'})

            big_table['circ_aperture_to_nJy_{}'.format(key)] = big_table['phot_calib_mean_{}'.format(key)] * big_table['base_CircularApertureFlux_3_0_instFlux_{}'.format(key)]
            big_table['psf_flux_to_nJy_{}'.format(key)] = big_table['phot_calib_mean_{}'.format(key)] * big_table['base_PsfFlux_instFlux_{}'.format(key)]
            big_table['psf_flux_to_mag_{}'.format(key)] = [pc.FluxJyToABMag(f*1e-9)[0] for f in big_table['psf_flux_to_nJy_{}'.format(key)]]
            #big_table['base_SdssCentroid_x_{}'.format(key)] = 
            #big_table['base_SdssCentroid_y_{}'.format(key)] = 
            i+=1
        else:
            table = dictio[key].to_pandas()[columns_picked]
            new_column_names = ['{}_{}'.format(c, key) for c in columns_picked]
            table.columns = new_column_names
            table['coord_ra_trunc'] = [Truncate(f, 3) for f in np.array(table['coord_ra_ddegrees_{}'.format(key)])]
            table['coord_dec_trunc'] = [Truncate(f, 3) for f in np.array(table['coord_dec_ddegrees_{}'.format(key)])]
            #table['photo_calib_{}'.format(key)] = photom_calib[i]
            table['circ_aperture_to_nJy_{}'.format(key)] = table['phot_calib_mean_{}'.format(key)] * table['base_CircularApertureFlux_3_0_instFlux_{}'.format(key)]
            table['psf_flux_to_nJy_{}'.format(key)] = table['phot_calib_mean_{}'.format(key)] * table['base_PsfFlux_instFlux_{}'.format(key)]
            table['psf_flux_to_mag_{}'.format(key)] = [pc.FluxJyToABMag(f*1e-9)[0] for f in table['psf_flux_to_nJy_{}'.format(key)]]
            
            big_table = pd.merge(big_table, table, on=['coord_ra_trunc', 'coord_dec_trunc'], how='outer')

            i+=1
    big_table
    
    return big_table


def Inter_Join_Tables_from_LSST(repo, visits, ccdnum, collection_diff, well_subtracted =False, tp='after_ID', save=False, isolated=True):
    """
    returns the common stars used for calibration

    input:
    -----
    repo
    visits
    ccdnum
    collection_diff
    well_subtracted
    tp
    save

    output:
    ------
    phot_table [pandas dataFrame]:
    """
    big_table = Join_Tables_from_LSST(repo, visits, ccdnum, collection_diff,well_subtracted = well_subtracted, tp=tp)
    phot_table = big_table.dropna()
    #if isolated:
    #    phot_table = phot_table.drop_duplicates(subset=['coord_ra_trunc', 'coord_dec_trunc'])
    phot_table = phot_table.reset_index()
    #phot_table =  phot_table[phot_table['']]
    #phot_table = phot_table.drop('index')
    if save:
        field = collection_diff.split('/')[-1]
        phot_table.to_csv('stars_from_{}.txt'.format(field))
    return phot_table

def Find_stars_from_LSST_to_PS1(repo, visit, ccdnum, collection_diff, n, well_subtracted=True, verbose=False):
    """
    Finds n stars in a rectangular aperture, which I intend to be the ccd size.
    
    Inputs:
    ------
    repo :
    visit : 
    ccdnum : 
    collection_diff : 
    n : [int] number of stars we wan to find
    
    Outputs:
    -------
    stars_table : [astropy.table] table with the selected stars 
    
    """
    butler = Butler(repo)

    diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    wcs = diffexp.getWcs()
     
    src = butler.get('src',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    diaSrcTable = butler.get('goodSeeingDiff_diaSrc', visit=visit, detector=ccdnum, collections=collection_diff, instrument='DECam') 
    diaSrcTable_pandas = diaSrcTable.asAstropy().to_pandas()
    src_pandas = src.asAstropy().to_pandas()
    srcMatchFull = butler.get('srcMatchFull', visit=visit, detector=ccdnum, collections=collection_diff, instrument='DECam')
    srcMatchFull_pandas = srcMatchFull.asAstropy().to_pandas()

    
    src_pandas = src_pandas.rename(columns = {'id':'src_id'})
    diaSrcTable_pandas = diaSrcTable_pandas.rename(columns = {'id' : 'src_id'})

    x_pix_stars = []
    y_pix_stars = []
    for src in diaSrcTable:
        x_pix_stars.append(src.getX())
        y_pix_stars.append(src.getY())
    #new_df = pd.merge(diaSrcTable, srcMatchFull_pandas, on='src_id', how = 'outer')
    #sources = pd.merge(src_pandas, new_df, on=['src_id'], how='outer')
    x_pix_stars = np.array(x_pix_stars)
    y_pix_stars = np.array(y_pix_stars)

    pdimx = 2048 
    pdimy = 4096 


    sources = pd.merge(src_pandas, diaSrcTable_pandas, on='src_id', how='outer')
    sources = pd.merge(sources, srcMatchFull_pandas, on='src_id', how='outer')
    mask = (sources['calib_photometry_used'] == True) & (sources['ip_diffim_forced_PsfFlux_instFlux'].isnull()) & (sources['ref_id'].isnull())
    phot_table = Table.from_pandas(sources[mask])
    phot_table['coord_ra_ddegrees'] = (phot_table['coord_ra_x'] * u.rad).to(u.degree)
    phot_table['coord_dec_ddegrees'] = (phot_table['coord_dec_x'] * u.rad).to(u.degree)

    if n > len(phot_table):
        n = len(phot_table)

    print('trying with {} stars'.format(n))
    stars_table = Table()
    i = 0 
    while i < n:
        if verbose:
            print('looking at star number {}'.format(i+1))
        ra = phot_table['coord_ra_ddegrees'][i]
        dec = phot_table['coord_dec_ddegrees'][i]
        c = SkyCoord(ra * u.degree, dec * u.degree, frame='icrs')
        result = Vizier.query_region(c,
                                    radius = 2 * u.arcsec,
                                    catalog='I/355/gaiadr3',
                                    column_filters={'Gmag': '<25','Var':"!=VARIABLE"})
        # 'Gmag': '>15' took away the upper limit
        obj_pos_lsst_star = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        x_star, y_star = wcs.skyToPixel(obj_pos_lsst_star) 
        
        j, = np.where(np.fabs(np.array(x_pix_stars) - x_star) < 3)
        k, = np.where(np.fabs(np.array(y_pix_stars) - y_star) < 3)

        inter = np.intersect1d(j,k)

        if np.fabs(pdimy - y_star) <= 100 or np.fabs(pdimx - x_star) <= 100 or x_star <=100 or y_star <= 100:
            if verbose:
                print('this star is to close to the edges, discarded')
            i+=1
            continue
        
        if len(inter)>0 and well_subtracted:
            #print('near pixel xpix {} ypix {}'.format(x_pix_stars[inter], y_pix_stars[inter]))
            if verbose:
                print('a bad subtracted star is identified, discarded')
            #Calib_and_Diff_plot_cropped(repo, collection_diff, collection_diff, ra, dec, [visit], ccdnum, s=10)
            #print('x_pix {} y_pix {}'.format(x_star , y_star))
            i+=1
            continue
        if verbose:
            print('star in position ra {} dec {}'.format(ra,dec))
        #print('We are trying to add: ', result)
        try:
            #print('We are trying to add: ', result[0])
            if len(stars_table) == 0:
                if verbose:
                    print('we add first star!')
                little_table = result[0]
                little_table = little_table['RA_ICRS', 'DE_ICRS']
                stars_table = transpose_table(little_table, id_col_name='RA_ICRS')
                    
            else:
                if verbose:
                    print('adding one more star!')
                little_table = result[0]['RA_ICRS', 'DE_ICRS']
                stars_table = join(stars_table,transpose_table(little_table, id_col_name='RA_ICRS'))

            i+=1
        except:
            if verbose:
                print('not in Gaia dr3 catalog uwu...')
            i+=1
            pass
    if len(stars_table)==0:
        if verbose:
            print('No stars found meet the criteria')
        return None
    stars_table = transpose_table(stars_table, id_col_name='RA_ICRS')
    print('Found {} stars to calculate their LCs '.format(len(stars_table)))
    return stars_table


def transpose_table(tab_before, id_col_name='ID'):
    '''Returns a copy of tab_before (an astropy.Table) with rows and columns interchanged
        id_col_name: name for optional ID column corresponding to
        the column names of tab_before
        
        https://gist.github.com/PBarmby - github, tab_trans.py 
        
        '''
    # contents of the first column of the old table provide column names for the new table
    # TBD: check for duplicates in new_colnames & resolve
    new_colnames=tuple(tab_before[tab_before.colnames[0]])
    # remaining columns of old table are row IDs for new table 
    new_rownames=tab_before.colnames[1:]
    # make a new, empty table
    tab_after=Table(names=new_colnames)
    # add the columns of the old table as rows of the new table
    for r in new_rownames:
        tab_after.add_row(tab_before[r])
    if id_col_name != '':
        # add the column headers of the old table as the id column of new table
        tab_after.add_column(Column(new_rownames, name=id_col_name),index=0)
    return(tab_after)

def detector_number_to_ccd(num):
    """
    
    Input
    -----
    num : [int]
    
    Output
    -----
    ccd : [str]
    
    """
    ccd = ccd_name[num]
    return ccd
    
def ccd_to_detector_number(ccd):
    """
    
    Input:
    -----
    ccd : [str]
    
    Otput: 
    -----
    num : [int]
    """
    
    num = detector_nomenclature[ccd]
    return num


def center_brightest_zone(data, thresh, area):
    """
    
    retrieves x and y position of the brightest source 
    in the data matrix

    input:
    -----
    data
    thresh
    area

    output:
    ------
    x, y 

    """
    objects = sep.extract(data, thresh, minarea=area)
    obj = Select_largest_flux(data, objects)
    return obj['x'], obj['y']


def Select_largest_flux(data_sub, objects, na=6):
    """
    Uses source extractor to select the brightest detected source

    Input
    -----
    data_sub : [np.matrix]
    objects : 
    na :

    Output
    -----
    objects[j]
    """
    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],na*objects['a'])
    #print(flux)
    j, = np.where(flux == max(flux))
    

    return objects[j], j    

def Find_coadd(repo, collection_coadd, ra, dec, instrument='DECam', plot=False, cutout=50):
    """
    Finds coadd of corresponding source in ra dec position

    Input
    -----
    collection_coadd : [str] Name of the collection corresponding to the coadds
    ra : [float] right ascention in decimal degrees
    dec : [float] declination in decimal degrees
    
    Output
    -----
    dataId : [list of dicts] Data Id of the coadd
    """

    dataId = []
    butler = Butler(repo)
    registry = butler.registry
    for ref in registry.queryDatasets('goodSeeingCoadd', collections = collection_coadd,instrument=instrument):
        coadd = butler.get('goodSeeingCoadd', collections=collection_coadd, instrument=instrument, dataId = ref.dataId.full)
        
        x_pix, y_pix = radec_to_pixel(ra,dec,coadd.getWcs())
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        x_half_width = cutout
        y_half_width = cutout
        bbox = lsst.geom.Box2I()
        bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
        bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))
        try:
            afwDisplay.setDefaultMaskTransparency(100)
            afwDisplay.setDefaultBackend('matplotlib')

            coadd_cutout = coadd.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

            if plot:        
                fig = plt.figure(figsize=(10, 5))
                stamp_display = []

                fig.add_subplot(1,2,1)
                stamp_display.append(afwDisplay.Display(frame=fig))
                stamp_display[0].scale('linear', -1, 10)
                stamp_display[0].mtv(coadd_cutout)
                stamp_display[0].dot('o', x_pix, y_pix, ctype='#0827F5', size=5)
                plt.title('Coadd of source in ra {} dec {}'. format(ra,dec), fontsize=17)
                plt.tight_layout()
            
            dataId.append(ref.dataId.full)
            break

        except:
            pass
    return dataId



def zogy_lc(repo, collection_calexp, collection_coadd, ra, dec, ccd_num, visits, r, instrument = 'DECam', plot_diffexp=False, plot_coadd = False, cutout=50):
    """
    Does Zogy Image differencing and plots result

    
    """
    butler = Butler(repo)
    dataIds = Find_coadd(repo, collection_coadd, ra, dec, instrument=instrument, plot=plot_coadd, cutout=cutout)
    ncoadds = len(dataIds)
    print('number of coadds found: {}'.format(ncoadds))
    
    flux = []
    fluxerr = []
    flag = []
    dates = []
    arsec_to_pixel = 0.263
    radii = r/arsec_to_pixel
    for visit in visits:
        n = 0
        coadd =  butler.get('goodSeeingCoadd', collections=collection_coadd, instrument=instrument, dataId = dataIds[n])

        calexp = butler.get('calexp', visit= visit, detector= ccd_num, instrument=instrument, collections=collection_calexp)
        zogy = zt()
        #zogy.emptyMetadata()
        #matchExp = zogy.matchExposures(scienceExposure = calexp, templateExposure=coadd, doWarping=True)
        #coadd_warped = matchExp.getDict()['matchedImage']
        while(n<ncoadds):
            coadd =  butler.get('goodSeeingCoadd', collections=collection_coadd, instrument=instrument, dataId = dataIds[n])
            print('trying with coadd {}'.format(n+1))
            try:
                results = zogy.run(scienceExposure = calexp, templateExposure=coadd, doWarping=True)
                print('Success!!')
                diffexp = results.getDict()['diffExp']
                data = np.asarray(diffexp.image.array, dtype='float') 
                wcs = diffexp.getWcs()
                x_pix, y_pix = radec_to_pixel(ra,dec,wcs)
                f, ferr, flg = sep.sum_circle(data, [x_pix], [y_pix], radii, var = np.asarray(diffexp.variance.array, dtype='float'))

                print('Flux: {} Fluxerr: {}'.format(f,ferr))
                flux.append(f[0])
                fluxerr.append(ferr[0])
                flag.append(flg)
                exp_visit_info = diffexp.getInfo().getVisitInfo()
                
                visit_date_python = exp_visit_info.getDate().toPython()
                visit_date_astropy = Time(visit_date_python)
                print(visit_date_astropy)            
                dates.append(visit_date_astropy.mjd)
                break
            except:
                print('failed')
                n+=1



        if plot_diffexp:
            print('visit: {}'.format(visit))
            obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)

            cutout=50
            x_half_width = cutout
            y_half_width = cutout
            bbox = lsst.geom.Box2I()
            bbox.include(lsst.geom.Point2I(x_pix - x_half_width, y_pix - y_half_width))
            bbox.include(lsst.geom.Point2I(x_pix + x_half_width, y_pix + y_half_width))
            afwDisplay.setDefaultMaskTransparency(100)
            afwDisplay.setDefaultBackend('matplotlib')

            diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))
                            
            fig = plt.figure(figsize=(10, 5))
            stamp_display = []

            fig.add_subplot(1,2,1)
            stamp_display.append(afwDisplay.Display(frame=fig))
            stamp_display[0].scale('asinh', 'zscale')
            stamp_display[0].mtv(diffexp_cutout)
            stamp_display[0].dot('o', x_pix, y_pix, ctype='magenta', size=radii)
            plt.title('Difference Image of source in ra {} dec {}'.format(ra,dec))
            plt.tight_layout()
            plt.show()
    mean = np.mean(flux)
    source_of_interest = pd.DataFrame()
    source_of_interest['dates'] = dates - min(dates)
    source_of_interest['flux'] = flux
    source_of_interest['flux_err'] = fluxerr
    source_of_interest = source_of_interest.sort_values(by='dates')
    #plt.errorbar(source_of_interest.dates, source_of_interest.flux, yerr=source_of_interest.flux_err, capsize=4, fmt='s', label ='Cáceres-Burgos', color='#0827F5', ls ='dotted')
    #plt.ylabel('Excess Flux in arbitrary units', fontsize=15 )
    #plt.xlabel('MJD', fontsize=15)
    return source_of_interest

def checksMagAtOneInstFlux(repo, dataType, visit, collection, detector, instrument='DECam'):
    """
    Returns the AB magnitude that equates one count for the Instrumental Flux (ADU)

    Input:
    -----
    repo : [str]
    dataType : [str]
    visit : [int]
    collection : [str]
    detector : [int]
    instrument: [str]

    Ouput:
    -----
    mag : [float]
    """
    butler = Butler(repo)
    data = butler.get(dataType,visit=visit, detector=detector, collections=collection, instrument=instrument)
    p = data.getPhotoCalib()
    mag = p.instFluxToMagnitude(1)
    return mag

def compare_to(path, sfx, factor, beforeDate=57072):
    '''
    Returns Jorge Martinez-Palomera or Francisco Forsters code
    
    Input
    -----
    directory :
    sfx :
    factor :
    beforeDate :
    
    Output
    ------
    x, y, yerr
    '''

    SIBLING = path
    if SIBLING[0:24]=="/home/pcaceres/Jorge_LCs":
        Jorge_LC = pd.read_csv(SIBLING, header=5)
        Jorge_LC = Jorge_LC[Jorge_LC['mjd']<beforeDate] 
        
        sfx_aux = 'mag'
        
        scaled_apertures = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        k, = np.where(scaled_apertures == factor)
        
        y_value = Jorge_LC['aperture_{}_{}'.format(sfx, k[0])]
        yerr_value = Jorge_LC['aperture_{}_err_{}'.format(sfx, k[0])]
        x_value = Jorge_LC.mjd
        return x_value, y_value, yerr_value

    HiTS = directory 

    if HiTS[0:24]=="/home/pcaceres/HiTS_LCs/":
        HiTS_LC = pd.read_csv(HiTS, skiprows = 2, delimiter=' ')
        HiTS_LC = HiTS_LC.dropna()
        HiTS_LC = HiTS_LC[(HiTS_LC['MJD']<beforeDate) & (HiTS_LC['band']=='g')]
        x = HiTS_LC.MJD

        if sfx == 'flx':
            y = HiTS_LC.ADU
            yerr = HiTS_LC.e_ADU
            return x, y, yerr

        if sfx == 'mag':
            y = HiTS_LC.mag
            yerr = HiTS_LC.e1_mag
            return x, y, yerr

    else:
        None

def plot_Jorges_LCs(sibling_allcand, campaign = 'Blind15A', sfx='flx'):
    '''
    Plotting Jorges LCs 

    '''
    internalID = list(sibling_allcand.internalID)
    fields_sibling = np.unique([i[:11] for i in internalID if type(i)==str and i[:8] == campaign])
    
    for field in fields_sibling:
        cands = Find_sources(sibling_allcand, field)
        index = cands.index
        ccds = [f.split('_')[2] for f in cands.internalID]
        all_ccds_Jorge(field,ccds,campaign, sfx=sfx)
    return


def Truncate(num, decim):
    '''
    Truncates number to a desired decimal
    
    Input:
    -----
    - num : [float]
    - decimal : [int]
    
    Output:
    --------
    - trunc_num : [float] truncated number 
    
    '''
    d = 10**(-decim)
    trunc_num = float(decimal.Decimal(num).quantize(decimal.Decimal('{}'.format(d)), rounding=decimal.ROUND_DOWN))
    return trunc_num


def Excess_variance(mag, magerr):
    """
    Calculates excess variance as defined by Sanchez et al. 2017

    Input
    -----
    mag
    magerr

    Output
    ----
    sigma_rms_sq, sigma_rms_sq_err, sigma_rms_sq - sigma_rms_sq_err

    """
    mean_mag = np.mean(mag)
    nobs = float(len(mag))
    a = (mag - mean_mag)**2
    sigma_rms_sq = np.sum(a - magerr**2) / (nobs * mean_mag**2)
    sd = 1/nobs * np.sum(((a - magerr**2) - sigma_rms_sq * mean_mag**2 )**2)
    sigma_rms_sq_err = sd / (mean_mag**2 * nobs**(0.5))

    return sigma_rms_sq, sigma_rms_sq_err, sigma_rms_sq - sigma_rms_sq_err


def flux_profile(exposure, ra, dec , rmin, rmax, title_plot = '', save_plot =False, field=None, name =None):
    """
    Returns an array of the values across a rectangular slit of a source,
    that is wider in the x-axis

    input:
    -----
    exposure : 
    ra : [float] right ascention in degrees
    dec : [float] declination in degrees
    rmin : [float] minimum radius in arcsec
    rmax : [float] maximum radius in arcsec 
    title_plot : [string]
    save_plot : [bool]
    field : [string]
    name  : [string]

    output 
    ------
    fluxes_ap : [list]
    """
    wcs = exposure.getWcs()
    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
    #exp_photocalib = exposure.getPhotoCalib()
    exp_array = np.asarray(exposure.image.array, dtype='float')
    #obj_pos_2d = lsst.geom.Point2D(ra, dec)
    
    fluxes_ap = []
    fluxes_ap_err = []
    apertures = np.linspace(rmin, rmax, 15)
    arcsec_to_pixel = 0.27

    for r in apertures:
        r/=arcsec_to_pixel
        f, ferr, flag = sep.sum_circle(exp_array, [x_pix], [y_pix], r, var = np.asarray(exposure.variance.array, dtype='float'))
        fluxes_ap.append(f[0])
        fluxes_ap_err.append(ferr[0])

    #fluxes = [exp_photocalib.instFluxToNanojansky(f, obj_pos_2d) for f in adu_values]
    #ai, aip, bi, bip = special.airy(fluxes)
    fluxes_ap /= fluxes_ap[-1]
    fluxes_ap_err /= fluxes_ap[-1] #sum(fluxes_ap)

    #plt.figure(figsize=(10,6))
    #plt.plot(apertures, fluxes_ap, '*', color='magenta')
    #plt.xlabel('arcsec aperture')
    #plt.ylabel('Normalized flux counts')

    if save_plot:
        #f.savefig('light_curves/{}/{}.jpeg'.format(field, name), bbox_inches='tight')
        pass
    #plt.show()

    return fluxes_ap


def flux_profile_array(exp_array, x_pix, y_pix, rmin, rmax, naps = 15):
    """
    Returns an array of the values across a rectangular slit of a source,
    that is wider in the x-axis (growth curve)

    input:
    -----
    exp_array [np.matrix] : Image array  
    ra : [float] right ascention in degrees
    dec : [float] declination in degrees
    rmin : [float] minimum radius in arcsec
    rmax : [float] maximum radius in arcsec 
    naps : [int] number of apertures between rmin and rmax 
            which we will do photometry 
    
    output 
    ------
    f : [list] : list of fluxes 
    """    
    arcsec_to_pixel = 0.27
    apertures = np.linspace(rmin, rmax, naps)/arcsec_to_pixel # Here apertures are in pixels
    
    f, ferr, flag = sep.sum_circle(exp_array, [x_pix], [y_pix], apertures)
        
    return f.flatten()
