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

sys.path.append('/home/pcaceres/kkernel/lib/')
sys.path.append('/home/pcaceres/kkernel/etc/')

from kkernel import *
#from sklearn.preprocessing import Normalize


bblue='#0827F5'
dark_purple = '#2B018E'
lilac='#a37ed4'
neon_green = '#00FF00'

main_root = '/home/pcaceres'
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
        arcsec_to_pixel = 0.2626 #arcsec/pixel
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


def Calib_Diff_and_Coadd_plot_cropped_astropy(repo, collection_diff, ra, dec, visits, ccd_num, cutout=40, s=20, sd=5, field='', name=''):
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
        diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout_diff*2, cutout_diff*2))
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
        plt.imshow(np.asarray(calexp.image.array,dtype='float'), cmap='rocket', origin='lower', vmin = 0, vmax=np.max(calexp_cutout_forPeak.flatten()))
        plt.colorbar()
        #levels=np.logspace(1.3, 2.5, 10),
        plt.contour(np.asarray(calexp.image.array,dtype='float'),  colors ='white', alpha=0.5)
        circle = plt.Circle((x_pix,y_pix), radius = s, color='red', fill = False, linewidth=4)
        plt.gca().add_patch(circle)
        circle2 = plt.Circle((x_pix,y_pix), radius = sd, color=neon_green, fill = False, linewidth=4)
        plt.scatter(x_pix,y_pix, color=neon_green, marker='x', linewidth=3)
        plt.xlim(x_pix - cutout, x_pix + cutout)
        plt.ylim(y_pix - cutout, y_pix + cutout)
        plt.gca().add_patch(circle2)

        plt.title('Reduced Image', fontsize=15)

        fig.add_subplot(1,3,2)
        plt.imshow(np.asarray(coadd.image.array,dtype='float'), cmap='rocket', origin='lower', vmin = 0 , vmax = np.max(coadd_cutout_forPeak.flatten()))
        plt.colorbar()
        #levels=np.logspace(1.3, 2.5, 10),
        plt.contour(np.asarray(coadd.image.array,dtype='float'),  colors ='white', alpha=0.5)
        plt.scatter(x_pix, y_pix, color=neon_green, marker='x', linewidth=3)
        #plt.scatter(x_half_width, y_half_width, s=np.pi*s**2, facecolors='none', edgecolors='red')
        circle = plt.Circle((x_pix, y_pix), radius = s, color='red', fill = False, linewidth=4)
        plt.gca().add_patch(circle)
        circle2 = plt.Circle((x_pix, y_pix), radius = sd, color=neon_green, fill = False, linewidth=4)
        plt.gca().add_patch(circle2)
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
        circle = plt.Circle((x_pix, y_pix), radius = sd, color=neon_green, fill = False, linewidth=4)
        plt.gca().add_patch(circle)
        plt.title('Difference Image', fontsize=15)
        plt.xlim(x_pix - cutout_diff, x_pix + cutout_diff)
        plt.ylim(y_pix - cutout_diff, y_pix + cutout_diff)

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
    n_rows = 5
    n_cols = len(visits)

    # Create the figure and subplots using gridspec
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, 5*2), facecolor='k')
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

    plt.savefig('{}_forpaper_stamps.jpeg'.format(name_to_save), dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), facecolor='k')
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
    #ax1 = plt.subplot(gs[0, :])
    #ax1.plot(Results_galaxy.dates - min(Results_galaxy.dates), seeing, color='k')
    #ax1.set_ylabel('Seeing sigma [pixels]', fontsize=12)

    ax2 = plt.subplot(gs[0, :])
    ax2.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.flux1_nJy, yerr = Results_galaxy.fluxerr1_nJy, capsize=4, fmt='s', color='w', ls ='dotted', label = 'diff image radii = {}"'.format(r_diff)) # , label ='Fixed aperture of {}" * 1'.format(r_diff)
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
        ax2.errorbar(x-min(x), f -  np.mean(f), yerr=ferr,  capsize=4, fmt='^', ecolor='orange', color='orange', label='Martinez-Palomera et al. 2020', ls ='dotted')
    
    ax2.legend(frameon=False, ncol=2, fontsize=12)

    ax3 =  plt.subplot(gs[1, :], sharex=ax2)

    ax3.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), Results_star.stars_science_1sigmalow_byEpoch,  Results_star.stars_science_1sigmaupp_byEpoch, alpha=0.3, color='blue', label = 'convolved stars 1-2 $\sigma$ dev') #
    ax3.fill_between(Results_galaxy.dates - min(Results_galaxy.dates), 2*Results_star.stars_science_1sigmalow_byEpoch,  2*Results_star.stars_science_1sigmaupp_byEpoch, alpha=0.3, color='blue') #, label = 'stars 2-$\sigma$ dev'
    ax3.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.fluxConv_cal_nJy - np.mean(Results_galaxy.fluxConv_cal_nJy), yerr=Results_galaxy.fluxerrConv_cal_nJy, capsize=4, fmt='s', label='science after conv radii = {}"'.format(r_science), color='m', ls='dotted')
    
    if SIBLING!=None:
        x, y, yerr = compare_to(SIBLING, sfx='mag', factor=0.75)
        f, ferr = pc.ABMagToFlux(y, yerr)# in nJy
        #mfactor = 5e-10
        ax3.errorbar(x-min(x), f -  np.mean(f), yerr=ferr,  capsize=4, fmt='^', ecolor='orange', color='orange', label='Martinez-Palomera et al. 2020', ls ='dotted')
    
    #ax6.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.flux_nJy_cal - np.mean(Results_galaxy.flux_nJy_cal), yerr = Results_galaxy.fluxerr_nJy_cal, capsize=4, fmt='d', label ='science before conv radii = {}'.format(r_diff), color='green', ls ='dotted')
    #ax3.set_ylim(-2500,2500)
    ax3.set_ylabel('Flux [nJy] - Median', fontsize=18)
    ax3.set_xlabel('MJD - {}'.format(first_mjd), fontsize=18)
    ax3.legend(frameon=False, ncol=2, fontsize=12)
    plt.subplots_adjust(hspace=0, wspace=0.2)
    plt.savefig('{}_forpaper_LCs.jpeg'.format(name_to_save), dpi=300, bbox_inches='tight')
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
    ax2.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.flux1_nJy, yerr = Results_galaxy.fluxerr1_nJy, capsize=4, fmt='s', color='k', ls ='dotted', label = 'diff image radii = {}"'.format(r_diff)) # , label ='Fixed aperture of {}" * 1'.format(r_diff)
    ax2.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.fluxConv_cal_nJy - np.mean(Results_galaxy.fluxConv_cal_nJy), yerr=Results_galaxy.fluxerrConv_cal_nJy, capsize=4, fmt='s', label='science after conv radii = {}"'.format(r_science), color='m', ls='dotted')
      
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

    ax3.errorbar(Results_galaxy.dates - min(Results_galaxy.dates), Results_galaxy.fluxConv_cal_nJy - np.mean(Results_galaxy.fluxConv_cal_nJy), yerr=Results_galaxy.fluxerrConv_cal_nJy, capsize=4, fmt='s', label='science after conv radii = {}"'.format(r_science), color='m', ls='dotted')
    
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
    Returns visits and dates by the order of the latter.


    '''
    butler = Butler(repo)
    dates = []
    for i in range(len(visits)):
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        calexp = butler.get('calexp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 
        

        exp_visit_info = diffexp.getInfo().getVisitInfo()
        
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)        
        dates.append(visit_date_astropy.mjd)

    dates = dates #- min(dates)
    zipped = zip(dates, visits)
    res = sorted(zipped, key = lambda x: x[0])

    dates_aux, visits_aux = zip(*list(res))

    return dates_aux, visits_aux

def Find_worst_seeing(repo, visits, ccd_num, collection_calexp, arcsec_to_pixel = 0.2626):
    '''
    Returns

    input
    ------
    repo
    visits
    ccd_num
    collection_calexp

    output
    -------
    min(Seeing)
    '''
    butler = Butler(repo)
    Seeing = []
    for i in range(len(visits)):
        #diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        calexp = butler.get('calexp',visit=visits[i], detector=ccd_num , collections=collection_calexp, instrument='DECam') 
        psf = calexp.getPsf() 
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))

        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()#*sigma2fwhm #* arcsec_to_pixel # to arcseconds 
        Seeing.append(seeing)
    print('Worst Seeing in pixels: ', max(Seeing))
    j, = np.where(Seeing == np.max(Seeing))
    worst_visit = visits[j[0]]
    return np.max(Seeing), worst_visit



def get_light_curve(repo, visits, collection_diff, collection_calexp, ccd_num, ra, dec, r_science, r_diff, field='', factor=0.75, cutout=40, save=False, title='', hist=False, sparse_obs=False, SIBLING=None, save_as='', do_lc_stars = False, nstars=10, seedstars=200, save_lc_stars = False, show_stamps=True, show_star_stamps=True, r_star = 6, correct_coord=False, bs=531, box=100, do_zogy=False, collection_coadd=None, plot_zogy_stamps=False, plot_coadd=False, instrument='DECam', sfx='flx', save_stamps=False, well_subtracted=False, verbose=False, tp='after_ID', area=None, thresh=None, mfactor=1, do_convolution=True, mode='Eridanus', name_to_save='', type_kernel = 'mine'):
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
    factor : [float] values can be: [0.5, 0.75, 1., 1.25, 1.5], these are the values that are multiplied to the seeing
            of Jorge's forced photometry
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


    Data_science = {}
    Data_convol = {}
    Data_coadd = {}
    Data_diff = {}
    
    coords_coadd = {}
    coords_science = {}
    coords_convol = {}

    profiles = {}

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
    #new_err = []
    
    magzero = []

    flux_coadd=0
    fluxerr_coadd=0
    #fluxes_cal = []
    #fluxes_err_cal = []
    Mag_coadd = []
    Magerr_coadd = []
    Airmass = []

    Fluxes_njsky = []
    Fluxeserr_njsky = []


    Fluxes0p5_njsky = []
    Fluxeserr0p5_njsky = []

    Fluxes0p75_njsky = []
    Fluxeserr0p75_njsky = []

    Fluxes1_njsky = []
    Fluxeserr1_njsky = []

    Fluxes1p25_njsky = []
    Fluxeserr1p25_njsky = []

    Fluxes1p5_njsky = []
    Fluxeserr1p5_njsky = []

    Fluxes0p75dyn_njsky = []
    Fluxeserr0p75dyn_njsky = []


    FluxesFs_njsky = []
    FluxeserrFs_njsky = []

    Fluxes_njsky_coadd = []
    Fluxeserr_njsky_coadd = []

    FluxConv_njsky = []
    FluxerrConv_njsky = []
    
    stars = pd.DataFrame()
    
    flags = []
    
    stats = {}
    #arcsec_to_pixel = 0.2626 #arcsec/pixel, value from Manual of NOAO - DECam
    arcsec_to_pixel = 0.2626
    r_in_arcsec = r_diff 
    #if type(r) != str:
    #    r_aux = r/arcsec_to_pixel

    flux_reference = 0
    calib_reference = 0 
    #fluxerr_reference = 0
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

    dates_aux, visits_aux = Order_Visits_by_Date(repo, visits, ccd_num, collection_diff)
    worst_seeing, worst_seeing_visit = Find_worst_seeing(repo, visits, ccd_num, collection_diff) # sigma in pixels
    worst_cal = butler.get('calexp', visit=worst_seeing_visit, detector=ccd_num , collections=collection_diff, instrument='DECam')
    #worst_wcs = worst_cal.getWcs()
    worst_psf = worst_cal.getPsf() 
    sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
    #fixed_radii = worst_seeing/arcsec_to_pixel/sigma2fwhm * 0.75 # in pixels 
    fixed_radii = r_science/arcsec_to_pixel # in pixels 
    
    print('fixed radii: ',fixed_radii)

    RA_source = []
    DEC_source = []
    print('worst seeing visit: ', worst_seeing_visit)
    distance_me_lsst = []
    kernel_stddev = []
    for i in range(len(visits_aux)):
        skip_observation = False
        zero_set_aux = zero_set
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        calexp = butler.get('calexp', visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 
        try:
            coadd = butler.get('goodSeeingDiff_matchedExp',visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        except:
            coadd = butler.get('goodSeeingDiff_templateExp',visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam')

        calexpbkg = butler.get('calexpBackground', visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 

        wcs_coadd = coadd.getWcs()
        wcs_cal = calexp.getWcs()
        wcs = diffexp.getWcs()

        px = 2048
        py = 4096

        data = np.asarray(diffexp.image.array, dtype='float')
        Data_diff[visits_aux[i]] = data
        
        data_cal = np.asarray(calexp.image.array, dtype='float')
        Data_science[visits_aux[i]] = data_cal

        TOTAL_counts = np.sum(np.sum(data_cal))

        data_coadd = np.asarray(coadd.image.array, dtype='float')
        Data_coadd[visits_aux[i]] = data_coadd

        data_cal_bkg = np.asarray(calexpbkg.getImage().array,dtype='float')
        bkg_rms = np.sqrt(np.mean((data_cal_bkg.flatten()- np.mean(data_cal_bkg.flatten()))**2))

        
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        coords_science[visits_aux[i]] = [x_pix, y_pix]

        x_pix_coadd, y_pix_coadd = wcs_coadd.skyToPixel(obj_pos_lsst)
        coords_coadd[visits_aux[i]] = [x_pix, y_pix]


        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        psf = diffexp.getPsf() 
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()#*sigma2fwhm # in pixels! 

        psf_calexp = calexp.getPsf() 
        seeing_calexp = psf_calexp.computeShape(psf_calexp.getAveragePosition()).getDeterminantRadius()#*sigma2fwhm

        rs_aux = r_science/arcsec_to_pixel
        rd_aux = r_diff/arcsec_to_pixel 
        rd_dyn = 2*seeing*0.75
        Seeing.append(seeing)
        calConv_image = 0 
        
        meanValue_plot = 0 
        stdValue_plot = 0 

        if correct_coord:

            print('before centering correction: xpix, y_pix: {} {} '.format(x_pix, y_pix))

            cut_aux = 10
            cut_aux2 = 10 # in pixels 
            minarea = (1/arcsec_to_pixel)**2
            
            if mode == 'Eridanus':
                cut_aux = 10 #cutout
                cut_aux2 = cutout
                minarea = (1/arcsec_to_pixel)**2
    
            sub_data = data_cal[int(y_pix-cut_aux2):int(cut_aux2+y_pix),int(x_pix-cut_aux2):int(cut_aux2+x_pix)].copy(order='C')
            sub_data_toplot = data_cal[int(y_pix-cut_aux):int(cut_aux+y_pix),int(x_pix-cut_aux):int(cut_aux+x_pix)].copy(order='C')
            m, s = np.mean(sub_data_toplot), np.std(sub_data_toplot)
            sepThresh = m + 2*s
            peak_value = m + 3*s

            objects = sep.extract(sub_data, sepThresh, minarea=minarea)
            print('value thresh: ', sepThresh)

            obj, j = Select_largest_flux(sub_data, objects)
            x_pix_aux = obj['x']
            y_pix_aux = obj['y']

            x_pix_OgImage = x_pix_aux + x_pix - cut_aux2
            y_pix_OgImage = y_pix_aux + y_pix - cut_aux2

            if len(j)>1:

                dis_aux = ((x_pix_OgImage - x_pix)**2 + (y_pix_OgImage - y_pix)**2)**(1/2)
                try:
                    l, = np.where(dis_aux == np.min(dis_aux))

                    x_pix_OgImage = np.array(x_pix_OgImage)[l[0]]
                    y_pix_OgImage = np.array(y_pix_OgImage)[l[0]]
                except:
                    print('Didnt found a detected source from the pipeline, I skip this observation')

                    skip_observation = True
                    continue

            print('New pixels after centering with SEP: xpix = {}, y_pix = {}'.format(x_pix_OgImage, y_pix_OgImage))
            ra, dec = wcs_cal.pixelToSkyArray([x_pix_OgImage], [y_pix_OgImage], degrees=True)
            obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
            obj_pos_2d = lsst.geom.Point2D(ra, dec)
            x_pix, y_pix = x_pix_OgImage, y_pix_OgImage
            coords_science[visits_aux[i]] = [x_pix, y_pix]
            if show_stamps:
                fig, ax = plt.subplots()
                m, s = np.mean(sub_data_toplot), np.std(sub_data_toplot)
                meanValue_plot = m 
                stdValue_plot = s
                plt.title('Corrected coord',  fontsize=17)
                plt.imshow(data_cal, cmap='rocket', vmin=m-s, vmax=m + 4*s)
                plt.colorbar()
                plt.scatter(x_pix,y_pix, marker = 'x', color=neon_green, label = 'largest flux sep')
                circle = plt.Circle((x_pix,y_pix), radius = rd_aux, color=neon_green, fill = False, linewidth=4)
                plt.gca().add_patch(circle)
                plt.xlim(x_pix - cut_aux, x_pix + cut_aux)
                plt.ylim(y_pix - cut_aux, y_pix + cut_aux)
                plt.legend()
                plt.show()

        ra_center, dec_center = wcs.pixelToSkyArray([px/2], [py/2], degrees=True)
        ra_center = ra_center[0]
        dec_center = dec_center[0]
        #print('should be center of exposre')
        #print('RA center : {} DEC center : {}'.format(ra_center, dec_center))

        ExpTime = diffexp.getInfo().getVisitInfo().exposureTime 
        ExpTimes.append(ExpTime)
        #gain = 4

        if skip_observation:
            continue

        if not correct_coord:
            obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
            obj_pos_2d = lsst.geom.Point2D(ra, dec)
            x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)

        print('xpix, y_pix: {} {} '.format(x_pix, y_pix))

        if do_convolution:

            print('worst_seeing: ', worst_seeing)
            cut_aux = 10
            sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
            
            calConv, stddev_kernel = do_convolution_image(repo, main_root, visits_aux[i], ccd_num, collection_calexp, ra, dec, worst_seeing_visit =  worst_seeing_visit, mode=mode, type_kernel = type_kernel)
            calConv_image = calConv[visits_aux[i]]
            calConv_variance = calConv['{}_variance'.format(visits_aux[i])]
            TOTAL_convolved_counts = np.sum(np.sum(calConv_image))
            print('fraction of Flux lost after convolution: ',1-TOTAL_convolved_counts/TOTAL_counts)

            KERNEL[visits_aux[i]] = calConv['{}_kernel'.format(visits_aux[i])]
            Data_convol[visits_aux[i]] = calConv_image
            print('stddev_kernel: ', stddev_kernel)
            detectionTask = SourceDetectionTask()
            kernel_stddev.append(stddev_kernel)
            dataconv_cut = calConv_image[int(y_pix-cut_aux): int(y_pix+cut_aux),int(x_pix-cut_aux): int(x_pix+cut_aux)].copy(order='C')
            peak_value = np.max(np.array(dataconv_cut).flatten())

            x_pix_conv = x_pix
            y_pix_conv = y_pix

            if len(j)>1:
                dis_aux = ((x_pix_conv - x_pix)**2 + (y_pix_conv - y_pix)**2)**(1/2)
                try:
                    l, = np.where(dis_aux == np.min(dis_aux))

                    x_pix_conv = np.array(x_pix_conv)[l[0]]
                    y_pix_conv = np.array(y_pix_conv)[l[0]]
                except:
                    print('Skipping this observation')
                    continue
            coords_convol[visits_aux[i]] = [x_pix_conv, y_pix_conv]#[x_pix_conv, y_pix_conv]
            
            if show_stamps:
                #m, s = np.mean(dataconv_cut.flatten()), np.std(dataconv_cut.flatten())

                #norm = mcolors.LogNorm(vmin=m-s, vmax=m + s)
                plt.title('Convolution downgrade to the worst seeing', fontsize=15)
                plt.imshow(calConv_image, cmap='rocket', vmin=meanValue_plot-stdValue_plot, vmax=meanValue_plot+ 4 * stdValue_plot)
                plt.colorbar()
                #plt.scatter(x_pix_conv, y_pix_conv, marker='x', color=neon_green, label='centroid for Convolve image')                
                plt.xlim(x_pix_conv - cut_aux, x_pix_conv + cut_aux)
                plt.ylim(y_pix_conv - cut_aux, y_pix_conv + cut_aux)
                circle = plt.Circle((x_pix,y_pix), radius = rd_aux, color=neon_green, fill = False, linewidth=4)
                plt.gca().add_patch(circle)
                if correct_coord:
                    plt.scatter(x_pix, y_pix, marker='x', color='blue', label = 'Corrected centroid in science image')
                else:
                    plt.scatter(x_pix, y_pix, marker='x', color='blue', label= ' Position for science image')
                plt.legend()
                plt.show()
        # Histogram of random part of image
        if hist:
            data_onedim = np.ndarray.flatten(data)
            #data_for_hist.append(data_onedim)
            #plt.hist(data_onedim, bins=500, histtype='step', log=True, color='#7A68A6')
            #plt.title('Diffexp histogram', fontsize=15)
            stats['{}'.format(visits_aux[i])] = data_onedim
            plt.show()

        ##############
        
        exp_visit_info = diffexp.getInfo().getVisitInfo()
        
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)
        print(visit_date_astropy)            
        b = np.nan_to_num(np.array(data))
        

        print('Aperture radii: {} px'.format(rd_aux))
        #print('radii for the template: {} px'.format(seeing2*factor))

        # Doing photometry step 

        photocalib = diffexp.getPhotoCalib()
        photocalib_cal = calexp.getPhotoCalib()
        photocalib_coadd = coadd.getPhotoCalib()

        calib_image = photocalib_cal.getCalibrationMean()
        calib_image_err = photocalib_cal.getCalibrationErr()
        calib_lsst.append(calib_image)
        calib_lsst_err.append(calib_image_err)

        #print('Calibration Mean of Difference: ', photocalib.getCalibrationMean())
        #print('Calibration Mean of Science: ', photocalib_cal.getCalibrationMean())
        #print('Calibration Mean of Template: ', photocalib_coadd.getCalibrationMean())

        #r_aux*=calib_image

        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], rd_aux, var = np.asarray(diffexp.variance.array, dtype='float')) # fixed aperture 
        flux_Fs, fluxerr_Fs, flag_FS = sep.sum_circle(data, [x_pix], [y_pix], factor*seeing, var = np.asarray(diffexp.variance.array, dtype='float'))
        
        # aperture photometry of 5 fixed radii (like Jorges)
        flux_Fs0p5, fluxerr_Fs0p5, flag_FS0p5 = sep.sum_circle(data, [x_pix], [y_pix], 0.5*rd_aux, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_Fs0p75, fluxerr_Fs0p75, flag_FS0p75 = sep.sum_circle(data, [x_pix], [y_pix], 0.75*rd_aux, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_Fs1, fluxerr_Fs1, flag_FS1 = sep.sum_circle(data, [x_pix], [y_pix], 1*rd_aux, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_Fs1p25, fluxerr_Fs1p25, flag_FS1p25 = sep.sum_circle(data, [x_pix], [y_pix], 1.25*rd_aux, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_Fs1p5, fluxerr_Fs1p5, flag_FS1p5 = sep.sum_circle(data, [x_pix], [y_pix], 1.5*rd_aux, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_Dyn0p75, fluxerr_Dyn0p75, flag_DYN0p75 = sep.sum_circle(data, [x_pix], [y_pix], rd_dyn, var= np.asarray(diffexp.variance.array, dtype='float'))
        
        #flux_an, fluxerr_an, flag_an = sep.sum_circann(data, [x_pix], [y_pix], r_aux*2, r_aux*5, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_cal, fluxerr_cal, flag_cal = sep.sum_circle(data_cal, [x_pix], [y_pix], fixed_radii, var = np.asarray(calexp.variance.array, dtype='float'))
        
        if do_convolution:
            #x_pix_conv, y_pix_conv = x_pix, y_pix

            flux_conv, fluxerr_conv, flux_convFlag = sep.sum_circle(calConv_image, [x_pix_conv], [y_pix_conv], rd_aux, var=calConv_variance)
            #flux_calConv, fluxerr_calConv, flag_calConv = sep.sum_circle(calConv_image, [x_pix], [y_pix], worst_seeing/sigma2fwhm, var = np.asarray(calexp.variance.array, dtype='float'), gain=4)
        
        flux_coadd, fluxerr_coadd, flag_coadd = sep.sum_circle(data_coadd, [x_pix], [y_pix], rs_aux, var = np.asarray(coadd.variance.array, dtype='float'))
        #flux_coadd_an, fluxerr_coadd_an, flag_coadd = sep.sum_circann(data_coadd, [x_pix], [y_pix], r_aux*2, r_aux*5, var = np.asarray(coadd.variance.array, dtype='float'))       
        
        print('Coords: ra = {}, dec = {}'.format(ra,dec))
        print('visit : {}'.format(visits_aux[i]))
        
        if show_stamps:
            print('DATE: ', dates_aux[i])
            #Calib_and_Diff_plot_cropped(repo, collection_diff, collection_calexp, ra, dec, [visits_aux[i]], ccd_num, s=r)
            print('aperture that enters stamp plots: ', rs_aux)
            Calib_Diff_and_Coadd_plot_cropped_astropy(repo, collection_diff, ra, dec, [visits_aux[i]], ccd_num, s=rs_aux, sd=rd_aux, cutout=40, field=field, name=title)
            #values_across_source(calexp, ra, dec , x_length = r_aux, y_length=1.5, stat='median', title_plot='Calibrated exposure', save_plot =True, field=field, name='slit_science_{}_{}'.format(save_as, sfx))
            #values_across_source(diffexp, ra, dec , x_length = r_aux, y_length=1.5, stat='median', title_plot = 'Difference exposure', save_plot = True, field=field, name='slit_difference_{}_{}'.format(save_as, sfx))
            
            
        #prof = flux_profile(calexp, ra, dec, 0.05, r_science)
        prof = flux_profile_array(calConv_image, x_pix_conv, y_pix_conv, 0.05, 6)
        
        profiles['{}'.format(visits_aux[i])] = prof
            
       

        

        expTime = float(calexp.getInfo().getVisitInfo().exposureTime)
        print('exposure Time: ', expTime)

        print('calibration mean: ', calib_image)

        flux_physical = photocalib.instFluxToNanojansky(flux[0], fluxerr[0], obj_pos_2d)
        print('Flux and Flux error in nanoJansky: ',flux_physical)
    
        flux_jsky = flux_physical.value
        fluxerr_jsky = flux_physical.error#flux_jsky * np.sqrt((calib_image/calib_image_err)**2 + (flux[0]/fluxerr[0])**2)

        fluxFs_physical = photocalib.instFluxToNanojansky(flux_Fs[0], fluxerr_Fs[0], obj_pos_2d)
        print('Flux and Flux error in nanoJansky: ',fluxFs_physical)
        
        fluxFs_jsky = fluxFs_physical.value
        fluxerrFs_jsky = fluxFs_physical.error#flux_jsky * np.sqrt((calib_image/calib_image_err)**2 + (flux[0]/fluxerr[0])**2)

        ## njsky conversion for the 5 fixed apertures:

        fluxFs0p5_physical = photocalib.instFluxToNanojansky(flux_Fs0p5[0], fluxerr_Fs0p5[0], obj_pos_2d)        
        fluxFs0p5_jsky = fluxFs0p5_physical.value
        fluxerrFs0p5_jsky = fluxFs0p5_physical.error

        fluxFs0p75_physical = photocalib.instFluxToNanojansky(flux_Fs0p75[0], fluxerr_Fs0p75[0], obj_pos_2d)        
        fluxFs0p75_jsky = fluxFs0p75_physical.value
        fluxerrFs0p75_jsky = fluxFs0p75_physical.error

        fluxFs1_physical = photocalib.instFluxToNanojansky(flux_Fs1[0], fluxerr_Fs[0], obj_pos_2d)        
        fluxFs1_jsky = fluxFs1_physical.value
        fluxerrFs1_jsky = fluxFs1_physical.error

        fluxFs1p25_physical = photocalib.instFluxToNanojansky(flux_Fs1p25[0], fluxerr_Fs1p25[0], obj_pos_2d)        
        fluxFs1p25_jsky = fluxFs1p25_physical.value
        fluxerrFs1p25_jsky = fluxFs1p25_physical.error

        fluxFs1p5_physical = photocalib.instFluxToNanojansky(flux_Fs1p5[0], fluxerr_Fs1p5[0], obj_pos_2d)        
        fluxFs1p5_jsky = fluxFs1p5_physical.value
        fluxerrFs1p5_jsky = fluxFs1p5_physical.error

        fluxDyn0p75_physical = photocalib.instFluxToNanojansky(flux_Dyn0p75[0], fluxerr_Dyn0p75[0], obj_pos_2d)
        fluxDyn0p75_jsky = fluxDyn0p75_physical.value
        fluxerrDyn0p75_jsky = fluxDyn0p75_physical.error
        
        if do_convolution:
            print('flux_conv: {} [ADU], fluxerr_conv: {} [ADU]'.format(flux_conv[0], fluxerr_conv[0]))
            print('photocalib cal scaling factor: ', photocalib_cal.getCalibrationMean())
            fluxConv_physical = photocalib_cal.instFluxToNanojansky(flux_conv[0], fluxerr_conv[0], obj_pos_2d)

            fluxConv_jsky = fluxConv_physical.value
            fluxerrConv_jsky = fluxConv_physical.error
            print('flux_conv: {} [nJy], fluxerr_conv: {} [nJy]'.format(fluxConv_jsky, fluxerrConv_jsky))


        #####



        flux_physical_coadd = photocalib_coadd.instFluxToNanojansky(flux_coadd[0], fluxerr_coadd[0], obj_pos_2d)

        flux_jsky_coadd = flux_physical_coadd.value 
        fluxerr_jsky_coadd = flux_physical_coadd.error #flux_jsky_coadd * np.sqrt((calib_image/calib_image_err)**2 + (flux_coadd[0]/fluxerr_coadd[0])**2) # good error

        magzero.append(flux_jsky_coadd/flux_coadd[0])
        
        flux_physical_cal = photocalib_cal.instFluxToNanojansky(flux_cal[0], fluxerr_cal[0], obj_pos_2d)
        flux_jsky_cal = flux_physical_cal.value 
        fluxerr_jsky_cal = flux_physical_cal.error #flux_jsky_cal * np.sqrt((calib_image/calib_image_err)**2 + (flux_cal[0]/fluxerr_cal[0])**2)


        f = (flux_jsky + flux_jsky_coadd)*1e-9 
        f_err = float(np.sqrt((fluxerr_jsky*1e-9)**2 + (fluxerr_jsky_coadd*1e-9)**2))

        print('total flux : {}, fluxerr: {}'.format(f, f_err))

        mags = pc.FluxJyToABMag(f, fluxerr=f_err)
        mag_diff_ab = mags[0]
        print('mag AB ', mag_diff_ab)
        mag_diff_ab_err = mags[1]
        print('mag AB err ', mag_diff_ab_err)

        mags_coadd = pc.FluxJyToABMag(flux_jsky_coadd*1e-9, fluxerr=fluxerr_jsky_coadd*1e-9)

        mag_ab_coadd = mags_coadd[0]
        mag_ab_coadd_err = mags_coadd[1]
        
        print('flux before scaling: ', flux[0])

        #if flux[0] > 1500 :
        #    print('This source is bad subtracted')

        #    pass

        Fluxes_unscaled.append(flux_coadd[0])
        Fluxes_err_unscaled.append(fluxerr_coadd[0])

        Fluxes_cal.append(flux_jsky_cal)
        Fluxeserr_cal.append(fluxerr_jsky_cal)

        Fluxes_njsky.append(flux_jsky)
        Fluxeserr_njsky.append(fluxerr_jsky)

        FluxesFs_njsky.append(fluxFs_jsky)
        FluxeserrFs_njsky.append(fluxerrFs_jsky)


        ##### add different aperture sizes to their corresponding list:

        Fluxes0p5_njsky.append(fluxFs0p5_jsky)
        Fluxeserr0p5_njsky.append(fluxerrFs0p5_jsky)

        Fluxes0p75_njsky.append(fluxFs0p75_jsky)
        Fluxeserr0p75_njsky.append(fluxerrFs0p75_jsky)

        Fluxes1_njsky.append(fluxFs1_jsky)
        Fluxeserr1_njsky.append(fluxerrFs1_jsky)

        Fluxes1p25_njsky.append(fluxFs1p25_jsky)
        Fluxeserr1p25_njsky.append(fluxerrFs1p25_jsky)

        Fluxes1p5_njsky.append(fluxFs1p5_jsky)
        Fluxeserr1p5_njsky.append(fluxerrFs1p5_jsky)

        Fluxes0p75dyn_njsky.append(fluxDyn0p75_jsky)
        Fluxeserr0p75dyn_njsky.append(fluxerrDyn0p75_jsky)

        #####################################
        if do_convolution:
            FluxConv_njsky.append(fluxConv_jsky)
            FluxerrConv_njsky.append(fluxerrConv_jsky)


        #####################################

        Fluxes_njsky_coadd.append(flux_jsky_coadd)
        Fluxeserr_njsky_coadd.append(fluxerr_jsky_coadd)

        Mag.append(mag_diff_ab)
        Magerr.append(mag_diff_ab_err)

        Mag_coadd.append(mag_ab_coadd)
        Magerr_coadd.append(mag_ab_coadd_err)
        
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        Airmass.append(airmass)
        #Mag.append(pc.Calibration([m_instrumental, airmass], Z_value, k_value))
        #Magerr.append()

        #Mag.append(photocalib.instFluxToMagnitude(flux[0], fluxerr[0], obj_pos_2d).value)
        #Magerr.append(photocalib.instFluxToMagnitude(flux[0], fluxerr[0], obj_pos_2d).error)
        
        #print('flux: ', flux[0])
        print('fluxerr: ', fluxerr[0])
        flags.append(flags)
        print('------------------------------------------')

    # comparison between calibration factors 

    #plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, my_calib, '*', color='black', label='My calib')
    #plt.errorbar(dates_aux, calib_lsst, yerr=calib_lsst_err, fmt='o', color='blue', label='lsst cal')
    
    #plt.xlabel('MJD', fontsize=17)
    #plt.ylabel('Calibration mean', fontsize=17)
    #plt.title('Calibration scaling comparison', fontsize=17)
    #plt.legend()
    #plt.show()

    '''
    # calibration factor plots
    plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, calib_relative, '*', color='black', label='My calibration', linestyle='--')
    plt.errorbar(dates_aux, calib_lsst, yerr=calib_lsst_err, fmt='o', color='blue', label='LSST pipeline', linestyle='--')
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('Calibration mean', fontsize=17)
    plt.title('LSST calibration mean', fontsize=17)
    plt.show()
    '''
    # plorrint calib intercept


    #plotting everything together: 




    plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, calib_relative_intercept, '*', color='black', label='My calib', linestyle = '--')
    plt.plot(dates_aux, np.array(calib_lsst)/sum(np.array(calib_lsst)), 'o', color='blue', linestyle='--', label='scaling factor')
    
    #plt.xlabel('MJD', fontsize=17)
    #plt.ylabel('Calibration intercept', fontsize=17)
    #plt.title('Calibration scaling intercept relative to first visit', fontsize=17)
    #plt.legend()
    #plt.show()


    # Airmass plot
    #plt.figure(figsize=(10,6))
    plt.plot(dates_aux, np.array(Airmass)/sum(np.array(Airmass)), 'o', color='magenta', linestyle='--', label='airmass')
    #plt.title('Airmass', fontsize=17)
    #plt.xlabel('MJD', fontsize=17)
    #plt.ylabel('Airmass', fontsize=17)
    #plt.show()


    # Seeing plot
    plt.figure(figsize=(10,6))
    plt.plot(dates_aux, np.array(Seeing), 'o', color='black', linestyle='--', label='sigma seeing')
    #if correct_coord:
    #    #plt.plot(dates_aux, np.array(distance_me_lsst),'^', color='blue', linestyle = '--', label= 'distance LSST and sep')
    #if do_convolution:
    #    plt.plot(dates_aux, kernel_stddev, 'd', linestyle='dotted', color = 'm', label='sigma kernel')
    #plt.title('seeing sigma observation', fontsize=17)
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('Pixels', fontsize=17)
    plt.legend(frameon=False)
    plt.show()
 
    Max_flux_source = np.max(Fluxes_cal)

    if do_lc_stars == True:
        py = 2048 - 200
        px = 4096 - 200
        width = px*arcsec_to_pixel
        height = py*arcsec_to_pixel

        #print('width: {} , height : {}'.format(width, height))

        #stars_table = Find_stars(ra_center, dec_center, width, height, nstars, seed=[True, 200])
        stars_table = Inter_Join_Tables_from_LSST(repo, visits_aux, ccd_num, collection_diff, well_subtracted=well_subtracted, tp=tp)

        if tp=='before_ID':
            random.seed(15030)
            random_indexes = random.sample(range(len(stars_table)), int(len(stars_table)*0.5))
            stars_table = stars_table.loc[random_indexes]
            stars_table = stars_table.reset_index()
            #pass
        #stars_table = stars_table.reset_index()

        #Inter_Join_Tables_from_LSST(repo, visits_aux, ccd_num, collection_diff) #Find_stars_from_LSST_to_PS1(repo, visits_aux[0], ccd_num, collection_diff, n=nstars, well_subtracted=well_subtracted)
        #print(stars_table)
        if len(stars_table) == 0: # or stars_table==None
            print('No stars well subtracted were found :(')
            pass
        
        #nstars = len(stars_table)
        
        print('available stars that satisfy criteria: ', len(stars_table))
        nstars = len(stars_table)
        columns_stars = np.ndarray.flatten(np.array([['star_{}_f'.format(i+1), 'star_{}_ferr'.format(i+1), 'star_{}_fs'.format(i+1), 'star_{}_fserr'.format(i+1), 'star_{}_mag'.format(i+1), 'star_{}_magErr'.format(i+1)] for i in range(nstars)]))
        stars_calc_byme = pd.DataFrame(columns=columns_stars)
        #stars_table = stars_table.sample(n=nstars)
        
        print('number of stars we will revise: ', len(stars_table))

        ra_s = np.array(stars_table['coord_ra_ddegrees_{}'.format(visits_aux[0])])
        dec_s = np.array(stars_table['coord_dec_ddegrees_{}'.format(visits_aux[0])])
        try:
            ps1_info = pc.get_from_catalog_ps1(ra_s, dec_s)
            ps1_info = ps1_info.sort_values('gmag')
        except:
            print('couldnt retrieve the stars from the panstarss archive online')    
        #coords_stars = {}

        for j in range(len(visits_aux)):
            coords_stars = {}
            print('visits_aux[j]: ', visits_aux[j])
            RA = np.array(stars_table['coord_ra_ddegrees_{}'.format(visits_aux[j])], dtype=float)
            DEC = np.array(stars_table['coord_dec_ddegrees_{}'.format(visits_aux[j])], dtype=float)
            
            diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits_aux[j], detector=ccd_num , collections=collection_diff, instrument='DECam')

            try:
                coadd = butler.get('goodSeeingDiff_matchedExp',visit=visits_aux[j], detector=ccd_num , collections=collection_diff, instrument='DECam')
            except:
                coadd = butler.get('goodSeeingDiff_warpedExp',visit=visits_aux[j], detector=ccd_num , collections=collection_diff, instrument='DECam')
            calexp = butler.get('calexp',visit=visits_aux[j], detector=ccd_num , collections=collection_diff, instrument='DECam')
            
            photocalib = diffexp.getPhotoCalib()
            photocalib_coadd = coadd.getPhotoCalib()
            photocalib_calexp = calexp.getPhotoCalib()

            wcs = diffexp.getWcs()
            np.asarray(diffexp.image.array, dtype='float')            
            data_coadd = np.asarray(coadd.image.array, dtype='float')
            
            #detectionTask = SourceDetectionTask()
            #convCalexp = detectionTask.convolveImage(calexp.maskedImage, worst_psf)
            #calConv_image = convCalexp.middle.image.array.copy(order='C')
            #calConv_variance = convCalexp.middle.variance.array

            #calConv_image = do_convolution_image(repo, main_root, visits_aux[j], ccd_num, collection_calexp, x_pix, y_pix, worst_seeing_visit=worst_seeing_visit)[visits_aux[j]]
            data_calexp = np.asarray(calexp.image.array, dtype='float')

            if do_convolution:
                #calConv, stddev_kernel = do_convolution_image(repo, main_root, visits_aux[j], ccd_num, collection_calexp, ra, dec, worst_seeing_visit =  worst_seeing_visit, mode=mode)
                calConv_image = Data_convol[visits_aux[j]] #calConv[visits_aux[j]]
                data_calexp = calConv_image
            
            #np.asarray(calexp.image.array, dtype='float')           
            psf = diffexp.getPsf()
            fwhm = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()* arcsec_to_pixel *sigma2fwhm 

            flux_stars_and_errors = []
            #r_star = 2 #2.5
            star_aperture = r_star # arcseconds #r_star * fwhm/2
            star_aperture/=arcsec_to_pixel # transform it to pixel values 

            exp_visit_info = diffexp.getInfo().getVisitInfo()
            visit_date_python = exp_visit_info.getDate().toPython()
            visit_date_astropy = Time(visit_date_python)        
            d = visit_date_astropy.mjd

            print('star aperture in pixels: ', star_aperture)

            saturated_stars = []



            
            for i in range(nstars):
                
                ra_star = RA[i]
                dec_star = DEC[i]
                obj_pos_2d_star = lsst.geom.Point2D(ra_star, dec_star)
                if verbose:
                    print('⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂ (◕ᴥ◕) -. * STAR {} *.- (◕ᴥ◕) ⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑ '.format(i+1))
                
                obj_pos_lsst_star = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
                x_star, y_star = wcs.skyToPixel(obj_pos_lsst_star)  
                coords_stars['science_star_{}'.format(i+1)] = [x_star, y_star]

                if verbose:
                    print('x_pix : {}  y_pix : {}'.format(x_star, y_star))
                
                if do_convolution:
                    cut_aux=20
                    dataconv_cut = data_calexp[int(y_star-cut_aux): int(y_star+cut_aux),int(x_star-cut_aux): int(x_star+cut_aux)].copy(order='C')
                    #print('', dataconv_cut)
                    #circle = plt.Circle((x_star, y_star), radius = star_aperture, color=neon_green, fill = False, linewidth=4)
                    #plt.gca().add_patch(circle)
                    #plt.scatter(x_star, y_star, marker='x', color=neon_green)
                    #plt.show()
                    #m, s = np.mean(np.array(dataconv_cut).flatten()),  np.std(np.array(dataconv_cut).flatten())
                    #plt.title('Convolution downgrade to the worst seeing', fontsize=15)
                    m, s = np.mean(dataconv_cut), np.std(dataconv_cut)
                    #if show_star_stamps:
                    #    plt.imshow(dataconv_cut, cmap='rocket', vmin=m-s, vmax = m+s)
                    #    plt.show()
                    #plt.colorbar()
                    
                    try:
                        peak_value = np.max(np.array(dataconv_cut).flatten())
                        objects = sep.extract(dataconv_cut, m+s, minarea=10)

                        obj, j = Select_largest_flux(dataconv_cut, objects)
                        x_star_conv = obj['x'] + int(x_star - cut_aux)
                        y_star_conv = obj['y'] + int(y_star - cut_aux)
                        coords_stars['scienceConv_star_{}'.format(i+1)] = [x_star_conv, y_star_conv]
                        #if show_star_stamps:# and i==5:
                        #    print('⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂ (◕ᴥ◕) -. * STAR {} *.- (◕ᴥ◕) ⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑ '.format(i+1))
                        #    plt.title('Convolution downgrade to the worst seeing', fontsize=15)
                        #    plt.imshow(data_calexp, cmap='rocket', vmin=0, vmax = peak_value)
                        #    plt.colorbar()
                        #    circle = plt.Circle((x_star_conv, y_star_conv), radius = star_aperture, color=neon_green, fill = False, linewidth=4)
                        #    plt.gca().add_patch(circle)
                        #    plt.scatter(x_star_conv, y_star_conv, marker='x', color=neon_green)
                        #    plt.xlim(x_star_conv - cut_aux, x_star_conv + cut_aux)
                        #    plt.ylim(y_star_conv - cut_aux, y_star_conv + cut_aux)
                        #    plt.show()
                    except:
                        pass
                    #Calib_Diff_and_Coadd_plot_cropped(repo, collection_diff, ra_star, dec_star, [visits_aux[j]], ccd_num, s=star_aperture, cutout=2*star_aperture)
                    #values_across_source(calexp, ra_star, dec_star , x_length = star_aperture, y_length=1.5, stat='median', title_plot='Calibrated exposure of star {}'.format(i+1), save_plot = True, field=field, name='median_slit_hist_star{}_mjd_{}'.format(i+1, Truncate(d, 4)))
                    #values_across_source(diffexp, ra_star, dec_star , x_length = star_aperture, y_length=1.5, stat='median', title_plot = 'Difference exposure of star {}'.format(i+1))
            
                    
                
                f, f_err, fg = sep.sum_circle(data, [x_star], [y_star], star_aperture, var = np.asarray(diffexp.variance.array, dtype='float'))
                #ft, ft_err, ftg = sep.sum_circle(data_coadd, [x_star], [y_star], star_aperture, var = np.asarray(coadd.variance.array, dtype='float'))
                if do_convolution:
                    fs, fs_err, fsg = sep.sum_circle(data_calexp, [x_star_conv], [y_star_conv], star_aperture)
                else:
                    fs, fs_err, fsg = sep.sum_circle(data_calexp, [x_star], [y_star], star_aperture, var = np.asarray(calexp.variance.array, dtype='float'))
                                
                # Using LSST photocalibration
                f_star_physical = photocalib.instFluxToNanojansky(f[0], f_err[0], obj_pos_2d_star)
                #ft_star_physical = photocalib_coadd.instFluxToNanojansky(ft[0], ft_err[0], obj_pos_2d_star)
                fs_star_physical = photocalib_calexp.instFluxToNanojansky(fs[0], fs_err[0], obj_pos_2d_star)


                flux_stars_and_errors.append(f_star_physical.value)
                flux_stars_and_errors.append(f_star_physical.error)
                #flux_stars_and_errors.append(ft_star_physical.value)
                #flux_stars_and_errors.append(ft_star_physical.error)
                flux_stars_and_errors.append(fs_star_physical.value)
                flux_stars_and_errors.append(fs_star_physical.error)


                Fstar = np.array((fs_star_physical.value)*1e-9)
                Fstar_err = np.sqrt((fs_star_physical.error*1e-9)**2)

                Magstars = pc.FluxJyToABMag(Fstar, Fstar_err)
                Mag_star = Magstars[0]
                Mag_star_err = Magstars[1]

                #Magstars_coadd = pc.FluxJyToABMag(ft_star_physical.value*1e-9, ft_star_physical.error*1e-9)
                #Mag_star_coadd = Magstars_coadd[0]
                #Mag_star_coadd_err = Magstars_coadd[1]

                # Using my calibration  
                
                #flux_stars_and_errors.append(f[0]*calib_lsst[j])
                #flux_stars_and_errors.append(f_err[0]*calib_lsst[j])
                #flux_stars_and_errors.append(ft[0]*calib_lsst[j])
                #flux_stars_and_errors.append(ft_err[0]*calib_lsst[j])
                #flux_stars_and_errors.append(fs[0]*calib_lsst[j])
                #flux_stars_and_errors.append(fs_err[0]*calib_lsst[j])

                #Fstar = np.array((f[0]*calib_lsst[j] + ft[0]*calib_lsst[j])) #flux [nJy]
                #Fstar_err = np.sqrt((ft_err[0]*calib_lsst[j])**2 + (f_err[0]*calib_lsst[j])**2) #flux [nJy]

                #Magstars = pc.FluxJyToABMag(Fstar*1e-9, Fstar_err*1e-9)
                #Mag_star = Magstars[0]
                #Mag_star_err = Magstars[1]

                #Magstars_coadd = pc.FluxJyToABMag(ft[0]**calib_lsst[j]*1e-9, ft_err[0]**calib_lsst[j]*1e-9)
                #Mag_star_coadd = Magstars_coadd[0]
                #Mag_star_coadd_err = Magstars_coadd[1]
                
                flux_stars_and_errors.append(Mag_star)
                flux_stars_and_errors.append(Mag_star_err)
                #flux_stars_and_errors.append(Mag_star_coadd)
                #flux_stars_and_errors.append(Mag_star_coadd_err)
                #print('len flux_stars_and_errors: ', len(flux_stars_and_errors))

                ###########
                if verbose:
                    print('Flux star: {} Error flux: {}'.format(fs[0], fs_err[0]))
                    #xprint('Flux star in coadd: {} Error flux in coadd: {}'.format(fs[0], fs_err[0]))
                    
                    print('꒰✩ ’ω`ૢ✩꒱ -------------------- ⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂')
            
            stars_calc_byme.loc[len(stars_calc_byme.index)] = flux_stars_and_errors
            
            print('looking at visit:', np.array(visits_aux)[j])

            if show_star_stamps:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 32))
                ax1.axis('off')
                ax2.axis('off')
                im1 = ax1.imshow(np.asarray(calexp.image.array,dtype='float'), vmin = m-s, vmax = m+s)
                im2 = ax2.imshow(calConv_image, vmin = m-s, vmax = m+s)
                #print('coords_stars: ', coords_stars)
                
                for s in range(nstars):
                    x_star, y_star = coords_stars['science_star_{}'.format(s+1)]
                    ax1.add_patch(plt.Circle((x_star, y_star), radius=star_aperture, color=neon_green, fill=False))
                    ax1.text(x_star+star_aperture, y_star, '{}'.format(s+1), color=neon_green)

                    if do_convolution:
                        x_starc, y_starc = coords_stars['scienceConv_star_{}'.format(s+1)]
                        ax2.add_patch(plt.Circle((x_starc, y_starc), radius=star_aperture, color=neon_green, fill=False))
                        ax2.text(x_starc+star_aperture*1.05, y_starc, '{}'.format(s+1), color=neon_green)

                plt.show()

        
        #field = collection_diff[13:24]
        #here we plot the stars vs panstarss magnitude:
        norm = matplotlib.colors.Normalize(vmin=0,vmax=32)
        c_m = matplotlib.cm.plasma

        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(0,27,len(visits_aux))
        fluxs_stars = [np.median(np.array(stars_calc_byme['star_{}_fs'.format(i+1)])) for i in range(nstars)]
        stars_indexbF, =  np.where(fluxs_stars <= Max_flux_source) # stars that are equal or below the flux of the galaxy
        if len(stars_indexbF) == 0:
            stars_indexbF, =  np.where(fluxs_stars <= np.percentile(fluxs_stars,50))

        columns_stars_bF = np.ndarray.flatten(np.array([['star_{}_f'.format(i+1), 'star_{}_ferr'.format(i+1), 'star_{}_fs'.format(i+1), 'star_{}_fserr'.format(i+1), 'star_{}_mag'.format(i+1), 'star_{}_magErr'.format(i+1)] for i in np.array(stars_indexbF)]))
        #print('columns stars_calc_byme: ', stars_calc_byme.columns)
        #print('columns stars bF: ', columns_stars_bF)
        stars_calc_byme = stars_calc_byme[list(columns_stars_bF)]
        fluxs_stars_filtered = [np.median(np.array(stars_calc_byme['star_{}_fs'.format(i+1)])) for i in stars_indexbF]

        #fluxt_stars_norm_factor = np.linalg.norm(fluxt_stars)
        #fluxt_stars_norm = fluxt_stars/fluxt_stars_norm_factor

        plt.show()

        #'''
        
        #plt.figure(figsize=(10,10))
        #stars_table = stars_table.sort_values('base_PsfFlux_mag_{}'.format(visits_aux[0]))
        ##kk = 0
        #for i in range(len(visits_aux)):
        #    plt.errorbar(np.array(ps1_info.gmag) , np.array(stars_table['base_PsfFlux_mag_{}'.format(visits_aux[i])]) - np.array((ps1_info.gmag)), yerr= np.fabs(np.array(stars_table['base_PsfFlux_magErr_{}'.format(visits_aux[i])])), fmt='*', label=' visit {}'.format(visits_aux[i]), color=s_m.to_rgba(T[i]), markersize=10, alpha=0.5, linestyle='--')
        #    #k+=0
        #    #plt.plot(np.array(ps1_mags.ps1_mag), np.array(ps1_mags.ps1_mag) - np.array(ps1_mags.ps1_mag), label='PS1')
        #plt.xlabel('PS1 magnitudes', fontsize=17)
        #plt.plot(np.sort(np.array(ps1_info.gmag)), np.array(ps1_info.gmag) - np.array(ps1_info.gmag),'*', markersize=10, label='PS1', color='black', linestyle='--')
        #color_term = 20e-3
        #color_term_rms = 6e-3
        ##plt.plot(ps1_info.gmag, ps1_info.g_r*color_term , '*', color='green', markersize=10, alpha=0.5, linestyle='--', label='g-r * color term')
        #plt.errorbar(ps1_info.gmag,  ps1_info.g_i*color_term , yerr=ps1_info.g_i*color_term_rms , fmt='*', color='black', capsize=4, markersize=10, alpha=0.5, linestyle='--', label='g-i * color term')        
        #plt.ylabel('Magnitude - PS1 magnitude', fontsize=17)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.title('Difference between magnitudes', fontsize=17)
        #if save_lc_stars:
        #    plt.savefig('light_curves/{}/{}_{}_magnitude_and_colors.jpeg'.format(field, field, ccd_num), bbox_inches='tight')
        #plt.show()

        #color_correc = pd.Series(ps1_info.g_i*color_term ,index=ps1_info.index).to_dict()
        #'''

        
        #print(color_correc)
        #try:
        #    plt.figure(figsize=(10,10))
        #    abs_zero = 0#.04
        #    stars_table = stars_table.dropna()
        #    for i in range(len(visits_aux)):
        #        mag_unc_err = np.sqrt(np.array(stars_table['base_PsfFlux_magErr_{}'.format(visits_aux[i])])**2 + (ps1_info.g_i*color_term_rms)**2 )
        #        plt.errorbar(np.array(ps1_info.gmag) , np.array(stars_table['base_PsfFlux_mag_{}'.format(visits_aux[i])].astype(float)) - np.array(ps1_info.gmag) - ps1_info.g_i*color_term - abs_zero ,yerr=mag_unc_err, fmt='*', capsize=4, label=' visit {}'.format(visits_aux[i]), color=s_m.to_rgba(T[i]), markersize=10, alpha=0.5, linestyle='--')
        #        #plt.plot(np.array(ps1_mags.ps1_mag), np.array(ps1_mags.ps1_mag) - np.array(ps1_mags.ps1_mag), label='PS1')
        #    #stars_table['color_color_term'] = ps1_info.g_i*color_term 
        #    plt.xlabel('PS1 magnitudes', fontsize=17)
        #    plt.plot(np.sort(np.array(ps1_info.gmag)), np.array(ps1_info.gmag) - np.array(ps1_info.gmag),'*', markersize=10, label='PS1', color='black', linestyle='--')
        #    
        #    #color_term = 20e-3
        #    #plt.plot(ps1_info.gmag, ps1_info.g_r*color_term , '*', color='green', markersize=10, alpha=0.5, linestyle='--', label='g-r * color term')
        #    #plt.plot(ps1_info.gmag, ps1_info.g_i*color_term , '*', color='black', markersize=10, alpha=0.5, linestyle='--', label='g-i * color term')
        #    
        #    plt.ylabel('Magnitude - PS1 magnitude', fontsize=17)
        #    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #    plt.title('Difference between magnitudes', fontsize=17)
        #    #plt.savefig('light_curves/{}/{}_{}_magnitude_and_colors.png'.format(field, field, ccd_num), bbox_inches='tight')
        #    plt.show()
        #except:
        #    pass
        
        #norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
        #c_m = matplotlib.cm.plasma

        #s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        #s_m.set_array([])
        #T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

        columns_mag = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        columns_magErr = ['base_PsfFlux_magErr_{}'.format(v) for v in visits_aux]
        #fig, axs = plt.subplots(int(np.sqrt(nstars))+1,int(np.sqrt(nstars))+1 , figsize=(10, 6), constrained_layout=True)
        ##fig.set_title('Flux of stars measured by LSST')
        #for ax, markevery in zip(axs.flat, range(nstars)):           
        #    ax.set_title(f'star number {markevery+1}')
        #    magss = np.array(stars_table.loc[markevery][columns_mag])
        #    magsserr = np.array(stars_table.loc[markevery][columns_magErr])
        #    flux = [pc.ABMagToFlux(m)*1e9 for m in magss]
        #    #fluxErr  = [pc.ABMagToFlux()]
        #    ax.errorbar(dates_aux, flux, fmt = 'o', ls='-', color=s_m.to_rgba(fluxt_stars[markevery]))
        #here we plot the stars vs panstarss magnitude:
        
        #norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
        #c_m = matplotlib.cm.plasma

        # create a ScalarMappable and initialize a data structure
        #s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        #s_m.set_array([])
        #T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

        plt.figure(figsize=(10,10))
        ii = 0
        stars_lsst_calib = stars_table[columns_mag]
        #print(stars_lsst_calib)
        
        #stars_lsst_calib = stars_lsst_calib.apply(pc.ABMagToFlux)
        #stars_lsst_calib = stars_lsst_calib*1e9
        
        #print(stars_lsst_calib)

        #print(stars_lsst_calib.median(axis=1))
        #median_stars_lsst = pd.Series(stars_lsst_calib.median(axis=1))
        #stars_lsst_calib = stars_lsst_calib.reset_index()
        #stars_lsst_calib = stars_lsst_calib.sub(median_stars_lsst, axis='index')
        #print(stars_lsst_calib)
        #stars_lsst_mean = np.array([np.mean(stars_lsst_calib[i]) for i in columns_mag])
        #stars_lsst_std = np.array([np.std(stars_lsst_calib[i]) for i in columns_mag])
        mags_stars = np.array(stars_table[columns_mag[0]])
        c_m = matplotlib.cm.plasma
        norm = matplotlib.colors.Normalize(vmin=min(mags_stars),vmax=max(mags_stars))
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(min(fluxs_stars),max(fluxs_stars),nstars)
        for j in range(len(stars_table)):
            #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
            mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
            magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
            fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
            flux_of_star_j = fluxes_of_star_j[0]#*1e9
            fluxErr_of_star_j = fluxes_of_star_j[1]#*1e9
            #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
            plt.errorbar(dates_aux - min(dates_aux), flux_of_star_j - np.median(flux_of_star_j), fluxErr_of_star_j, fmt='*', color=s_m.to_rgba(np.median(mag_of_star_j)), ls='--', label='star {}'.format(j+1))
            marker_labels = np.ones(len(dates_aux))*(int(j+1))
            
            stars_xarray = np.array(dates_aux - min(dates_aux))
            stars_yarray = np.array(flux_of_star_j - np.median(flux_of_star_j))
            for i, label in enumerate(marker_labels):
                plt.annotate(str(int(label)), (stars_xarray[i], stars_yarray[i]), color='k')
            
            ii+=1
        plt.colorbar(s_m, label = 'Median Flux [nJy]')
        plt.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
        plt.xlabel('MJD', fontsize=17)
        plt.ylabel('Flux [nJy] of LSST - median', fontsize=17)
        plt.title('Lightcurves of stars, measured by LSST', fontsize=17)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        try:
            plt.savefig('{}/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(main_root, field), bbox_inches='tight')
        except:
            print('could save the lightcurves for stars')
        plt.show()


        #plt.figure(figsize=(10,10))
        #ii = 0
        #norm = matplotlib.colors.Normalize(vmin=min(fluxs_stars),vmax=max(fluxs_stars))
        #s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        #s_m.set_array([])
        ##deviation_from_median = []
        #for j in range(len(stars_table)):
        #    #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        #    mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
        #    magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
        #    #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
        #    plt.hist(mag_of_star_j - np.median(mag_of_star_j), alpha=0.5, label='star {}'.format(j+1), color=s_m.to_rgba(mag_of_star_js[j]))
        #    ii+=1

        ##plt.ylabel('MJD', fontsize=17)
        #plt.xlabel('Flux [nJy] of LSST - median', fontsize=17)
        #plt.title('Lightcurves of stars, measured by LSST', fontsize=17)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.savefig('/home/pcaceres/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        #plt.show()


        #plt.figure(figsize=(10,10))
        #ii = 0
        
        #excess_var = []
        #for j in range(len(stars_table)):
            #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        #    mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
        #    magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
            #fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
            #flux_of_star_j = fluxes_of_star_j[0]*1e9
            #fluxErr_of_star_j = fluxes_of_star_j[1]*1e9
            #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
            #plt.hist(Excess_variance(mag_of_star_j, magErr_of_star_j), alpha=0.5, label='star {}'.format(j+1), histtype='step', color=s_m.to_rgba(fluxt_stars[j]))
        #    excess_var.append(Excess_variance(mag_of_star_j, magErr_of_star_j))
            #print(Excess_variance(mag_of_star_j, magErr_of_star_j))
        #    ii+=1

        #plt.hist(excess_var, color='black')
        #print(excess_var)
        #plt.xlabel('Excess Variance of stars', fontsize=17)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        #plt.savefig('/home/pcaceres/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        #plt.show()


        #plt.figure(figsize=(10,10))
        #ii = 0
        #
        #for j in range(len(stars_table)):
        #    #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        #    mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
        #    magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
        #    fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
        #    flux_of_star_j = fluxes_of_star_j[0]*1e9
        #    fluxErr_of_star_j = fluxes_of_star_j[1]*1e9
        #    #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
        #    plt.hist((flux_of_star_j - np.median(flux_of_star_j)) / np.median(flux_of_star_j) * 100, alpha=0.5, label='star {}'.format(j+1), color=s_m.to_rgba(fluxt_stars[j]))
        #    ii+=1

        #plt.xlabel('(flux_j - flux_median) / flux_median', fontsize=17)
        #plt.title('Percentage of deviation from median, in flux', fontsize=17)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ##plt.savefig('/home/pcaceres/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        #plt.show()

        #plt.figure(figsize=(10,10))
        #ii = 0

        #deviation_from_median = []
        #for j in range(len(stars_table)):
        #    mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
        #    plt.hist(mag_of_star_j - np.median(mag_of_star_j), alpha=0.5, label='star {}'.format(j+1), color=s_m.to_rgba(fluxt_stars[j]))
        #    ii+=1

        #plt.ylabel('MJD', fontsize=17)
        #plt.xlabel('AB mag of LSST - median', fontsize=17)
        #plt.title('Lightcurves of stars, measured by LSST', fontsize=17)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.savefig('/home/pcaceres/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        #plt.show()

        try:
            plt.figure(figsize=(10,6))
            mags_visits_list = []
            mags_visits_p16 = []
            mags_visits_p84 = []
            stars['dates'] = dates_aux - min(dates_aux)#dates - min(dates)

            for i in range(len(visits_aux)):
                mag_stars = np.array(stars_table['base_PsfFlux_mag_{}'.format(visits_aux[i])])
                mags_visits = np.mean(mag_stars)
                #mags_visits_p16.append(np.percentile(mag_stars,))
                mags_visits_list.append(mags_visits)
                mags_ps1_mean = np.mean(ps1_info.gmag)
            plt.plot(stars['dates'], np.array(mags_visits_list) - mags_ps1_mean, '*', color='magenta', markersize=10, alpha=0.5, linestyle='--')
            plt.xlabel('MJD', fontsize=17)
            plt.ylabel('mean mag LSST - mean mag PS1', fontsize=17)
            #plt.legend()
            plt.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
            plt.title('Difference between means of magnitudes', fontsize=17)
            plt.show()
        except:
            print('Couldnt query the stars from the archive online in panstarss')
        saturated_stars = np.unique(np.array(saturated_stars))
        #plt.figure(figsize=(10,6))       
        #stars = stars.sort_values(by='dates')
        #column_stars_diff_flux = ['star_{}_f'.format(i+1) for i in range(nstars)]
        
        ## plotting flxs of stars 
        #for i in range(nstars):
        #    ft_star = (np.array(stars['star_{}_ft'.format(i+1)])).flatten() #* scaling
        #    ft_star_err = np.ndarray.flatten(np.array(stars['star_{}_fterr'.format(i+1)])) #* scaling
        #    
        #    #Dates = np.array(stars['dates'])
        #    new_dates = dates_aux
        #    j, = np.where(saturated_stars==i+1)
        #    
        #    norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
        #    c_m = matplotlib.cm.plasma

        #    # create a ScalarMappable and initialize a data structure
        #    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        #    s_m.set_array([])
        #    T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

        #    if len(j)==0:
        #        plt.errorbar(dates_aux, ft_star - np.median(ft_star), yerr= ft_star_err, capsize=4, fmt='s', ls='solid', label = 'star {} coadd'.format(i+1), color = s_m.to_rgba(fluxt_stars[i]))
        #if well_subtracted:
        #    plt.title('stars LCs in template/coadd image from {} and {} with Aperture of {}*FWHM", well subtracted'.format(field, ccd_name[ccd_num], r_star)) 
        #if not well_subtracted:
        #    plt.title('stars LCs in template/coadd image from {} and {} with Aperture {}*FWHM'.format(field, ccd_name[ccd_num], r_star)) 

        #plt.xlabel('MJD', fontsize=15)
        #plt.ylabel('offset Flux [nJy] from median', fontsize=15)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #plt.show()

        #norm = matplotlib.colors.Normalize(vmin=min(fluxs_stars),vmax=max(fluxs_stars))
        c_m = matplotlib.cm.plasma

        s_m = matplotlib.cm.ScalarMappable(cmap=c_m)
        s_m.set_array([])
        T = np.linspace(min(fluxs_stars),max(fluxs_stars),nstars)

        columns_mag = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        columns_magErr = ['base_PsfFlux_magErr_{}'.format(v) for v in visits_aux]
        #fig, axs = plt.subplots(int(np.sqrt(nstars))+1,int(np.sqrt(nstars))+1 , figsize=(10, 6), constrained_layout=True)
        #fig.set_title('Flux measured on coadd template', fontsize=17)
        #for ax, markevery in zip(axs.flat, range(nstars)):
        #    ft_star = (np.array(stars['star_{}_ft'.format(markevery+1)])).flatten() #* scaling
        #    ft_star_err = np.ndarray.flatten(np.array(stars['star_{}_fterr'.format(markevery+1)])) #* scaling

        #    ax.set_title(f'star number {markevery+1}')
        #    ax.errorbar(dates_aux, ft_star, yerr = ft_star_err, fmt = 'o', ls='-', color=s_m.to_rgba(fluxt_stars[markevery]))
        # plotting flxs of stars 
        #plt.title('Flux measured on coadd template', fontsize=17)
        
        # science flux
        Results_star = pd.DataFrame(columns = ['date', 'stars_science_1sigmalow_byEpoch', 'stars_science_1sigmaupp_byEpoch', 'stars_science_2sigmalow_byEpoch', 'stars_science_2sigmaupp_byEpoch', 'stars_diff_1sigmalow_byEpoch', 'stars_diff_1sigmaupp_byEpoch', 'stars_diff_2sigmalow_byEpoch', 'stars_diff_2sigmaupp_byEpoch'])
        
        stars_science_flux_columns = ['star_{}_fs'.format(i+1) for i in np.array(stars_indexbF)]
        stars_science_flux = stars_calc_byme[stars_science_flux_columns]
        stars_science_flux -= stars_science_flux.median()
        #stars_science_flux = stars_science_flux.reset_index()
        stars_science_1sigmalow_byEpoch = np.array([np.percentile(np.array(stars_science_flux.loc[i]), 16) for i in range(len(stars_science_flux))])
        stars_science_1sigmaupp_byEpoch = np.array([np.percentile(np.array(stars_science_flux.loc[i]), 84) for i in range(len(stars_science_flux))])
        stars_science_2sigmalow_byEpoch = np.array([np.percentile(np.array(stars_science_flux.loc[i]), 2.5) for i in range(len(stars_science_flux))])
        stars_science_2sigmaupp_byEpoch = np.array([np.percentile(np.array(stars_science_flux.loc[i]), 97.5) for i in range(len(stars_science_flux))])
        Results_star['date'] = stars['dates']
        Results_star['stars_science_1sigmalow_byEpoch'] = stars_science_1sigmalow_byEpoch
        Results_star['stars_science_1sigmaupp_byEpoch'] = stars_science_1sigmaupp_byEpoch
        Results_star['stars_science_2sigmalow_byEpoch'] = stars_science_2sigmalow_byEpoch
        Results_star['stars_science_2sigmaupp_byEpoch'] = stars_science_2sigmaupp_byEpoch


        stars_science_mean_byEpoch = np.array([np.mean(np.array(stars_science_flux.loc[i])) for i in range(len(stars_science_flux))])

        # difference flux
        stars_diff_flux_columns = ['star_{}_f'.format(i+1) for i in np.array(stars_indexbF)]
        stars_diff_flux = stars_calc_byme[stars_diff_flux_columns]
        #stars_science_flux -= stars_science_flux.median()
        #stars_science_flux = stars_science_flux.reset_index()
        stars_diff_sigma_byEpoch = np.array([np.std(np.array(stars_diff_flux.loc[i])) for i in range(len(stars_science_flux))])
        stars_diff_mean_byEpoch = np.array([np.mean(np.array(stars_diff_flux.loc[i])) for i in range(len(stars_science_flux))])
        stars_diff_1sigmalow_byEpoch = np.array([np.percentile(np.array(stars_diff_flux.loc[i]), 16) for i in range(len(stars_diff_flux))])
        stars_diff_1sigmaupp_byEpoch = np.array([np.percentile(np.array(stars_diff_flux.loc[i]), 84) for i in range(len(stars_diff_flux))])
        stars_diff_2sigmalow_byEpoch = np.array([np.percentile(np.array(stars_diff_flux.loc[i]), 2.5) for i in range(len(stars_diff_flux))])
        stars_diff_2sigmaupp_byEpoch = np.array([np.percentile(np.array(stars_diff_flux.loc[i]), 97.5) for i in range(len(stars_diff_flux))])

        Results_star['stars_diff_1sigmalow_byEpoch'] = stars_diff_1sigmalow_byEpoch
        Results_star['stars_diff_1sigmaupp_byEpoch'] = stars_diff_1sigmaupp_byEpoch
        Results_star['stars_diff_2sigmalow_byEpoch'] = stars_diff_2sigmalow_byEpoch
        Results_star['stars_diff_2sigmaupp_byEpoch'] = stars_diff_2sigmaupp_byEpoch  
        
        plt.show()
        plt.figure(figsize=(10,10))
        for i in np.array(stars_indexbF):
            fs_star = (np.array(stars_calc_byme['star_{}_fs'.format(i+1)])).flatten() #* scaling
            fs_star_err = np.ndarray.flatten(np.array(stars_calc_byme['star_{}_fserr'.format(i+1)])) #* scaling
            
            #Dates = np.array(stars['dates'])
            new_dates = dates_aux
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=min(fluxs_stars_filtered),vmax=max(fluxs_stars_filtered))
            c_m = matplotlib.cm.plasma

            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            #if len(j)==0:
            #print('fs_star: ', fs_star)
            #, label = 'star {} science'.format(i+1)
            plt.errorbar(dates_aux, fs_star - np.median(fs_star), yerr= fs_star_err, capsize=4, fmt='s', ls='solid', color = s_m.to_rgba(fluxs_stars[i]))
            
            marker_labels = np.ones(len(dates_aux))*(int(i+1))
            
            stars_xarray = np.array(dates_aux)
            stars_yarray = np.array(fs_star - np.median(fs_star))

            for w, label in enumerate(marker_labels):
                plt.annotate(str(int(label)), (stars_xarray[w], stars_yarray[w]), color='k')

        if well_subtracted:
            plt.title('stars LCs in science image from {} and {} with Aperture radii of {}", well subtracted'.format(field, ccd_name[ccd_num], star_aperture*arcsec_to_pixel)) 
        if not well_subtracted:
            plt.title('stars LCs in science image from {} and {} with Aperture radii of {}" '.format(field, ccd_name[ccd_num], star_aperture*arcsec_to_pixel)) 

        plt.xlabel('MJD', fontsize=15)
        plt.ylabel('offset Flux [nJy] from median', fontsize=15)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.colorbar(s_m, label = 'Median Flux [nJy]')
        plt.legend()
        plt.show()

        #plt.figure(figsize=(10,6))
        #for i in range(nstars):
        #    fs_star = (np.array(stars_calc_byme['star_{}_fs'.format(i+1)])).flatten() #* scaling
        #    fs_star_err = np.ndarray.flatten(np.array(stars_calc_byme['star_{}_fserr'.format(i+1)])) #* scaling
        #    
        #    #Dates = np.array(stars['dates'])
        #    new_dates = dates_aux
        #    j, = np.where(saturated_stars==i+1)
        #    
        #    norm = matplotlib.colors.Normalize(vmin=min(fluxs_stars),vmax=max(fluxs_stars))
        #    c_m = matplotlib.cm.plasma

        #    # create a ScalarMappable and initialize a data structure
        #    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        #    s_m.set_array([])
        #    T = np.linspace(min(fluxs_stars),max(fluxs_stars),nstars)
        #    #if len(j)==0:
        #    plt.hist(fs_star - np.median(fs_star), alpha=0.5, label = 'star {} science'.format(i+1), color = s_m.to_rgba(fluxs_stars[i]))  
        #plt.xlabel('MJD', fontsize=15)
        #plt.ylabel('offset Flux [nJy] from median', fontsize=15)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #plt.show()


        norm = matplotlib.colors.Normalize(vmin=min(fluxs_stars),vmax=max(fluxs_stars))
        c_m = matplotlib.cm.plasma

        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(min(fluxs_stars),max(fluxs_stars),nstars)

        columns_mag = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        columns_magErr = ['base_PsfFlux_magErr_{}'.format(v) for v in visits_aux]
        #fig, axs = plt.subplots(int(np.sqrt(nstars))+1,int(np.sqrt(nstars))+1 , figsize=(10, 6), constrained_layout=True)
        ##fig.set_title('Flux measured on coadd template', fontsize=17)
        #for ax, markevery in zip(axs.flat, range(nstars)):
        #    ft_star = (np.array(stars['star_{}_fs'.format(markevery+1)])).flatten() #* scaling
        #    ft_star_err = np.ndarray.flatten(np.array(stars['star_{}_fserr'.format(markevery+1)])) #* scaling

        #    ax.set_title(f'star number {markevery+1}')
        #    ax.errorbar(dates_aux, ft_star, yerr = ft_star_err, fmt = 'o', ls='-', color=s_m.to_rgba(fluxt_stars[markevery]))
        ## plotting flxs of stars 
        ##plt.title('Flux measured on coadd template', fontsize=17)
        #plt.show()
        plt.figure(figsize=(10,10))
        for i in np.array(stars_indexbF):
            f_star = np.array(stars_calc_byme['star_{}_f'.format(i+1)]) #* scaling
            f_star_err = np.array(stars_calc_byme['star_{}_ferr'.format(i+1)]) #* scaling
                  
            Dates = np.array(stars['dates'])
            new_dates = dates_aux - min(dates_aux)
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=min(fluxs_stars_filtered),vmax=max(fluxs_stars_filtered))
            c_m = matplotlib.cm.plasma

            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            #T = np.linspace(min(fluxs_stars),max(fluxs_stars),nstars)
            if len(j)==0:
                #,label = 'star {} diff'.format(i+1)
                plt.errorbar(Dates, f_star, yerr= f_star_err, capsize=4, fmt='s', ls='solid', color = s_m.to_rgba(fluxs_stars[i]))
            
            marker_labels = np.ones(len(dates_aux))*(int(i+1))
            
            stars_xarray = np.array(dates_aux)
            stars_yarray = np.array(f_star)

            for w, label in enumerate(marker_labels):
                plt.annotate(str(int(label)), (stars_xarray[w], stars_yarray[w]), color=neon_green)
        
        plt.ylabel('Difference Flux [nJy]', fontsize=15)    
        plt.xlabel('MJD', fontsize=15)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.colorbar(s_m, label = 'Median Flux [nJy]')
        plt.legend()

        plt.show()

        #plt.figure(figsize=(10,6))

        #for i in range(nstars):
        #    f_star = np.array(stars_calc_byme['star_{}_f'.format(i+1)]) #* scaling
        #    f_star_err = np.array(stars_calc_byme['star_{}_ferr'.format(i+1)]) #* scaling
        #    
        #    #Dates = np.array(stars['dates'])
        #    j, = np.where(saturated_stars==i+1)
        #    
        #    norm = matplotlib.colors.Normalize(vmin=min(fluxs_stars),vmax=max(fluxs_stars))
        #    c_m = matplotlib.cm.plasma

        #    # create a ScalarMappable and initialize a data structure
        #    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        #    s_m.set_array([])
        #    T = np.linspace(min(fluxs_stars),max(fluxs_stars),nstars)

        #    #if len(j)==0:
        #    plt.hist(f_star - np.median(f_star), alpha=0.5, label = 'star {} diff'.format(i+1), color = s_m.to_rgba(fluxs_stars[i]))
  
        ##plt.xlabel('MJD', fontsize=15)
        #plt.xlabel('offset Flux [nJy] from median', fontsize=15)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.show()


        

        #for i in range(nstars):
        #    Dates = np.array(stars['dates'])
        #    norm = matplotlib.colors.Normalize(vmin=0,vmax=32)
        #    c_m = matplotlib.cm.plasma
        #    # create a ScalarMappable and initialize a data structure
        #    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        #    s_m.set_array([])
        #    T = np.linspace(0,27,nstars)
        #    perc = stars['percent_var_star_{}'.format(i+1)]
        #    plt.plot(Dates, perc, linestyle = '--', color = s_m.to_rgba(T[i]), label = 'star {}'.format(i+1))
        #
        #plt.xlabel('MJD', fontsize=15)
        #plt.ylabel('Percentage of variation', fontsize=15)
        #plt.legend(loc=9, ncol=5)
        

    field = collection_diff[13:24]
    if save_lc_stars and well_subtracted:
        print('saving lcs stars as: ' +'light_curves/{}/{}_{}_random_stars.jpeg'.format(field, field, ccd_name[ccd_num]))
        plt.savefig('light_curves/{}/{}_{}_random_stars_sumwTemp.jpeg'.format(field, field, ccd_name[ccd_num]), bbox_inches='tight')
    if save_lc_stars and not well_subtracted:
        print('saving lcs stars as: ' +'light_curves/{}/{}_{}_random_stars.jpeg'.format(field, field, ccd_name[ccd_num]))
        plt.savefig('light_curves/{}/{}_{}_random_stars_wdipoles_sumwTemp.jpeg'.format(field, field, ccd_name[ccd_num]), bbox_inches='tight')
    plt.show()

 
    if do_zogy:
        zogy = zogy_lc(repo, collection_calexp, collection_coadd, ra, dec, ccd_num, visits, rd_aux, instrument = 'DECam', plot_diffexp=plot_zogy_stamps, plot_coadd = plot_coadd, cutout=cutout)
        print(zogy)
        z_flux = zogy.flux
        z_ferr = zogy.flux_err
        plt.errorbar(zogy.dates, z_flux, yerr=z_ferr, capsize=4, fmt='s', label ='ZOGY Cáceres-Burgos', color='orange', ls ='dotted')

                


    #area_source  = np.pi * r_aux**2
    #area_annuli = np.pi * (5*r_aux)**2 - np.pi * (2*r_aux)**2 
    source_of_interest = pd.DataFrame()
    source_of_interest['dates'] = dates_aux # dates 
    source_of_interest['flux_unscaled'] = np.array(Fluxes_unscaled) # unscaled flux template 
    source_of_interest['flux_err_unscaled'] = np.array(Fluxes_err_unscaled) # unscaled error flux template 
    source_of_interest['flux_nJy'] = np.array(Fluxes_njsky) # difference flux with fixed aperture 
    source_of_interest['fluxerr_nJy'] = np.array(Fluxeserr_njsky) # difference flux error with fixed aperture
    source_of_interest['fluxFs_nJy'] = np.array(FluxesFs_njsky) # difference flux with aperture as a factor 2 of seeing
    source_of_interest['fluxerrFs_nJy'] = np.array(FluxeserrFs_njsky) # difference flux error with a factor 2 of seeing
    
    ### adding the 5 differerent apertures:

    source_of_interest['flux0p5_nJy'] = np.array(Fluxes0p5_njsky) # difference flux with fixed aperture * factor 0.5 
    source_of_interest['fluxerr0p5_nJy'] = np.array(Fluxeserr0p5_njsky) # difference flux error with fixed aperture * factor 0.5 
    source_of_interest['flux0p75_nJy'] = np.array(Fluxes0p75_njsky) # difference flux with fixed aperture * factor 0.75 
    source_of_interest['fluxerr0p75_nJy'] = np.array(Fluxeserr0p75_njsky) # difference flux error with fixed aperture * factor 0.75 
    source_of_interest['flux1_nJy'] = np.array(Fluxes1_njsky) # difference flux with fixed aperture * factor 1
    source_of_interest['fluxerr1_nJy'] = np.array(Fluxeserr1_njsky) # difference flux error with fixed aperture * factor 1
    source_of_interest['flux1p25_nJy'] = np.array(Fluxes1p25_njsky) # difference flux with fixed aperture * factor 1.25 
    source_of_interest['fluxerr1p25_nJy'] = np.array(Fluxeserr1p25_njsky) # difference flux erro with fixed aperture * factor 1.25 
    source_of_interest['flux1p5_nJy'] = np.array(Fluxes1p5_njsky) # difference flux with fixed aperture * factor 1.5 
    source_of_interest['fluxerr1p5_nJy'] = np.array(Fluxeserr1p5_njsky) # difference flux error with fixed aperture * factor 1.5 
    source_of_interest['fluxDyn0p75_nJy'] = np.array(Fluxes0p75dyn_njsky)
    source_of_interest['fluxerrDyn0p75_nJy'] = np.array(Fluxeserr0p75dyn_njsky)
    ######################################
    if do_convolution:
        source_of_interest['fluxConv_cal_nJy'] = np.array(FluxConv_njsky)
        source_of_interest['fluxerrConv_cal_nJy'] = np.array(FluxerrConv_njsky)
    
    source_of_interest['flux_nJy_coadd'] = np.array(Fluxes_njsky_coadd) # template flux 
    source_of_interest['fluxerr_nJy_coadd'] = np.array(Fluxeserr_njsky_coadd) # template flux error
    source_of_interest['flux_nJy_cal'] = np.array(Fluxes_cal) # science flux
    source_of_interest['fluxerr_nJy_cal'] = np.array(Fluxeserr_cal) # science flux error
    source_of_interest['Mg_coadd'] = Mag_coadd
    source_of_interest['Mg_err_coadd'] = Magerr_coadd
    source_of_interest['Exptimes'] = ExpTimes
    
    source_of_interest['Mg'] = Mag 
    source_of_interest['Mg_err'] = Magerr
    source_of_interest['visit'] = visits_aux

    source_of_interest['calib'] = calib_lsst
    source_of_interest['calibErr'] = calib_lsst_err

    source_of_interest = source_of_interest.sort_values(by='dates')

    #if sfx=='mag':
    #    plt.errorbar(source_of_interest.dates, source_of_interest.Mg - np.median(source_of_interest.Mg), yerr = source_of_interest.Mg_err, capsize=4, fmt='s', label ='AL Cáceres-Burgos coadd + diff', color='#0827F5', ls ='dotted')
    #    plt.errorbar(source_of_interest.dates, source_of_interest.Mg_coadd - np.median(source_of_interest.Mg_coadd), yerr = source_of_interest.Mg_err_coadd, capsize=4, fmt='s', label ='AL Cáceres-Burgos coadd', color='#082785', ls ='dotted')
    #    plt.ylabel('Excess AB magnitude', fontsize=15)
    
    ## template and science image: 
    
    '''
    plt.figure(figsize=(10,6))
    plt.title('Aperture radii: {}", source {}'.format(r_science, title), fontsize=15)
    plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy_coadd - np.median(source_of_interest.flux_nJy_coadd), yerr = source_of_interest.fluxerr_nJy_coadd, capsize=4, fmt='s', label ='template', color='red', ls ='dotted')
    #plt.fill_between(source_of_interest.dates, stars_science_mean_byEpoch-stars_science_rms_byEpoch,  stars_science_mean_byEpoch+stars_science_rms_byEpoch, alpha=0.3, label = 'stars 1-sigma dev')
    if do_lc_stars:
        plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy_cal - np.median(source_of_interest.flux_nJy_cal), yerr = np.sqrt(source_of_interest.fluxerr_nJy_cal**2 +stars_science_sigma_byEpoch**2), capsize=4, fmt='s', label ='science', color='blue', ls ='dotted')
        plt.fill_between(source_of_interest.dates, stars_science_mean_byEpoch - stars_science_sigma_byEpoch, stars_science_mean_byEpoch + stars_science_sigma_byEpoch, alpha=0.5, label = 'stars 1-sigma dev', color= lilac)
    
    plt.legend()
    plt.ylabel('Excess flux nJy', fontsize=15 )
    if save and save_as!='':
        plt.savefig('{}/LSST_notebooks/light_curves/{}/{}_{}_{}_{}_template_science.jpeg'.format(main_root, field, field, ccd_name[ccd_num], save_as, sfx), bbox_inches='tight')
    
    '''

    #Difference
    plt.show()
    plt.figure(figsize=(10,6))
    plt.title('Difference Light curves', fontsize=17)
    #if do_lc_stars:
    #    plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy, yerr = np.sqrt(source_of_interest.fluxerr_nJy**2 + stars_diff_sigma_byEpoch**2), capsize=4, fmt='s', label ='Fixed aperture of {}"'.format(r_diff), color='magenta', ls ='dotted')
    
    ## plots of the 5 different apertures
    
    norm = mpl.colors.Normalize(vmin=0,vmax=1.5)
    c_m = mpl.cm.magma

    # create a ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    #T = np.linspace(4.5,20,40)
    #source_of_interest['flux_nJy_cal'] 
    #plt.errorbar(source_of_interest.dates, source_of_interest.flux0p5_nJy, yerr = source_of_interest.fluxerr0p5_nJy, capsize=4, fmt='s', label ='Fixed aperture of {}" * 0.5'.format(r_diff), color=s_m.to_rgba(0.5), ls ='dotted')
    
    #plt.errorbar(source_of_interest.dates, source_of_interest.flux0p75_nJy, yerr = source_of_interest.fluxerr0p75_nJy, capsize=4, fmt='s', color=s_m.to_rgba(0.75), ls ='dotted') # , label ='Fixed aperture of {}" * 0.75'.format(r_diff)
    #plt.errorbar(source_of_interest.dates, source_of_interest.fluxDyn0p75_nJy, yerr = source_of_interest.fluxerrDyn0p75_nJy, capsize=4, fmt='s', color = 'blue', ls='dotted') #, label = 'seeing * 0.75'
    plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), source_of_interest.flux1_nJy, yerr = source_of_interest.fluxerr1_nJy, capsize=4, fmt='s', color='k', ls ='dotted', label = 'diff image radii = {}'.format(r_diff)) # , label ='Fixed aperture of {}" * 1'.format(r_diff)
    
    if do_convolution:
        plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), source_of_interest.fluxConv_cal_nJy - np.mean(source_of_interest.fluxConv_cal_nJy), yerr=source_of_interest.fluxerrConv_cal_nJy, capsize=4, fmt='s', label='science after conv radii = {}'.format(fixed_radii * arcsec_to_pixel), color='m', ls='dotted')

    #plt.errorbar(source_of_interest.dates, source_of_interest.flux1p25_nJy, yerr = source_of_interest.fluxerr1p25_nJy, capsize=4, fmt='s', label ='Fixed aperture of {}" * 1.25'.format(r_diff), color=s_m.to_rgba(1.25), ls ='dotted')
    #plt.errorbar(source_of_interest.dates, source_of_interest.flux1p5_nJy, yerr = source_of_interest.fluxerr1p5_nJy, capsize=4, fmt='s', label ='Fixed aperture of {}" *1.5'.format(r_diff), color=s_m.to_rgba(1.5), ls ='dotted')
    
  
       
    if do_lc_stars:
    #   plt.errorbar(source_of_interest.dates, source_of_interest.fluxFs_nJy, yerr = np.sqrt(source_of_interest.fluxerrFs_nJy**2 + stars_diff_sigma_byEpoch**2), capsize=4, fmt='s', label ='Aperture {}*FWHM'.format(factor), color='blue', ls ='dotted')
        plt.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_diff_1sigmalow_byEpoch, stars_diff_1sigmaupp_byEpoch, alpha=0.1, color='m', label = 'stars 1-2 $\sigma$ dev') #
        plt.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_diff_2sigmalow_byEpoch, stars_diff_2sigmaupp_byEpoch, alpha=0.1, color='m') #, label = 'stars 2-$\sigma$ dev'
        
        #plt.fill_between(source_of_interest.dates, stars_lsst_mean- stars_lsst_std, stars_lsst_mean + stars_lsst_std, alpha=0.5, label = 'stars 1-sigma dev', color= lilac)    
    
    plt.ylabel('Flux [nJy]', fontsize=15 )
    plt.xlabel('MJD - {}'.format(int(min(dates_aux))), fontsize=15)
    
    if SIBLING!=None:
        x, y, yerr = compare_to(SIBLING, sfx='mag', factor=0.75)
        f, ferr = pc.ABMagToFlux(y, yerr)# in nJy
        #mfactor = 5e-10
        plt.errorbar(x-min(x), f*mfactor -  np.mean(f*mfactor), yerr=ferr*mfactor,  capsize=4, fmt='^', ecolor='orange', color='orange', label='Martinez-Palomera et al. 2020 nJy*' + str(mfactor), ls ='dotted')
    plt.legend(frameon=False)

    #plt.title('Aperture radii: {}", source {}'.format(r_in_arcsec, title), fontsize=15)

    if sparse_obs:
        plt.xscale('log')

    if save and save_as=='':
        plt.savefig('{}/LSST_notebooks/light_curves/ra_{}_dec_{}_{}.jpeg'.format(main_root, ra,dec,sfx), bbox_inches='tight')
    
    
    plt.show()

     #
    plt.figure(figsize=(10,6))
    plt.title('Science Light curves - median', fontsize=17)
    # science
    if do_lc_stars:
    #    plt.errorbar(source_of_interest.dates, source_of_interest.fluxFs_nJy, yerr = np.sqrt(source_of_interest.fluxerrFs_nJy**2 + stars_diff_sigma_byEpoch**2), capsize=4, fmt='s', label ='Aperture {}*FWHM'.format(factor), color='blue', ls ='dotted')
        plt.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_science_1sigmalow_byEpoch, stars_science_1sigmaupp_byEpoch, alpha=0.1, color='m', label = 'stars 1-2 $\sigma$ dev') #
        plt.fill_between(source_of_interest.dates - min(source_of_interest.dates), stars_science_2sigmalow_byEpoch,  stars_science_2sigmaupp_byEpoch, alpha=0.1, color='m') #, label = 'stars 2-$\sigma$ dev'

    if do_convolution:
       plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), source_of_interest.fluxConv_cal_nJy - np.mean(source_of_interest.fluxConv_cal_nJy), yerr=source_of_interest.fluxerrConv_cal_nJy, capsize=4, fmt='s', label='science after conv radii = {}'.format(fixed_radii * arcsec_to_pixel), color='m', ls='dotted')
    
    plt.errorbar(source_of_interest.dates - min(source_of_interest.dates), source_of_interest.flux_nJy_cal - np.mean(source_of_interest.flux_nJy_cal), yerr = source_of_interest.fluxerr_nJy_cal, capsize=4, fmt='d', label ='science before conv radii = {}'.format(fixed_radii * arcsec_to_pixel), color='green', ls ='dotted')
    plt.ylabel('Flux [nJy]', fontsize=15 )
    plt.xlabel('MJD - {}'.format(int(min(dates_aux))), fontsize=15)
    plt.show()



    plt.figure(figsize=(10,6))
    for k in range(len(visits_aux)):
        norm = mpl.colors.Normalize(vmin=min(Seeing),vmax=max(Seeing))
        c_m = mpl.cm.magma

        # create a ScalarMappable and initialize a data structure
        s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        plt.plot(np.linspace(0.05, 6 , 15), profiles['{}'.format(visits_aux[k])], label='{}'.format(visits_aux[k]), color=s_m.to_rgba(Seeing[k]))
        plt.axvline(1, ls='--', color='gray')
    
    plt.colorbar(s_m)
    plt.xlabel('arcseconds', fontsize=17)
    plt.ylabel('Normalized flux counts', fontsize=17)
    plt.legend(frameon=False)
    plt.show()
        
    #
    

    # This line below plots the stamps of the source as a single figure for all epochs available!
    if do_lc_stars:
        #print('data_diff: ', Data_diff)
        stamps_and_LC_plot_forPaper(Data_science, Data_convol, Data_diff, Data_coadd, coords_science, coords_convol, coords_coadd, source_of_interest, Results_star, visits, KERNEL, Seeing, r_diff=r_diff, r_science=r_science, SIBLING = SIBLING, cut_aux=cut_aux, first_mjd=int(min(dates_aux)), name_to_save=name_to_save)
        #Calib_Diff_and_Coadd_one_plot_cropped(repo, collection_diff, ra, dec, list(source_of_interest.visit), ccd_num, cutout=cutout, s=rd_aux, save_stamps=save_stamps, save_as=save_as+ '_stamps')
        #Calib_Diff_and_Coadd_plot_cropped_oneFig_astropy(repo, collection_diff, RA_source, DEC_source, list(source_of_interest.visit), ccd_num, s=rd_aux, sd=rd_aux, cutout=cutout, field=field, name=title)
        #plt.show()
    
    if hist:
        norm = matplotlib.colors.Normalize(vmin=0,vmax=len(visits_aux))
        c_m = matplotlib.cm.cool

        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(0,len(visits_aux),len(visits_aux))
        j = 0
        for key in stats:
            plt.hist(stats[key], bins=500, histtype='step', log=True, label=key, alpha=0.5, color = s_m.to_rgba(T[j]))
            print(np.median(stats[key]))
            plt.axvline(np.median(stats[key]), color = s_m.to_rgba(T[j]), alpha=0.5, linestyle='--')
            j+=1
        plt.legend()
        plt.xlim(-1, 1)
        plt.show()
    print('esto tiene un largo de: ',len(source_of_interest))
    print(source_of_interest)

    if len(name_to_save)>2:
        source_of_interest.to_csv(name_to_save+'_LCs_points.csv')
    return source_of_interest


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
        arcsec_to_pixel = 0.2626 #arcsec/pixel
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
            arcsec_to_pixel = 0.2626 #arcsec/pixel
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
        arcsec_to_pixel = 0.2626 #arcsec/pixel
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
        sum_stars = np.zeros((2*second_cutout + 1, 2*second_cutout + 1))

        for ra, dec, fl in zip(ra_stars, dec_stars, inst_star_flux):

            obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
            afwDisplay.setDefaultMaskTransparency(100)
            afwDisplay.setDefaultBackend('matplotlib')
            x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
            calexp_cutout = calexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2 + 1, cutout*2 + 1))
            thresh = fl*0.6

            objects = sep.extract(calexp_cutout.image.array, thresh, minarea=30)

            if len(objects)!=1:
                continue
            else:
                RAs.append(ra)
                DECs.append(dec)
                
                x_pix_cen = round(objects[0]['x'],0)
                y_pix_cen = round(objects[0]['y'],0)
                
                calexp_cutout_to_use = calexp_cutout.image.array[int(y_pix_cen-second_cutout) + 1 :int(y_pix_cen+second_cutout) + 1, int(x_pix_cen-second_cutout) + 1:int(x_pix_cen+second_cutout) + 1]
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
    #arcsec_to_pixel = 0.2626 #arcsec/pixel
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
    result = scipy.signal.convolve2d(input_array, kernel, mode='same', boundary='fill', fillvalue=0)
    
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


def objective_function(alpha, image_kernel, worst_kernel):
    
    airy_kernel = AiryDisk2DKernel(alpha).array
    
    target_kernel = convolve(image_kernel, airy_kernel)
    max_size = target_kernel.shape[0]
    worst_kernel_resized = np.zeros((max_size, max_size))
    worst_kernel_resized[:worst_kernel.shape[0], :worst_kernel.shape[1]] = worst_kernel#airy_kernel.array
    
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



def do_convolution_image(repo, main_root, visit, ccd_num, collection_calexp, ra, dec, worst_seeing_visit=None, mode='Eridanus', type_kernel='mine'):   
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
    #repo = '/home/pcaceres/data_hits'
    print('About to do the convolutionnn...')
    butler = Butler(repo)
    #rint(butler)
    field = collection_calexp.split('/')[-1]
    calexp = butler.get('calexp', visit=visit, detector=ccd_num, collections=collection_calexp, instrument='DECam') 
    calexp_array = np.asarray(calexp.image.array, dtype='float')
    var_calexp_array = np.asarray(calexp.variance.array, dtype='float')
    #print('calexp array: ', calexp_array)
    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    wcs = calexp.getWcs()
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
    convolved_image = {}    
    
    #directory = '{}/LSST_notebooks/convolved_images/convolved_image_{}_{}_{}_photutils_kernel'.format(main_root, visit, field, ccd_num)
    #print('directory to search for convolved images: ', directory)
    #file = '{}.pickle'.format(directory)
 
    if visit == worst_seeing_visit:
        
        convolved_image[visit] = calexp_array
        convolved_image['{}_variance'.format(visit)] = var_calexp_array
        convolved_image['{}_kernel'.format(visit)] = np.zeros((25,25))
        return convolved_image, 0 

    # loading kernel from worst image 
    worst_calexp = butler.get('calexp', visit=worst_seeing_visit, detector=ccd_num, collections=collection_calexp, instrument='DECam')
    worst_psf =  worst_calexp.getPsf()
    worst_seeing = worst_psf.computeShape(worst_psf.getAveragePosition()).getDeterminantRadius()
    
    obj_pos_2d = lsst.geom.Point2D(ra, dec)
    wimageKernel = worst_psf.computeKernelImage(obj_pos_2d)
    
    # loading kernel of current image 
    psf = calexp.getPsf()
    imageKernel = psf.computeKernelImage(obj_pos_2d)    

    im = imageKernel.array # kernel array of image 
    wim = wimageKernel.array # kernel array of worst image 

    if (type_kernel=='mine'):

        if (mode == 'Eridanus'):
            alpha = 0.3
            beta = 0.25
            kernel = create_matching_kernel(im, wim, SplitCosineBellWindow(alpha=alpha, beta=beta))   

        if (mode == 'HITS'):
            
            initial_alpha = 1.5 
            result = minimize(objective_function, initial_alpha, args=(im, wim), method='Nelder-Mead', bounds=[(0, None)])    
            optimized_alpha = result.x[0]
            print('alpha = ', optimized_alpha)
            kernel = AiryDisk2DKernel(optimized_alpha).array
    else:

        kernel = Panchos_kernel(repo, collection_calexp, ccd_num, visit, worst_seeing_visit).solfilter
    print('using the configuration for: ', mode)    

    # doing convolution 

    conv = custom_convolve(calexp_array, kernel) 
    conv_variance = custom_convolve(var_calexp_array, kernel**2)

    convolved_image[visit] = conv
    convolved_image['{}_variance'.format(visit)] = conv_variance
    convolved_image['{}_kernel'.format(visit)] = kernel
    print(' I did the convolution !!!!')
    return convolved_image, 0


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
    if seed[0] == True:
        random.seed(seed[1])
    
    len_table = len(result[0])
    random_indexes = random.sample(range(len_table), n)
    
    table = result[0][random_indexes]


    return table

def Select_table_from_one_calib_exposure(repo, visit, ccdnum, collection_calexp, stars=True, s_to_n_star = 5):
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
    if stars:

        mask = (src_pandas['calib_photometry_used'] == True) & (src_pandas['base_PsfFlux_instFlux']/src_pandas['base_PsfFlux_instFluxErr'] > s_to_n_star)
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
        if tp == 'after_ID':
            Dict_tables['{}'.format(visits[i])] = Select_table_from_one_exposure(repo, visits[i], ccdnum, collection_diff, well_subtracted=well_subtracted)
        if tp == 'before_ID':
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
    columns_picked = ['src_id', 'coord_ra', 'coord_dec', 'coord_ra_ddegrees', 'coord_dec_ddegrees', 'base_CircularApertureFlux_3_0_instFlux', 'base_PsfFlux_instFlux', 'base_PsfFlux_mag', 'base_PsfFlux_magErr','slot_PsfFlux_mag', 'phot_calib_mean']
    for key in dictio:
        if key == '{}'.format(visits[0]):
            big_table = dictio[key].to_pandas()[columns_picked]
            new_column_names = ['{}_{}'.format(c, key) for c in columns_picked]
            big_table.columns = new_column_names
            big_table['coord_ra_trunc'] = [Truncate(f, 5) for f in np.array(big_table['coord_ra_{}'.format(visits[0])])]
            big_table['coord_dec_trunc'] = [Truncate(f, 5) for f in np.array(big_table['coord_dec_{}'.format(visits[0])])]
            #big_table['photo_calib_{}'.format(visits[0])] = photom_calib[0]
            #big_table = big_table.rename(columns = {'coord_ra_trunc_{}'.format():'src_id'})

            big_table['circ_aperture_to_nJy_{}'.format(visits[0])] = big_table['phot_calib_mean_{}'.format(visits[0])] * big_table['base_CircularApertureFlux_3_0_instFlux_{}'.format(visits[0])]
            big_table['psf_flux_to_nJy_{}'.format(key)] = big_table['phot_calib_mean_{}'.format(key)] * big_table['base_PsfFlux_instFlux_{}'.format(key)]
            big_table['psf_flux_to_mag_{}'.format(key)] = [pc.FluxJyToABMag(f*1e-9)[0] for f in big_table['psf_flux_to_nJy_{}'.format(key)]]
            i+=1
        else:
            table = dictio[key].to_pandas()[columns_picked]
            new_column_names = ['{}_{}'.format(c, key) for c in columns_picked]
            table.columns = new_column_names
            table['coord_ra_trunc'] = [Truncate(f, 5) for f in np.array(table['coord_ra_{}'.format(key)])]
            table['coord_dec_trunc'] = [Truncate(f, 5) for f in np.array(table['coord_dec_{}'.format(key)])]
            #table['photo_calib_{}'.format(key)] = photom_calib[i]
            table['circ_aperture_to_nJy_{}'.format(key)] = table['phot_calib_mean_{}'.format(key)] * table['base_CircularApertureFlux_3_0_instFlux_{}'.format(key)]
            table['psf_flux_to_nJy_{}'.format(key)] = table['phot_calib_mean_{}'.format(key)] * table['base_PsfFlux_instFlux_{}'.format(key)]
            table['psf_flux_to_mag_{}'.format(key)] = [pc.FluxJyToABMag(f*1e-9)[0] for f in table['psf_flux_to_nJy_{}'.format(key)]]
            
            big_table = pd.merge(big_table, table, on=['coord_ra_trunc', 'coord_dec_trunc'], how='outer')

            i+=1
    big_table
    
    return big_table


def Inter_Join_Tables_from_LSST(repo, visits, ccdnum, collection_diff, well_subtracted =False, tp='after_ID', save=False):
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
    phot_table = phot_table.drop_duplicates('coord_ra_trunc')
    phot_table = phot_table.reset_index()
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

def compare_to(directory, sfx, factor, beforeDate=57072):
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

    SIBLING = directory

    if SIBLING!=None and SIBLING[0:24]=="/home/pcaceres/Jorge_LCs" and type(SIBLING)==str:
        Jorge_LC = pd.read_csv(SIBLING, header=5)
        Jorge_LC = Jorge_LC[Jorge_LC['mjd']<beforeDate] 
        sfx_aux = 'mag'
        if factor==0.5:
            
            param = Jorge_LC['aperture_{}_0'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_0'.format(sfx)]
            #median_jorge=np.median(param)
            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]
            #    param = 
            #    median_jorge=0 
            #
            #std = np.norm(Jorge_LC.aperture_flx_0)
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_0 - mean, yerr=Jorge_LC.aperture_flx_err_0,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            x = Jorge_LC.mjd# - min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr
        if factor==0.75:

            
            param = Jorge_LC['aperture_{}_1'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_1'.format(sfx)]
            median_jorge = np.median(param)
            mean = np.mean(param)
            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]

            x = Jorge_LC.mjd#- min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr

        if factor==1:
            
            param = Jorge_LC['aperture_{}_2'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_2'.format(sfx)]
            mean = np.mean(param)
            norm = np.linalg.norm(np.array(param))
            median_jorge = np.median(param)

            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]

            x = Jorge_LC.mjd#- min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_2 - mean, yerr=Jorge_LC.aperture_flx_err_2,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
        if factor==1.25:
            #std = np.std(Jorge_LC.aperture_flx_3)
            param = Jorge_LC['aperture_{}_3'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_3'.format(sfx)]
            mean = np.mean(param)
            median_jorge= np.median(param)
            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]

            x = Jorge_LC.mjd #- min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_3 - mean, yerr=Jorge_LC.aperture_flx_err_3,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
        if factor==1.5:
            #std = np.std(Jorge_LC.aperture_flx_4)
            
            param = Jorge_LC['aperture_{}_4'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_4'.format(sfx)]             
            norm = np.linalg.norm(np.array(param))
            median_jorge= np.median(param)
            mean = np.mean(param)
            
            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]

            x = Jorge_LC.mjd #- min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr
    
    HiTS = directory 

    if HiTS!=None and HiTS[0:24]=="/home/pcaceres/HiTS_LCs/" and type(HiTS)==str:
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
    sigma_rms_sq - sigma_rms_sq_err

    """
    mean_mag = np.mean(mag)
    nobs = len(mag)
    sigma_rms_sq = 1/(nobs * mean_mag**2) * np.sum((mag - mean_mag)**2 - magerr**2) 
    sd = 1/nobs * np.sum((((mag - mean_mag)**2 - magerr**2) - sigma_rms_sq * mean_mag**2 )**2)
    sigma_rms_sq_err = sd / (mean_mag**2 * nobs**1/2)

    return sigma_rms_sq - sigma_rms_sq_err


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
    arcsec_to_pixel = 0.2626

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


def flux_profile_array(exp_array, x_pix, y_pix, rmin, rmax, title_plot = '', save_plot =False, field=None, name =None):
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

    fluxes_ap = []
    apertures = np.linspace(rmin, rmax, 15)
    arcsec_to_pixel = 0.2626

    for r in apertures:
        r/=arcsec_to_pixel
        f, ferr, flag = sep.sum_circle(exp_array, [x_pix], [y_pix], r)
        fluxes_ap.append(f[0])
        #fluxes_ap_err.append(ferr[0])
    fluxes_ap /= fluxes_ap[-1]

    if save_plot:
        pass
    return fluxes_ap
