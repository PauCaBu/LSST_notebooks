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
from sklearn import *
import photometric_calib as pc
from astropy.table import Table, join, Column
import decimal
import seaborn as sns


bblue='#0827F5'
dark_purple = '#2B018E'

detector_nomenclature= {'S29':1, 'S30':2, 'S31':3, 'S28':7, 'S27':6, 'S26':5, 'S25':4, 'S24':12, 'S23':11, 'S22':10, 'S21':9, 'S20':8, 'S19':18, 'S18':17, 'S17':16, 'S16':15, 'S15':14, 'S14':13, 'S13':24, 'S12':23, 'S11':22, 'S10':21, 'S9':20,'S8':19, 'S7':31, 'S6':30, 'S5':29, 'S4':28, 'S3':27, 'S2':26, 'S1':25, 'N29':60, 'N30':61, 'N31':62, 'N28':59, 'N27':58, 'N26':57, 'N25':56, 'N24':55, 'N23':54, 'N22':53, 'N21':52, 'N20':51, 'N19':50, 'N18':49, 'N17':48, 'N16':47, 'N15':46, 'N14':45, 'N13':44, 'N12':43, 'N11':42, 'N10':41, 'N9':40,'N8':39, 'N7':38, 'N6':37, 'N5':36, 'N4':35, 'N3':34, 'N2':33, 'N1':32 }
ccd_name = dict(zip(detector_nomenclature.values(), detector_nomenclature.keys()))
sibling_allcand = pd.read_csv('/home/jahumada/testdata_hits/SIBLING_allcand.csv', index_col=0)
Blind15A_26_magzero_outputs = pd.read_csv('/home/jahumada/testdata_hits/LSST_notebooks/output_magzeros.csv')
main_path = '/home/jahumada/testdata_hits/LSST_notebooks/'


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
        plt.title('Calexp Image and Source Catalog')

        fig.add_subplot(1,2,2)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[1].scale('asinh', -10,10)
        stamp_display[1].mtv(diffexp_cutout.maskedImage)


        stamp_display[1].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)

        for src in diffexp_cat:
            stamp_display[1].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
            #if np.fabs(x_pix - src.getX())<cutout and np.fabs(y_pix - src.getY())<cutout:
            #    print('from catalog that is within the : {} {}'.format(src.getX(), src.getY()))
        plt.title('Diffexp Image and Source Catalog')

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
        plt.title('Calexp Image and Source Catalog')

        fig.add_subplot(1,3,2)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[1].scale('asinh', -10,10)
        stamp_display[1].mtv(diffexp_cutout.maskedImage)


        stamp_display[1].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)

        for src in diffexp_cat:
            stamp_display[1].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
            if np.fabs(x_pix - src.getX())<cutout and np.fabs(y_pix - src.getY())<cutout:
                print('from catalog that is within the : {} {}'.format(src.getX(), src.getY()))
        plt.title('Diffexp Image and Source Catalog')

        fig.add_subplot(1,3,3)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[2].scale('asinh', -10, 10)
        stamp_display[2].mtv(coadd_cutout.maskedImage)
        stamp_display[2].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)
        
        plt.title('Coadd template')

        

        plt.tight_layout()
        plt.show()
        
        
    return


def Calib_Diff_and_Coadd_plot_cropped_astropy(repo, collection_diff, ra, dec, visits, ccd_num, cutout=40, s=20):
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

        calexp_cutout_arr = np.asarray(calexp_cutout.image.array, dtype='float')
        diffexp_cutout_arr = np.asarray(diffexp_cutout.image.array, dtype='float')
        coadd_cutout_arr = np.asarray(coadd_cutout.image.array, dtype='float')

        fig = plt.figure(figsize=(16, 5))

        stamp_display = []

        fig.add_subplot(1,3,1)
        plt.imshow(calexp_cutout_arr)
        plt.colorbar()
        plt.contour(calexp_cutout_arr, levels=np.logspace(1.3, 2.5, 10), colors ='white', alpha=0.5)
        circle = plt.Circle((x_half_width, y_half_width), radius = s, color='red', fill = False)
        plt.gca().add_patch(circle)
        
        plt.title('Calexp Image and Source Catalog')

        fig.add_subplot(1,3,2)

        plt.imshow(diffexp_cutout_arr)
        plt.colorbar()
        plt.contour(diffexp_cutout_arr, levels=np.logspace(1.3, 2.2, 10), colors ='white', alpha=0.5)

        #plt.scatter(x_half_width, y_half_width, s=np.pi*s**2, facecolors='none', edgecolors='red')
        circle = plt.Circle((x_half_width, y_half_width), radius = s, color='red', fill = False)
        plt.gca().add_patch(circle)

        plt.title('Diffexp Image and Source Catalog')

        fig.add_subplot(1,3,3)
        plt.imshow(coadd_cutout_arr)
        plt.colorbar()
        plt.contour(coadd_cutout_arr, levels=np.logspace(1.3, 2.5, 10), colors ='white', alpha=0.5)

        #plt.scatter(x_half_width, y_half_width, s=np.pi*s**2, facecolors='none', edgecolors='red')
        circle = plt.Circle((x_half_width, y_half_width), radius = s, color='red', fill = False)
        plt.gca().add_patch(circle)
        plt.title('Coadd template')

        plt.tight_layout()
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
        plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}.jpeg'.format(save_as))

        
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

    f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, sharex=True, figsize=(10,6))
    ax1.set_title(title_plot, fontsize=17)
    ax1.imshow(exp_cutout_array)
    ax1.scatter(x_length, y_length, color='red', s=20)
    ax2.bar(range(len(adu_values)), fluxes, color = 'm')
    ax2.set_ylabel('Flux [nJy]', fontsize=17)
    ax2.set_xlabel('x-axis pixels', fontsize=17)
    ax1.set_ylabel('y-axis pixels', fontsize=17)
    f.subplots_adjust(hspace=0)
    if save_plot:
        f.savefig('light_curves/{}/{}.jpeg'.format(field, name), bbox_inches='tight')
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
        plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}.jpeg'.format(save_as), bbox_inches='tight')

        
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

    dates = dates - min(dates)
    zipped = zip(dates, visits)
    res = sorted(zipped, key = lambda x: x[0])

    dates_aux, visits_aux = zip(*list(res))

    return dates_aux, visits_aux


def get_light_curve(repo, visits, collection_diff, collection_calexp, ccd_num, ra, dec, r, field='', factor=0.75, cutout=40, save=False, title='', hist=False, sparse_obs=False, SIBLING=None, save_as='', do_lc_stars = False, nstars=10, seedstars=200, save_lc_stars = False, show_stamps=True, show_star_stamps=True, factor_star = 2, correct_coord=False, bs=531, box=100, do_zogy=False, collection_coadd=None, plot_zogy_stamps=False, plot_coadd=False, instrument='DECam', sfx='flx', save_stamps=False, well_subtracted=False, config='SIBLING', verbose=False, tp='after_ID'):
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

    Fluxes_njsky_coadd = []
    Fluxeserr_njsky_coadd = []
    
    stars = pd.DataFrame()
    
    flags = []
    
    stats = {}
    pixel_to_arcsec = 0.2626 #arcsec/pixel, value from Manual of NOAO - DECam
    
    r_in_arcsec = r 
    #if type(r) != str:
    #    r_aux = r/pixel_to_arcsec

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

    dates_aux, visits_aux = Order_Visits_by_Date(repo, visits, ccd_num, collection_diff)

    for i in range(len(visits_aux)):
        #print('hello im a branch jiji')
        zero_set_aux = zero_set
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        calexp = butler.get('calexp', visit=visits_aux[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 
        coadd = butler.get('goodSeeingDiff_matchedExp', visit=visits_aux[i], detector=ccd_num, collections=collection_diff, instrument='DECam')


        wcs_coadd = coadd.getWcs()
        wcs = diffexp.getWcs()
        px = 2048
        py = 4096

        data = np.asarray(diffexp.image.array, dtype='float')
        data_cal = np.asarray(calexp.image.array, dtype='float')
        data_coadd = np.asarray(coadd.image.array, dtype='float')


        
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        x_pix_coadd, y_pix_coadd = wcs_coadd.skyToPixel(obj_pos_lsst)
        
        wcs = diffexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)

        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        psf = diffexp.getPsf() 
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm # in pixels! 
        r_aux=r

        psf2 = coadd.getPsf() 
        seeing2 = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm

        #* pixel_to_arcsec
        if r == 'seeing':
            r_aux = seeing * factor
        else:
            r_aux = r/pixel_to_arcsec

        Seeing.append(seeing)

        if correct_coord and i==0:

            print('before centroid correction: xpix, y_pix: {} {} '.format(x_pix, y_pix))

            sub_data = data_cal[int(y_pix)-box:box+int(y_pix),int(x_pix)-box:box+int(x_pix)]
            
            #plt.hist(sub_data.flatten(), bins='auto')
            #plt.xlim(sub_data.min(), 500)
            #plt.show()

            

            sub_data = sub_data.copy(order='C')
            objects = sep.extract(sub_data, 100, minarea=10)
            
            obj = Select_largest_flux(sub_data, objects)
            ox = objects[:]['x']
            xc = float(obj['x'][0]) # x coordinate pixel
            yc = float(obj['y'][0]) # y coordinate pixel
            
            x_pix = xc + int(x_pix)-box 
            y_pix = yc + int(y_pix)-box

            print('after centroid correction: xpix, y_pix: {} {} '.format(x_pix, y_pix))
            ra_cor, dec_cor =  wcs.pixelToSkyArray([x_pix], [y_pix], degrees=True)
            ra = ra_cor[0]
            dec = dec_cor[0]
            
            #fig, ax = plt.subplots()
            
            plt.imshow(sub_data, norm=LogNorm(), cmap='viridis')
            plt.colorbar(ticks = [sub_data.min(), 0, 100, 500])
            plt.scatter(xc,yc, color='r', alpha=0.5, s = r_aux**2 * np.pi)
            print('coordinates xc, yc: ', xc, yc)
            plt.show()

        ra_center, dec_center = wcs.pixelToSkyArray([px/2], [py/2], degrees=True)
        ra_center = ra_center[0]
        dec_center = dec_center[0]
        #print('should be center of exposre')
        #print('RA center : {} DEC center : {}'.format(ra_center, dec_center))

        ExpTime = diffexp.getInfo().getVisitInfo().exposureTime 
        ExpTimes.append(ExpTime)
        #gain = 4
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        obj_pos_2d = lsst.geom.Point2D(ra, dec)
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)

        print('xpix, y_pix: {} {} '.format(x_pix, y_pix))
        
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
        

        print('Aperture radii: {} px'.format(r_aux))
        print('radii for the template: {} px'.format(seeing2*factor))

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

        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], r_aux, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_an, fluxerr_an, flag_an = sep.sum_circann(data, [x_pix], [y_pix], r_aux*2, r_aux*5, var = np.asarray(diffexp.variance.array, dtype='float'))
        flux_cal, fluxerr_cal, flag_cal = sep.sum_circle(data_cal, [x_pix], [y_pix], r_aux, var = np.asarray(calexp.variance.array, dtype='float'))
        flux_coadd, fluxerr_coadd, flag_coadd = sep.sum_circle(data_coadd, [x_pix], [y_pix], r_aux, var = np.asarray(coadd.variance.array, dtype='float'))
        flux_coadd_an, fluxerr_coadd_an, flag_coadd = sep.sum_circann(data_coadd, [x_pix], [y_pix], r_aux*2, r_aux*5, var = np.asarray(coadd.variance.array, dtype='float'))
        

        print('Coords: ra = {}, dec = {}'.format(ra,dec))
        print('visit : {}'.format(visits_aux[i]))
        
        if show_stamps:
            print('DATE: ', dates_aux[i])
            #Calib_and_Diff_plot_cropped(repo, collection_diff, collection_calexp, ra, dec, [visits_aux[i]], ccd_num, s=r)
            print('aperture that enters stamp plots: ', r_aux)
            Calib_Diff_and_Coadd_plot_cropped_astropy(repo, collection_diff, ra, dec, [visits_aux[i]], ccd_num, s=r_aux, cutout=cutout)
            values_across_source(calexp, ra, dec , x_length = r_aux, y_length=1.5, stat='median', title_plot='Calibrated exposure', save_plot =True, field=field, name='slit_science_{}_{}.jpeg'.format(save_as, sfx))
            values_across_source(diffexp, ra, dec , x_length = r_aux, y_length=1.5, stat='median', title_plot = 'Difference exposure', save_plot = True, field=field, name='slit_difference_{}_{}.jpeg'.format(save_as, sfx))
            
       

        

        expTime = float(calexp.getInfo().getVisitInfo().exposureTime)
        print('exposure Time: ', expTime)
        
        #calib_path = main_path + 'calibration/calibration_scaling_{}_{}_{}_fwhm.npz'.format(field, ccd_name[ccd_num], factor_star)
        
        #magzero_image = photocalib_coadd.instFluxToMagnitude(1) #pc.MagAtOneCountFlux(repo, visits[i], ccd_num, collection_diff) #float(row.magzero)
        


        print('calibration mean: ', calib_image)
        #calib = pc.DoCalibration(repo, visits_aux[i], ccd_num, collection_diff, config=config)
        #calib_mean = 1#calib[0]
        #calib_intercept = 0#calib[1]
        
        #my_calib.append(calib_mean)
        #my_calib_inter.append(calib_intercept)

        #if not os.path.isfile(calib_path):
        #    
        #    #print(calib_path, ' doesnt exist')
        #    if i == 0:
        #        calib = pc.DoCalibration(repo, visits_aux[0], ccd_num, collection_diff, config=config) 
        #        calib_mean0 = calib[0] #calib_mean
        #        calib_intercept0 = calib[1] #calib_intercept
        #    
        #    calib_rel = pc.DoRelativeCalibration(repo, visits_aux[0], calib_mean0, calib_intercept0, visits_aux[i], ccd_num, collection_diff, config=config) 
        #    calibRel_mean = calib_rel[0]
        #    calibRel_intercept = calib_rel[1]

        #    calib_relative.append(calibRel_mean)
        #    calib_relative_intercept.append(calibRel_intercept)
        #else:
        #    npzfile = np.load(calib_path)
        #    calib_relative = npzfile['x']
        #    calib_relative_intercept= npzfile['y'] #np.zeros(len(calib_relative)) #
        #    calibRel_mean = calib_relative[i]
        #    calibRel_intercept = calib_relative_intercept[i]

        #if i == 0:
            #flux_reference = flux_coadd[0]
            #fluxerr_reference = fluxerr_jsky_coadd 
            #calib_reference = calib_image #photocalib_coadd.getCalibrationMean()
            #magzero_reference = magzero_image
        
        #scaler = 1 # flux_reference/flux_coadd[0] * calib_reference/calib_image #10**(0.4*(magzero_reference - magzero_image))
        #scaling.append(scaler)

        #print('scaling factor between images: ', scaler)


        flux_physical = photocalib.instFluxToNanojansky(flux[0], fluxerr[0], obj_pos_2d)
        print('Flux and Flux error in nanoJansky: ',flux_physical)
        
        # Here I try my calibration: 

        flux_jsky = flux_physical.value
        fluxerr_jsky = flux_physical.error

        flux_physical_coadd = photocalib_coadd.instFluxToNanojansky(flux_coadd[0], fluxerr_coadd[0], obj_pos_2d)

        flux_jsky_coadd = flux_physical_coadd.value 
        fluxerr_jsky_coadd = flux_physical_coadd.error

        magzero.append(flux_jsky_coadd/flux_coadd[0])
        #scaler = flux_reference/flux_jsky_coadd
        #print('for now, we not apply scaling')
        #flux_jsky*=scaler 
        #flux_jsky_coadd*=scaler
        #fluxerr_jsky*=scaler
        #fluxerr_jsky_coadd*=scaler
        
        flux_physical_cal = photocalib_cal.instFluxToNanojansky(flux_cal[0], fluxerr_cal[0], obj_pos_2d)
        flux_jsky_cal = flux_physical_cal.value 
        fluxerr_jsky_cal = flux_physical_cal.error 


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
        #((-2.5*(fluxerr_jsky)/(flux_jsky *np.log(10)))**2 + (-2.5*(fluxerr_jsky_coadd)/(flux_jsky_coadd*np.log(10)))**2)**1/2 #mag_diff.error

        #zero_set_aux *= photocalib_cal.instFluxToNanojansky(flux[0], obj_pos_2d)/coadd_photocalib.instFluxToNanojansky(flux[0], obj_pos_2d)
        #zero_set_aux += photocalib_cal.getInstFluxAtZeroMagnitude()
        
        #
        #row = magzero_outputs[magzero_outputs['visit']==visits[i]]
        #print(row)

        
        #magzero_template = pc.MagAtOneCountFlux(repo, visits[i], ccd_num, collection_diff)

        #flux_at_zero_magnitude_img = photocalib.getInstFluxAtZeroMagnitude()
        #flux_at_zero_magnitude_img_cal = photocalib_cal.getInstFluxAtZeroMagnitude()
        #flux_at_zero_magnitude_img_coadd = photocalib_coadd.getInstFluxAtZeroMagnitude()
        
        #f_scaling = flux_at_zero_magnitude_ref/flux_at_zero_magnitude_img #
        #f_scaling_cal = flux_at_zero_magnitude_ref_cal/flux_at_zero_magnitude_img_cal #10**((magzero_reference - magzero_image)/-2.5)
        #f_scaling_coadd = flux_at_zero_magnitude_ref_coadd/flux_at_zero_magnitude_img_coadd #10**((magzero_reference - magzero_image)/-2.5)
        
        #phot = pc.mag_stars_calculation(repo, visits[i], ccd_num, collection_diff)
        #DF = phot['DataFrame']
        #seeing = np.unique(np.array(DF.seeing))[0]
        
        #alpha_scaling = phot['panstarss_counts']/phot['calcbyme_counts']
        #scale_coadd = phot['calcbyme_counts']/phot['calcbyme_counts_coadd']
        #print('alpha scaling: ', alpha_scaling)
        
        #f_scaling = alpha_scaling * expTime #10**((magzero_firstImage - magzero_image_i)/-2.5)
        #scaling.append(f_scaling)
        #scaling_coadd.append(scale_coadd)

        #flux_nJy = photocalib.instFluxToNanojansky(flux[0], fluxerr[0], obj_pos_2d).value
        #fluxerr_nJy = photocalib.instFluxToNanojansky(flux[0], fluxerr[0], obj_pos_2d).error

        print('flux before scaling: ', flux[0])
        #print('f scaling : ', f_scaling)
        #print('flux after scaling: ', flux[0]*f_scaling)
        #print('flux of source in template: ', flux_coadd[0]*f_scaling)

        if flux[0] > 1500 :
            print('This source is bad subtracted')

            pass

        Fluxes_unscaled.append(flux_coadd[0])
        Fluxes_err_unscaled.append(fluxerr_coadd[0])

        #Fluxes.append((flux[0] + flux_coadd[0]*scale_coadd) * f_scaling) 
        #Fluxes_err.append((fluxerr[0]+ fluxerr_coadd[0]*scale_coadd) * f_scaling)

        #Fluxes_scaled.append(flux[0]*f_scaling)
        #Fluxes_err_scaled.append(fluxerr[0]*f_scaling)

        Fluxes_cal.append(flux_jsky_cal)
        Fluxeserr_cal.append(fluxerr_jsky_cal)

        #Fluxes_annuli.append((flux_an[0]+flux_coadd_an[0])*f_scaling)
        #Fluxeserr_annuli.append((fluxerr_an[0]+fluxerr_coadd_an[0])*f_scaling)

        Fluxes_njsky.append(flux_jsky)
        Fluxeserr_njsky.append(fluxerr_jsky)

        Fluxes_njsky_coadd.append(flux_jsky_coadd)
        Fluxeserr_njsky_coadd.append(fluxerr_jsky_coadd)

        #Fluxes_njsky_coadd.append(flux_co[0])
        #Fluxeserr_njsky_coadd.append(fluxerr_co[0])


        Mag.append(mag_diff_ab)
        Magerr.append(mag_diff_ab_err)

        Mag_coadd.append(mag_ab_coadd)
        Magerr_coadd.append(mag_ab_coadd_err)
        # Adding photometry calculation step: 

        
        #m_instrumental = -2.5*np.log10((flux[0]+flux_coadd[0])/float(calexp.getInfo().getVisitInfo().exposureTime))
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        #Z_value = np.unique(np.array(phot.Z_value))
        #k_value = np.unique(np.array(phot.k_value))
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

    plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, my_calib, '*', color='black', label='My calib')
    plt.errorbar(dates_aux, calib_lsst, yerr=calib_lsst_err, fmt='o', color='blue', label='lsst cal')
    
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('Calibration mean', fontsize=17)
    #plt.title('Calibration scaling comparison', fontsize=17)
    #plt.legend()
    plt.show()

    # plorrint calib intercept

    #plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, my_calib_inter, '*', color='black', label='My calib')
    #plt.plot(dates_aux, calib_lsst, 'o', color='blue', label='lsst cal')
    
    #plt.xlabel('MJD', fontsize=17)
    #plt.ylabel('Calibration intercept', fontsize=17)
    #plt.title('Calibration scaling intercept', fontsize=17)
    #plt.legend()
    #plt.show()
    #if not os.path.isfile(calib_path):
    #    np.savez(calib_path, x = calib_relative, y = calib_relative_intercept)
        #calib_relative.savez('calibration/calibration_scaling_{}_{}'.format(field, ccd_num))
        #calib_relative_intercept.savez('calibration/calibration_scaling_inter_{}_{}'.format(field, ccd_num))
    # calib relative 

    plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, calib_relative, '*', color='black', label='My calibration', linestyle='--')
    plt.errorbar(dates_aux, calib_lsst, yerr=calib_lsst_err, fmt='o', color='blue', label='LSST pipeline', linestyle='--')
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('Calibration mean', fontsize=17)
    plt.title('LSST calibration mean', fontsize=17)
    plt.show()

    # plorrint calib intercept

    #plt.figure(figsize=(10,6))
    #plt.plot(dates_aux, calib_relative_intercept, '*', color='black', label='My calib', linestyle = '--')
    #plt.plot(dates_aux, calib_lsst, 'o', color='blue', label='lsst cal')
    
    #plt.xlabel('MJD', fontsize=17)
    #plt.ylabel('Calibration intercept', fontsize=17)
    #plt.title('Calibration scaling intercept relative to first visit', fontsize=17)
    #plt.legend()
    #plt.show()

    # Airmass plot
    plt.figure(figsize=(10,6))
    plt.plot(dates_aux, Airmass, 'o', color='magenta', linestyle='--')
    plt.title('Airmass', fontsize=17)
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('Airmass', fontsize=17)
    plt.show()

    # Seeing plot
    plt.figure(figsize=(10,6))
    plt.plot(dates_aux, Seeing, 'o', color='magenta', linestyle='--')
    plt.title('FWHM observation', fontsize=17)
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('FWHM', fontsize=17)
    plt.show()

    if do_lc_stars == True:
        py = 2048 - 200
        px = 4096 - 200
        width = px*pixel_to_arcsec
        height = py*pixel_to_arcsec

        #print('width: {} , height : {}'.format(width, height))

        #stars_table = Find_stars(ra_center, dec_center, width, height, nstars, seed=[True, 200])
        stars_table = Inter_Join_Tables_from_LSST(repo, visits_aux, ccd_num, collection_diff, tp=tp)

        if tp=='before_ID':
            random.seed(15030)
            random_indexes = random.sample(range(len(stars_table)), int(len(stars_table)*0.18))
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
        columns_stars = np.ndarray.flatten(np.array([['star_{}_f'.format(i+1), 'star_{}_ferr'.format(i+1), 'star_{}_ft'.format(i+1), 'star_{}_fterr'.format(i+1), 'star_{}_fs'.format(i+1), 'star_{}_fserr'.format(i+1), 'star_{}_mag'.format(i+1), 'star_{}_magErr'.format(i+1), 'star_{}_magt'.format(i+1), 'star_{}_magtErr'.format(i+1)] for i in range(nstars)]))
        stars = pd.DataFrame(columns=columns_stars)
        #stars_table = stars_table.sample(n=nstars)
        
        print('number of stars we will revise: ', len(stars_table))

        ra_s = np.array(stars_table['coord_ra_ddegrees_{}'.format(visits_aux[0])])
        dec_s = np.array(stars_table['coord_dec_ddegrees_{}'.format(visits_aux[0])])

        ps1_mags = pc.get_mags_from_catalog_ps1(ra_s, dec_s)        
        ps1_info = pc.get_from_catalog_ps1(ra_s, dec_s)
        ps1_info = ps1_info.sort_values('gmag')
        
        for j in range(len(visits_aux)):
            RA = np.array(stars_table['coord_ra_ddegrees_{}'.format(visits_aux[j])], dtype=float)
            DEC = np.array(stars_table['coord_dec_ddegrees_{}'.format(visits_aux[j])], dtype=float)
            
            diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits_aux[j], detector=ccd_num , collections=collection_diff, instrument='DECam')
            coadd = butler.get('goodSeeingDiff_matchedExp',visit=visits_aux[j], detector=ccd_num , collections=collection_diff, instrument='DECam')
            calexp = butler.get('calexp',visit=visits_aux[j], detector=ccd_num , collections=collection_diff, instrument='DECam')
            
            photocalib = diffexp.getPhotoCalib()
            photocalib_coadd = coadd.getPhotoCalib()
            photocalib_calexp = calexp.getPhotoCalib()

            wcs = diffexp.getWcs()
            data = np.asarray(diffexp.image.array, dtype='float')            
            data_coadd = np.asarray(coadd.image.array, dtype='float')
            data_calexp = np.asarray(calexp.image.array, dtype='float')
            #np.asarray(calexp.image.array, dtype='float')           
            psf = diffexp.getPsf()
            seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm * pixel_to_arcsec 

            flux_stars_and_errors = []
            #factor_star = 2 #2.5
            star_aperture = seeing * factor_star #2 # arcsec 
            star_aperture/=pixel_to_arcsec # transform it to pixel values 

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
                if verbose:
                    print('x_pix : {}  y_pix : {}'.format(x_star, y_star))
                
                if show_star_stamps:
                    Calib_Diff_and_Coadd_plot_cropped(repo, collection_diff, ra_star, dec_star, [visits_aux[j]], ccd_num, s=star_aperture, cutout=2*star_aperture)
                    values_across_source(calexp, ra_star, dec_star , x_length = star_aperture, y_length=1.5, stat='median', title_plot='Calibrated exposure of star {}'.format(i+1), save_plot = True, field=field, name='median_slit_hist_star{}_mjd_{}'.format(i+1, Truncate(d, 4)))
                    #values_across_source(diffexp, ra_star, dec_star , x_length = star_aperture, y_length=1.5, stat='median', title_plot = 'Difference exposure of star {}'.format(i+1))
            
                    
                
                f, f_err, fg = sep.sum_circle(data, [x_star], [y_star], star_aperture, var = np.asarray(diffexp.variance.array, dtype='float'))
                ft, ft_err, ftg = sep.sum_circle(data_coadd, [x_star], [y_star], star_aperture, var = np.asarray(coadd.variance.array, dtype='float'))
                fs, fs_err, fsg = sep.sum_circle(data_calexp, [x_star], [y_star], star_aperture, var = np.asarray(calexp.variance.array, dtype='float'))
                
                if (np.fabs(f[0])>2000  or np.fabs(f[0]/ft[0])>0.9):
                    saturated_stars.append(i+1)
                
                # Using LSST photocalibration
                f_star_physical = photocalib.instFluxToNanojansky(f[0], f_err[0], obj_pos_2d_star)
                ft_star_physical = photocalib_coadd.instFluxToNanojansky(ft[0], ft_err[0], obj_pos_2d_star)
                fs_star_physical = photocalib_calexp.instFluxToNanojansky(fs[0], fs_err[0], obj_pos_2d_star)


                flux_stars_and_errors.append(f_star_physical.value)
                flux_stars_and_errors.append(f_star_physical.error)
                flux_stars_and_errors.append(ft_star_physical.value)
                flux_stars_and_errors.append(ft_star_physical.error)
                flux_stars_and_errors.append(fs_star_physical.value)
                flux_stars_and_errors.append(fs_star_physical.error)


                Fstar = np.array((f_star_physical.value + ft_star_physical.value)*1e-9)
                Fstar_err = np.sqrt((ft_star_physical.error*1e-9)**2 + (f_star_physical.error*1e-9)**2)

                Magstars = pc.FluxJyToABMag(Fstar, Fstar_err)
                Mag_star = Magstars[0]
                Mag_star_err = Magstars[1]

                Magstars_coadd = pc.FluxJyToABMag(ft_star_physical.value*1e-9, ft_star_physical.error*1e-9)
                Mag_star_coadd = Magstars_coadd[0]
                Mag_star_coadd_err = Magstars_coadd[1]

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
                flux_stars_and_errors.append(Mag_star_coadd)
                flux_stars_and_errors.append(Mag_star_coadd_err)
                #print('len flux_stars_and_errors: ', len(flux_stars_and_errors))

                ###########
                if verbose:
                    print('Flux star: {} Error flux: {}'.format(f[0] + ft[0], f_err[0]))
                    print('Flux star in coadd: {} Error flux in coadd: {}'.format(ft[0], ft_err[0]))
                    
                    print('꒰✩ ’ω`ૢ✩꒱ -------------------- ⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂')
            
            stars.loc[len(stars.index)] = flux_stars_and_errors
        #field = collection_diff[13:24]
        #here we plot the stars vs panstarss magnitude:
        norm = matplotlib.colors.Normalize(vmin=0,vmax=32)
        c_m = matplotlib.cm.plasma

        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(0,27,len(visits_aux))
        fluxt_stars = [np.median(np.array(stars['star_{}_ft'.format(i+1)])) for i in range(nstars)]
        #fluxt_stars_norm_factor = np.linalg.norm(fluxt_stars)
        #fluxt_stars_norm = fluxt_stars/fluxt_stars_norm_factor

        plt.show()
        plt.figure(figsize=(10,10))
        stars_table = stars_table.sort_values('base_PsfFlux_mag_{}'.format(visits_aux[0]))
        #kk = 0
        for i in range(len(visits_aux)):
            plt.errorbar(np.array(ps1_info.gmag) , np.array(stars_table['base_PsfFlux_mag_{}'.format(visits_aux[i])]) - np.array((ps1_info.gmag)), yerr= np.array(stars_table['base_PsfFlux_magErr_{}'.format(visits_aux[i])]), fmt='*', label=' visit {}'.format(visits_aux[i]), color=s_m.to_rgba(T[i]), markersize=10, alpha=0.5, linestyle='--')
            #k+=0
            #plt.plot(np.array(ps1_mags.ps1_mag), np.array(ps1_mags.ps1_mag) - np.array(ps1_mags.ps1_mag), label='PS1')
        plt.xlabel('PS1 magnitudes', fontsize=17)
        plt.plot(np.sort(np.array(ps1_info.gmag)), np.array(ps1_info.gmag) - np.array(ps1_info.gmag),'*', markersize=10, label='PS1', color='black', linestyle='--')
        color_term = 20e-3
        color_term_rms = 6e-3
        #plt.plot(ps1_info.gmag, ps1_info.g_r*color_term , '*', color='green', markersize=10, alpha=0.5, linestyle='--', label='g-r * color term')
        plt.errorbar(ps1_info.gmag,  ps1_info.g_i*color_term , yerr=ps1_info.g_i*color_term_rms , fmt='*', color='black', capsize=4, markersize=10, alpha=0.5, linestyle='--', label='g-i * color term')        
        plt.ylabel('Magnitude - PS1 magnitude', fontsize=17)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Difference between magnitudes', fontsize=17)
        if save_lc_stars:
            plt.savefig('light_curves/{}/{}_{}_magnitude_and_colors.jpeg'.format(field, field, ccd_num), bbox_inches='tight')
        plt.show()


        color_correc = pd.Series(ps1_info.g_i*color_term ,index=ps1_info.index).to_dict()
        print(color_correc)
        plt.figure(figsize=(10,10))
        abs_zero = 0#.04
        for i in range(len(visits_aux)):
            mag_unc_err = np.sqrt(np.array(stars_table['base_PsfFlux_magErr_{}'.format(visits_aux[i])])**2 + (ps1_info.g_i*color_term_rms)**2 )
            plt.errorbar(np.array(ps1_info.gmag) , np.array(stars_table['base_PsfFlux_mag_{}'.format(visits_aux[i])]) - np.array(ps1_info.gmag) - ps1_info.g_i*color_term - abs_zero ,yerr=mag_unc_err, fmt='*', capsize=4, label=' visit {}'.format(visits_aux[i]), color=s_m.to_rgba(T[i]), markersize=10, alpha=0.5, linestyle='--')
            #plt.plot(np.array(ps1_mags.ps1_mag), np.array(ps1_mags.ps1_mag) - np.array(ps1_mags.ps1_mag), label='PS1')
        #stars_table['color_color_term'] = ps1_info.g_i*color_term 
        plt.xlabel('PS1 magnitudes', fontsize=17)
        plt.plot(np.sort(np.array(ps1_info.gmag)), np.array(ps1_info.gmag) - np.array(ps1_info.gmag),'*', markersize=10, label='PS1', color='black', linestyle='--')
        
        #color_term = 20e-3
        #plt.plot(ps1_info.gmag, ps1_info.g_r*color_term , '*', color='green', markersize=10, alpha=0.5, linestyle='--', label='g-r * color term')
        #plt.plot(ps1_info.gmag, ps1_info.g_i*color_term , '*', color='black', markersize=10, alpha=0.5, linestyle='--', label='g-i * color term')
        
        plt.ylabel('Magnitude - PS1 magnitude', fontsize=17)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Difference between magnitudes', fontsize=17)
        #plt.savefig('light_curves/{}/{}_{}_magnitude_and_colors.png'.format(field, field, ccd_num), bbox_inches='tight')
        plt.show()
        norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
        c_m = matplotlib.cm.plasma

        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

        columns_mag = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        columns_magErr = ['base_PsfFlux_magErr_{}'.format(v) for v in visits_aux]
        fig, axs = plt.subplots(int(np.sqrt(nstars))+1,int(np.sqrt(nstars))+1 , figsize=(10, 6), constrained_layout=True)
        #fig.set_title('Flux of stars measured by LSST')
        for ax, markevery in zip(axs.flat, range(nstars)):           
            ax.set_title(f'star number {markevery+1}')
            magss = np.array(stars_table.loc[markevery][columns_mag])
            magsserr = np.array(stars_table.loc[markevery][columns_magErr])
            flux = [pc.ABMagToFlux(m)*1e9 for m in magss]
            #fluxErr  = [pc.ABMagToFlux()]
            ax.errorbar(dates_aux, flux, fmt = 'o', ls='-', color=s_m.to_rgba(fluxt_stars[markevery]))
        #here we plot the stars vs panstarss magnitude:
        
        norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
        c_m = matplotlib.cm.plasma

        # create a ScalarMappable and initialize a data structure
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

        plt.figure(figsize=(10,10))
        ii = 0

        for j in range(len(stars_table)):
            #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
            mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
            magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
            fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
            flux_of_star_j = fluxes_of_star_j[0]*1e9
            fluxErr_of_star_j = fluxes_of_star_j[1]*1e9
            #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
            plt.errorbar(dates_aux, flux_of_star_j - np.median(flux_of_star_j), fluxErr_of_star_j, fmt='*', color=s_m.to_rgba(fluxt_stars[j]), ls='--', label='star {}'.format(j+1))
            ii+=1

        plt.xlabel('MJD', fontsize=17)
        plt.ylabel('Flux [nJy] of LSST - median', fontsize=17)
        plt.title('Lightcurves of stars, measured by LSST', fontsize=17)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        plt.show()


        plt.figure(figsize=(10,10))
        ii = 0

        #deviation_from_median = []
        for j in range(len(stars_table)):
            #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
            mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
            magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
            fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
            flux_of_star_j = fluxes_of_star_j[0]*1e9
            fluxErr_of_star_j = fluxes_of_star_j[1]*1e9
            #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
            plt.hist(flux_of_star_j - np.median(flux_of_star_j), alpha=0.5, label='star {}'.format(j+1), color=s_m.to_rgba(fluxt_stars[j]))
            ii+=1

        #plt.ylabel('MJD', fontsize=17)
        plt.xlabel('Flux [nJy] of LSST - median', fontsize=17)
        plt.title('Lightcurves of stars, measured by LSST', fontsize=17)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        plt.show()


        plt.figure(figsize=(10,10))
        ii = 0
        
        excess_var = []
        for j in range(len(stars_table)):
            #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
            mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
            magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
            #fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
            #flux_of_star_j = fluxes_of_star_j[0]*1e9
            #fluxErr_of_star_j = fluxes_of_star_j[1]*1e9
            #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
            #plt.hist(Excess_variance(mag_of_star_j, magErr_of_star_j), alpha=0.5, label='star {}'.format(j+1), histtype='step', color=s_m.to_rgba(fluxt_stars[j]))
            excess_var.append(Excess_variance(mag_of_star_j, magErr_of_star_j))
            #print(Excess_variance(mag_of_star_j, magErr_of_star_j))
            ii+=1

        plt.hist(excess_var, label='star {}'.format(j+1), color='black')
        print(excess_var)
        #plt.ylabel('MJD', fontsize=17)
        plt.xlabel('Excess Variance of stars', fontsize=17)
        #plt.title('Lightcurves of stars, measured by LSST', fontsize=17)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        plt.show()


        plt.figure(figsize=(10,10))
        ii = 0
        
        excess_var = []
        for j in range(len(stars_table)):
            #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
            mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
            magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
            fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
            flux_of_star_j = fluxes_of_star_j[0]*1e9
            fluxErr_of_star_j = fluxes_of_star_j[1]*1e9
            #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
            plt.hist((flux_of_star_j - np.median(flux_of_star_j)) / np.median(flux_of_star_j) * 100, alpha=0.5, label='star {}'.format(j+1), histtype='step', color=s_m.to_rgba(fluxt_stars[j]))
            #excess_var.append(Excess_variance(mag_of_star_j, magErr_of_star_j))
            #print(Excess_variance(mag_of_star_j, magErr_of_star_j))
            ii+=1

        plt.hist(excess_var, alpha=0.5, label='star {}'.format(j+1), histtype='step', color=s_m.to_rgba(fluxt_stars[j]))
        print(excess_var)
        #plt.ylabel('MJD', fontsize=17)
        plt.xlabel('(flux_j - flux_median) / flux_median', fontsize=17)
        plt.title('Percentage of deviation from median, in flux', fontsize=17)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10,10))
        ii = 0

        #deviation_from_median = []
        for j in range(len(stars_table)):
            #columns = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
            mag_of_star_j = np.array(stars_table.loc[j][columns_mag])
            #magErr_of_star_j = np.array(stars_table.loc[j][columns_magErr])
            #fluxes_of_star_j = pc.ABMagToFlux(mag_of_star_j, magErr_of_star_j)
            #flux_of_star_j = fluxes_of_star_j[0]*1e9
            #fluxErr_of_star_j = fluxes_of_star_j[1]*1e9
            #plt.plot(dates_aux, mag_of_star_j - np.median(mag_of_star_j), '*', color=s_m.to_rgba(fluxt_stars[j]), linestyle='--', label='star {}'.format(j+1))
            plt.hist(mag_of_star_j - np.median(mag_of_star_j), alpha=0.5, label='star {}'.format(j+1), color=s_m.to_rgba(fluxt_stars[j]))
            ii+=1

        #plt.ylabel('MJD', fontsize=17)
        plt.xlabel('AB mag of LSST - median', fontsize=17)
        plt.title('Lightcurves of stars, measured by LSST', fontsize=17)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/lightcurves_LSST_stars.jpeg'.format(field), bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10,6))
        mags_visits_list = []
        mags_visits_p16 = []
        mags_visits_p84 = []
        stars['dates'] = dates_aux #dates - min(dates)

        for i in range(len(visits_aux)):
            mag_stars = np.array(stars_table['base_PsfFlux_mag_{}'.format(visits_aux[i])])
            mags_visits = np.mean(mag_stars)
            #mags_visits_p16.append(np.percentile(mag_stars,))
            mags_visits_list.append(mags_visits)
            mags_ps1_mean = np.mean(ps1_mags.ps1_mag)
        plt.plot(dates_aux, np.array(mags_visits_list) - mags_ps1_mean, '*', color='magenta', markersize=10, alpha=0.5, linestyle='--')
        plt.xlabel('MJD', fontsize=17)
        plt.ylabel('mean mag LSST - mean mag PS1', fontsize=17)
        #plt.legend()
        plt.title('Difference between means of magnitudes', fontsize=17)
        plt.show()

        saturated_stars = np.unique(np.array(saturated_stars))
        plt.figure(figsize=(10,6))       
        stars = stars.sort_values(by='dates')
        #print('Here is the dataset of the stars calculated: ')
        #print(stars)
        column_stars_diff_flux = ['star_{}_f'.format(i+1) for i in range(nstars)]
        diff_flux_stars = stars[column_stars_diff_flux]
        max_value = max(list(stars.index))

        # plotting flxs of stars 
        for i in range(nstars):
            ft_star = (np.array(stars['star_{}_ft'.format(i+1)])).flatten() #* scaling
            ft_star_err = np.ndarray.flatten(np.array(stars['star_{}_fterr'.format(i+1)])) #* scaling
            
            #Dates = np.array(stars['dates'])
            new_dates = dates_aux
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
            c_m = matplotlib.cm.plasma

            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

            if len(j)==0:
                plt.errorbar(dates_aux, ft_star - np.median(ft_star), yerr= ft_star_err, capsize=4, fmt='s', ls='solid', label = 'star {} coadd'.format(i+1), color = s_m.to_rgba(fluxt_stars[i]))
        if well_subtracted:
            plt.title('stars LCs in template/coadd image from {} and {} with Aperture of {}*FWHM", well subtracted'.format(field, ccd_name[ccd_num], factor_star)) 
        if not well_subtracted:
            plt.title('stars LCs in template/coadd image from {} and {} with Aperture {}*FWHM'.format(field, ccd_name[ccd_num], factor_star)) 

        plt.xlabel('MJD', fontsize=15)
        plt.ylabel('offset Flux [nJy] from median', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

        norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
        c_m = matplotlib.cm.plasma

        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

        columns_mag = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        columns_magErr = ['base_PsfFlux_magErr_{}'.format(v) for v in visits_aux]
        fig, axs = plt.subplots(int(np.sqrt(nstars))+1,int(np.sqrt(nstars))+1 , figsize=(10, 6), constrained_layout=True)
        #fig.set_title('Flux measured on coadd template', fontsize=17)
        for ax, markevery in zip(axs.flat, range(nstars)):
            ft_star = (np.array(stars['star_{}_ft'.format(markevery+1)])).flatten() #* scaling
            ft_star_err = np.ndarray.flatten(np.array(stars['star_{}_fterr'.format(markevery+1)])) #* scaling

            ax.set_title(f'star number {markevery+1}')
            ax.errorbar(dates_aux, ft_star, yerr = ft_star_err, fmt = 'o', ls='-', color=s_m.to_rgba(fluxt_stars[markevery]))
        # plotting flxs of stars 
        #plt.title('Flux measured on coadd template', fontsize=17)
        plt.show()

        plt.figure(figsize=(10,6))
        for i in range(nstars):
            fs_star = (np.array(stars['star_{}_fs'.format(i+1)])).flatten() #* scaling
            fs_star_err = np.ndarray.flatten(np.array(stars['star_{}_fserr'.format(i+1)])) #* scaling
            
            #Dates = np.array(stars['dates'])
            new_dates = dates_aux
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
            c_m = matplotlib.cm.plasma

            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

            if len(j)==0:
                plt.errorbar(dates_aux, fs_star - np.median(fs_star), yerr= fs_star_err, capsize=4, fmt='s', ls='solid', label = 'star {} science'.format(i+1), color = s_m.to_rgba(fluxt_stars[i]))
        if well_subtracted:
            plt.title('stars LCs in science image from {} and {} with Aperture of {}*FWHM", well subtracted'.format(field, ccd_name[ccd_num], factor_star)) 
        if not well_subtracted:
            plt.title('stars LCs in science image from {} and {} with Aperture {}*FWHM'.format(field, ccd_name[ccd_num], factor_star)) 

        plt.xlabel('MJD', fontsize=15)
        plt.ylabel('offset Flux [nJy] from median', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

        plt.figure(figsize=(10,6))
        for i in range(nstars):
            fs_star = (np.array(stars['star_{}_fs'.format(i+1)])).flatten() #* scaling
            fs_star_err = np.ndarray.flatten(np.array(stars['star_{}_fserr'.format(i+1)])) #* scaling
            
            #Dates = np.array(stars['dates'])
            new_dates = dates_aux
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
            c_m = matplotlib.cm.plasma

            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

            if len(j)==0:
                plt.hist(fs_star - np.median(fs_star), alpha=0.5, label = 'star {} science'.format(i+1), color = s_m.to_rgba(fluxt_stars[i]))
  
        plt.xlabel('MJD', fontsize=15)
        plt.ylabel('offset Flux [nJy] from median', fontsize=15)
        plt.legend(loc=9, ncol=5)

        plt.show()


        norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
        c_m = matplotlib.cm.plasma

        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

        columns_mag = ['base_PsfFlux_mag_{}'.format(v) for v in visits_aux]
        columns_magErr = ['base_PsfFlux_magErr_{}'.format(v) for v in visits_aux]
        fig, axs = plt.subplots(int(np.sqrt(nstars))+1,int(np.sqrt(nstars))+1 , figsize=(10, 6), constrained_layout=True)
        #fig.set_title('Flux measured on coadd template', fontsize=17)
        for ax, markevery in zip(axs.flat, range(nstars)):
            ft_star = (np.array(stars['star_{}_fs'.format(markevery+1)])).flatten() #* scaling
            ft_star_err = np.ndarray.flatten(np.array(stars['star_{}_fserr'.format(markevery+1)])) #* scaling

            ax.set_title(f'star number {markevery+1}')
            ax.errorbar(dates_aux, ft_star, yerr = ft_star_err, fmt = 'o', ls='-', color=s_m.to_rgba(fluxt_stars[markevery]))
        # plotting flxs of stars 
        #plt.title('Flux measured on coadd template', fontsize=17)
        plt.show()
        plt.figure(figsize=(10,10))
        for i in range(nstars):
            f_star = np.array(stars['star_{}_f'.format(i+1)]) #* scaling
            f_star_err = np.array(stars['star_{}_ferr'.format(i+1)]) #* scaling
                  
            Dates = np.array(stars['dates'])
            new_dates = dates_aux
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
            c_m = matplotlib.cm.plasma

            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)
            if len(j)==0:
                plt.errorbar(Dates, f_star, yerr= f_star_err, capsize=4, fmt='s', ls='solid', label = 'star {} diff'.format(i+1), color = s_m.to_rgba(fluxt_stars[i]))
        
        plt.ylabel('Difference Flux [nJy]', fontsize=15)    
        plt.xlabel('MJD', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.show()

        plt.figure(figsize=(10,6))

        for i in range(nstars):
            f_star = np.array(stars['star_{}_f'.format(i+1)]) #* scaling
            f_star_err = np.array(stars['star_{}_ferr'.format(i+1)]) #* scaling
            
            #Dates = np.array(stars['dates'])
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=min(fluxt_stars),vmax=max(fluxt_stars))
            c_m = matplotlib.cm.plasma

            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            T = np.linspace(min(fluxt_stars),max(fluxt_stars),nstars)

            if len(j)==0:
                plt.hist(f_star - np.median(f_star), alpha=0.5, label = 'star {} science'.format(i+1), color = s_m.to_rgba(fluxt_stars[i]))
  
        #plt.xlabel('MJD', fontsize=15)
        plt.xlabel('offset Flux [nJy] from median', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()


        

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

    #f, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 1, 1]},sharex=True, figsize=(10,6))
    plt.figure(figsize=(10,6))


    if SIBLING!=None and sfx == 'flx':
        x, y, yerr = compare_to(SIBLING, sfx='mag', factor=0.75)
        f, ferr = pc.ABMagToFlux(y, yerr)
        plt.errorbar(x-min(x),f*10 - np.median(f*10), yerr=ferr*10,  capsize=4, fmt='^', ecolor='black', color='black', label='Martinez-Palomera et al. 2020', ls ='dotted')
    
    #plt.show()

    #plt.figure(figsize=(10,6))

    if do_zogy:
        zogy = zogy_lc(repo, collection_calexp, collection_coadd, ra, dec, ccd_num, visits, r_aux, instrument = 'DECam', plot_diffexp=plot_zogy_stamps, plot_coadd = plot_coadd, cutout=cutout)
        print(zogy)
        z_flux = zogy.flux
        z_ferr = zogy.flux_err
        plt.errorbar(zogy.dates, z_flux, yerr=z_ferr, capsize=4, fmt='s', label ='ZOGY Cáceres-Burgos', color='orange', ls ='dotted')

    area_source  = np.pi * r_aux**2
    area_annuli = np.pi * (5*r_aux)**2 - np.pi * (2*r_aux)**2 
    source_of_interest = pd.DataFrame()
    source_of_interest['dates'] = dates_aux # dates 
    #source_of_interest['flux'] = np.array(Fluxes) # flux template + difference 
    #source_of_interest['flux_err'] = np.array(Fluxes_err) # error flux template + difference 
    #source_of_interest['flux_scaled'] = np.array(Fluxes_scaled) # difference flux 
    #source_of_interest['flux_err_scaled'] = np.array(Fluxes_err_scaled) #difference flux error 
    source_of_interest['flux_unscaled'] = np.array(Fluxes_unscaled) # unscaled flux template + differenc
    source_of_interest['flux_err_unscaled'] = np.array(Fluxes_err_unscaled) # unscaled error flux template + difference 
    #source_of_interest['flux_annuli'] = np.array(Fluxes_annuli) #* area_source/area_annuli # flux of annuli template + difference 
    #source_of_interest['flux_err_annuli'] = np.array(Fluxeserr_annuli) # error flux of annuli template + difference 
    source_of_interest['flux_nJy'] = np.array(Fluxes_njsky)
    source_of_interest['fluxerr_nJy'] = np.array(Fluxeserr_njsky) 
    source_of_interest['flux_nJy_coadd'] = np.array(Fluxes_njsky_coadd)
    source_of_interest['fluxerr_nJy_coadd'] = np.array(Fluxeserr_njsky_coadd)
    source_of_interest['flux_nJy_cal'] = np.array(Fluxes_cal)
    source_of_interest['fluxerr_nJy_cal'] = np.array(Fluxeserr_cal)

    
    source_of_interest['Mg_coadd'] = Mag_coadd
    source_of_interest['Mg_err_coadd'] = Magerr_coadd
    source_of_interest['Exptimes'] = ExpTimes
    #source_of_interest['flux_annuli_subtracted_to_median'] =  source_of_interest['flux_annuli'] - np.median(source_of_interest['flux_annuli']) #/source_of_interest['flux_annuli_norm']
    #source_of_interest['flux_annuli_norm'] = np.array(Fluxes_annuli)/np.array(Fluxes_annuli).sum()
    #source_of_interest['flux_corrected'] = source_of_interest['flux'] - source_of_interest['flux_annuli_subtracted_to_median'] #/source_of_interest['flux_annuli_norm'] 
    
    #source_of_interest['flux_cal'] = np.array(Fluxes_cal) # scaled flux of science 
    #source_of_interest['flux_err_cal'] = np.array(Fluxeserr_cal) # scaled error flux of science 
    
    #print(source_of_interest['flux_annuli_subtracted_to_median'])
    
    source_of_interest['Mg'] = Mag #photocalib.instFluxToMagnitude(flux[0], fluxerr[0], obj_pos_2d).value #-2.5*np.log10(source_of_interest.flux + flux_coadd) + magzero_image
    source_of_interest['Mg_err'] = Magerr#photocalib.instFluxToMagnitude(flux[0], fluxerr[0], obj_pos_2d).error #np.sqrt(2.5*source_of_interest.flux_err/(source_of_interest.flux * np.log(10)))

    source_of_interest['visit'] = visits_aux
    source_of_interest = source_of_interest.sort_values(by='dates')

    if sfx=='mag':
        plt.errorbar(source_of_interest.dates, source_of_interest.Mg - np.median(source_of_interest.Mg), yerr = source_of_interest.Mg_err, capsize=4, fmt='s', label ='AL Cáceres-Burgos coadd + diff', color='#0827F5', ls ='dotted')
        plt.errorbar(source_of_interest.dates, source_of_interest.Mg_coadd - np.median(source_of_interest.Mg_coadd), yerr = source_of_interest.Mg_err_coadd, capsize=4, fmt='s', label ='AL Cáceres-Burgos coadd', color='#082785', ls ='dotted')

        plt.ylabel('Excess AB magnitude', fontsize=15)
    
    else: 
        #plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy + source_of_interest.flux_nJy_coadd - np.median(source_of_interest.flux_nJy + source_of_interest.flux_nJy_coadd), yerr = np.sqrt(source_of_interest.fluxerr_nJy**2 + source_of_interest.fluxerr_nJy_coadd**2) , capsize=4, fmt='s', label ='AL Cáceres-Burgos [diff + template]', color='#0827F5', ls ='dotted')
        plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy_coadd - np.median(source_of_interest.flux_nJy_coadd), yerr = source_of_interest.fluxerr_nJy_coadd, capsize=4, fmt='s', label ='AL Cáceres-Burgos coadd', color='red', ls ='dotted')
        plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy, yerr = source_of_interest.fluxerr_nJy, capsize=4, fmt='s', label ='AL Cáceres-Burgos diff', color='magenta', ls ='dotted')
        plt.errorbar(source_of_interest.dates, source_of_interest.flux_nJy_cal - np.median(source_of_interest.flux_nJy_cal), yerr = source_of_interest.fluxerr_nJy_cal, capsize=4, fmt='s', label ='AL Cáceres-Burgos cal', color='blue', ls ='dotted')
        
        #for i in range(len(source_of_interest)):
        #    plt.text(np.array(source_of_interest.dates)[i], 0, '{0:.8g}'.format(magzero[i]), rotation=45)
        
        plt.ylabel('Excess flux nJy', fontsize=15 )
        #plt.axhline(0, color='grey', linestyle='--')
    
    
    #plt.ylabel('Excess Flux in arbitrary units', fontsize=15 )

    #for i in range(len(source_of_interest.dates)):
    #    plt.text(np.array(source_of_interest.dates)[i], 0, '{0:.3g}"'.format(Airmass[i]), rotation=45)
    #plt.text()
    plt.legend(ncol=2)
    #ax2 = plt.subplot(212)
    
    #if SIBLING!=None:
        #ax2.plot(source_of_interest.dates, source_of_interest.flux - np.median(source_of_interest.flux) - y, '*' , color=dark_purple , label='residuals', linestyle ='--')
        #ax2.legend()
        #ax2.axhline(0, color='grey', linestyle='--')

        #for i in range(len(source_of_interest.dates)):
        #    ax2.text(np.array(source_of_interest.dates)[i], 0, '{0:.3g}"'.format(Seeing[i]), rotation=45)


    #ax3.set_xlabel('MJD', fontsize=15)
    
    
    plt.title('Aperture radii: {}", source {}'.format(r_in_arcsec, title), fontsize=15)
    #ax3.errorbar(source_of_interest.dates, source_of_interest.flux_annuli, yerr=source_of_interest.flux_err_annuli, capsize=4, fmt='s', label ='AL annuli Cáceres-Burgos [+template flux scaled]', color='#6600CC', ls ='dotted')
    #plt.ylabel('ADU', fontsize=12)
    #ax3.axhline(0, color='grey', linestyle='--')
    #f.subplots_adjust(hspace=0)
    #plt.legend(ncol=5)
    
    if sparse_obs:
        plt.xscale('log')

    if save and save_as=='':
        plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/ra_{}_dec_{}_{}.jpeg'.format(ra,dec,sfx), bbox_inches='tight')
    
    if save and save_as!='':
        plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}_{}.jpeg'.format(save_as, sfx), bbox_inches='tight')
    
    plt.show()
    

    # This line below plots the stamps of the source as a single figure for all epochs available!
    if save_stamps:
        Calib_Diff_and_Coadd_one_plot_cropped(repo, collection_diff, ra, dec, list(source_of_interest.visit), ccd_num, cutout=cutout, s=r_aux, save_stamps=save_stamps, save_as=save_as+ '_stamps')
    
    plt.show()
    
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
    return source_of_interest

def all_ccds(repo, field, collection_calexp, collection_diff, collection_coadd, sfx='flx', show_star_stamps = False, show_stamps=False, factor=1, well_subtracted=False, factor_star=2):
    """
    plots all sources located in a field

    """
    Dict = {}
    #repo = "/home/jahumada/data_hits"
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
            df = get_light_curve(repo, visits, collection_diff, collection_calexp, ccdnum, ra, dec, r=10, factor=factor, save=True, save_as = folder + name_file, SIBLING = '/home/jahumada/Jorge_LCs/'+cands.internalID.loc[index[i]] +'_g_psf_ff.csv', title=title, show_stamps=show_stamps, do_zogy=False, collection_coadd=collection_coadd, plot_coadd=False, save_stamps=True, do_lc_stars=True, save_lc_stars=True, show_star_stamps=show_star_stamps, sfx=sfx, nstars=10, well_subtracted=well_subtracted, field=field, factor_star = factor_star)
            
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
    plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/{}_all_ccds_diference.png'.format(field,field))
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
    plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/{}_all_ccds_template.png'.format(field,field))
    plt.show()

    #i=0
    #for key in Dict:
    #    source_of_interest = Dict[key]
    #    if type(source_of_interest) == type(None):
    #        continue
    #    plt.errorbar(source_of_interest.dates, source_of_interest.Mg - source_of_interest.Mg_coadd, yerr = np.sqrt(np.array(source_of_interest.Mg_err)**2 + np.array(source_of_interest.Mg_err_coadd)**2) , capsize=4, fmt='s', label ='AL Cáceres-Burgos in prep {}'.format(key), ls ='dotted',  color = s_m.to_rgba(T[i]))
    #    plt.xlabel('MJD', fontsize=15)
    #    plt.ylabel('offset Mag AB difference', fontsize=15)
    #    plt.title('Magnitude AB - Magnitude AB coadd', fontsize=15)
    #    i+=1
    ##plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/{}_mag_all_ccds_difference.png'.format(field, field))
    #plt.legend()
    ##plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/{}_all_ccds_ab_magnitude.png'.format(field, field))
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
    #plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/{}_mag_all_ccds_difference.png'.format(field, field))
    
    
    return ccds_used


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
        SIBLING = '/home/jahumada/Jorge_LCs/'+cands.internalID.loc[index[i]] +'_g_psf_ff.csv'
        #sfx = 'flx'
        factor = 0.75
        x,y,yerr = compare_to(SIBLING, sfx, factor, beforeDate=57072)
        print(ccds[i])
        plt.errorbar(x-min(x),y, yerr=yerr,  capsize=4, fmt='o', label='Martinez-Palomera et al. 2020 {}'.format(ccds[i]), ls ='dotted', color = s_m.to_rgba(T[i]))
        i+=1
    plt.legend()
    plt.title('Aperture Fotometry {} by Martinez-Palomera for {}'.format(sfx, field))
    plt.savefig('/home/jahumada/testdata_hits/LSST_notebooks/light_curves/{}/{}_all_ccds_{}_Jorge_aperture.png'.format(folder, field, sfx))
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

def Select_table_from_one_calib_exposure(repo, visit, ccdnum, collection_calexp):
    """
    Selects stars from calibrated exposures 

    """
    
    butler = Butler(repo)
    calexp = butler.get('calexp',visit=visit, detector=ccdnum , collections=collection_calexp, instrument='DECam')
    photocalib_calexp = calexp.getPhotoCalib()

    src = butler.get('src',visit=visit, detector=ccdnum , collections=collection_calexp, instrument='DECam')
    src = photocalib_calexp.calibrateCatalog(src)
    src_pandas = src.asAstropy().to_pandas()
    src_pandas['coord_ra_trunc'] = [Truncate(f, 5) for f in np.array(src['coord_ra'])]
    src_pandas['coord_dec_trunc'] = [Truncate(f, 5) for f in np.array(src['coord_dec'])]
    mask = (src_pandas['calib_photometry_used'] == True) & (src_pandas['base_PsfFlux_instFlux']/src_pandas['base_PsfFlux_instFluxErr'] > 50)
    stars_photometry = src_pandas[mask]
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
    coadd = butler.get('goodSeeingDiff_matchedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
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
    mask = (sources['src_calib_photometry_used'] == True) & (sources['base_PsfFlux_instFlux']/sources['base_PsfFlux_instFluxErr'] > 50)
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


def Inter_Join_Tables_from_LSST(repo, visits, ccdnum, collection_diff, well_subtracted =True, tp='after_ID'):
    """
    returns the common stars used for calibration
    """
    big_table = Join_Tables_from_LSST(repo, visits, ccdnum, collection_diff,well_subtracted = well_subtracted, tp=tp)
    phot_table = big_table.dropna()
    phot_table = phot_table.drop_duplicates('coord_ra_trunc')
    phot_table = phot_table.reset_index()
    #phot_table = phot_table.drop('index')
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
    print(flux)
    j, = np.where(flux == max(flux))
    
    return objects[j]    

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

    if SIBLING!=None and SIBLING[0:24]=="/home/jahumada/Jorge_LCs" and type(SIBLING)==str:
        Jorge_LC = pd.read_csv(SIBLING, header=5)
        Jorge_LC = Jorge_LC[Jorge_LC['mjd']<beforeDate] 
        sfx_aux = 'mag'
        if factor==0.5:
            
            param = Jorge_LC['aperture_{}_0'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_0'.format(sfx)]
            median_jorge=np.median(param)
            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]
            #    param = 
            #    median_jorge=0 
            #
            #std = np.norm(Jorge_LC.aperture_flx_0)
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_0 - mean, yerr=Jorge_LC.aperture_flx_err_0,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param - median_jorge
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

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param - median_jorge
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

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param - median_jorge
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

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param - median_jorge
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

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param - median_jorge
            yerr = param_err
            return x, y, yerr
    
    HiTS = directory 

    if HiTS!=None and HiTS[0:24]=="/home/jahumada/HiTS_LCs/" and type(HiTS)==str:
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



