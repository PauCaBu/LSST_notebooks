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

detector_nomenclature= {'S29':1, 'S30':2, 'S31':3, 'S28':7, 'S27':6, 'S26':5, 'S25':4, 'S24':12, 'S23':11, 'S22':10, 'S21':9, 'S20':8, 'S19':18, 'S18':17, 'S17':16, 'S16':15, 'S15':14, 'S14':13, 'S13':24, 'S12':23, 'S11':22, 'S10':21, 'S9':20,'S8':19, 'S7':31, 'S6':30, 'S5':29, 'S4':28, 'S3':27, 'S2':26, 'S1':25, 'N29':60, 'N30':61, 'N31':62, 'N28':59, 'N27':58, 'N26':57, 'N25':56, 'N24':55, 'N23':54, 'N22':53, 'N21':52, 'N20':51, 'N19':50, 'N18':49, 'N17':48, 'N16':47, 'N15':46, 'N14':45, 'N13':44, 'N12':43, 'N11':42, 'N10':41, 'N9':40,'N8':39, 'N7':38, 'N6':37, 'N5':36, 'N4':35, 'N3':34, 'N2':33, 'N1':32 }
ccd_name = dict(zip(detector_nomenclature.values(), detector_nomenclature.keys()))
sibling_allcand = pd.read_csv('../SIBLING_allcand.csv', index_col=0)

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

def Calib_and_Diff_plot(repo, collection_diff, collection_calexp, ra, dec, visits, ccd_num):
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


        fig = plt.figure(figsize=(16, 14))
        imag_display = []

        fig.add_subplot(1,2,1)
        imag_display.append(afwDisplay.Display(frame=fig))
        imag_display[0].scale('linear', 'zscale')
        imag_display[0].mtv(calexp_im)
        plt.title("Calibrated exposure", fontsize=17)

        fig.add_subplot(1,2,2)
        imag_display.append(afwDisplay.Display(frame=fig))
        imag_display[1].scale('linear', 'zscale')
        imag_display[1].mtv(diffexp_im)
        plt.title("Subtracted exposure", fontsize=17)

        x_pix, y_pix = radec_to_pixel(ra, dec, diffexp.getWcs())

        imag_display[1].dot('o', x_pix, y_pix, ctype='magenta', size=20)

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
        calexp = butler.get('calexp', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        calexp_im = calexp.getMaskedImage()
        calexp_cat = butler.get('src', visit= visits[i], detector= ccd_num, instrument='DECam', collections=collection_calexp)
        
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
        #for src in calexp_cat:
        #    stamp_display[0].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
        plt.title('Calexp Image and Source Catalog')

        fig.add_subplot(1,2,2)
        stamp_display.append(afwDisplay.Display(frame=fig))
        stamp_display[1].scale('asinh', 'zscale')
        stamp_display[1].mtv(diffexp_cutout.maskedImage)


        stamp_display[1].dot('o', x_pix, y_pix, ctype='#0827F5', size=s)

        #for src in diffexp_cat:
        #    stamp_display[1].dot('o', src.getX(), src.getY(), ctype='cyan', size=4)
        plt.title('Diffexp Image and Source Catalog')

        plt.tight_layout()
        plt.show()
        
        
    return


def get_light_curve(repo, visits, collection_diff, collection_calexp, ccd_num, ra, dec, r, factor=None, cutout=40, save=False, title='', hist=False, sparse_obs=False, SIBLING=None, save_as='', do_lc_stars = False, nstars=10, seedstars=200, save_lc_stars = False, show_stamps=True, show_star_stamps=True, correct_coord=False, bs=531, box=100, do_zogy=False, collection_coadd=None, plot_zogy_stamps=False, plot_coadd=False, instrument='DECam'):
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
    None
    
    """
    
    fluxes = []
    fluxes_err = []
    new_err = []
    
    fluxes_cal = []
    fluxes_err_cal = []
    
    stars = pd.DataFrame()
    
    flags = []
    dates = []
    stats = {}
    pixel_to_arcsec = 0.2626 #value from Manual of NOAO - DECam
    
    r_in_arcsec = r 
    
    if type(r) != str:
        r/=pixel_to_arcsec



    butler = Butler(repo)
    if collection_coadd != None:
        print('Looking at coadd')
        dataIds = Find_coadd(repo, collection_coadd, ra, dec, instrument=instrument, plot=plot_coadd, cutout=cutout)
        coadd = butler.get('goodSeeingCoadd', collections=collection_coadd, instrument=instrument, dataId = dataIds[0])
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        coadd_cutout = coadd.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(cutout*2, cutout*2))

        
        wcs = coadd_cutout.getWcs()
        px_0, py_0 = wcs.getPixelOrigin()
        
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        print(np.asarray(coadd_cutout.variance.array, dtype='float'))
        #print(wcs.copyAtShiftedPixelOrigin())
        data = np.asarray(coadd_cutout.image.array, dtype='float')
        #plt.imshow(data, cmap='gray')
        #plt.show()
        #fig = plt.figure(figsize=(10, 5))
        #stamp_display = []

        #fig.add_subplot(1,2,1)
        #stamp_display.append(afwDisplay.Display(frame=fig))
        #stamp_display[0].scale('linear', -1, 10)
        #stamp_display[0].mtv(coadd_cutout)
        #stamp_display[0].dot('o', x_pix, y_pix, ctype='#0827F5', size=5)
        #plt.title('Coadd of source in ra {} dec {}'. format(ra,dec), fontsize=17)
        #plt.tight_layout()
        print('px {} , py {}'.format(x_pix, y_pix))
        print('px_0 {}, py_0 {}'.format(px_0, py_0))
        #print(np.shape(data))
        #print(np.s)
        flux_coadd, fluxerr_coadd, flag_coadd = sep.sum_circle(data, [cutout] , [cutout] , r, var = np.asarray(coadd_cutout.variance.array, dtype='float'))
        
        print(flux_coadd[0])
        plt.show()

    for i in range(len(visits)):
        obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')
        calexp = butler.get('calexp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam') 
        wcs = diffexp.getWcs()
        
        px = 2048
        py = 4096

        ra_center, dec_center = wcs.pixelToSkyArray([px/2], [py/2], degrees=True)
        ra_center = ra_center[0]
        dec_center = dec_center[0]
        #print('should be center of exposre')
        print('RA center : {} DEC center : {}'.format(ra_center, dec_center))

        #ExpTime = diffexp.getInfo().getVisitInfo().exposureTime 
        #gain = 4
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        data = np.asarray(diffexp.image.array, dtype='float' )
        data_cal = np.asarray(calexp.image.array, dtype='float' )
        print('xpix, y_pix: {} {} '.format(x_pix, y_pix))
        
        # Histogram of random part of image
        if hist:
            #x = 1250
            #y = 510 
            #x_half_width = 100
            #y_half_width = 100
            #x_lower = x - x_half_width
            #x_upper = x + x_half_width

            #y_lower = y - y_half_width
            #y_upper = y + y_half_width

            #bbox = lsst.geom.Box2I()
            #bbox.include(lsst.geom.Point2I(x_lower, y_lower))
            #bbox.include(lsst.geom.Point2I(x_upper, y_pix + y_upper))

            #diffexp_cutout = diffexp.getCutout(obj_pos_lsst, size=lsst.geom.Extent2I(200, 200))
            #fig = plt.figure(figsize=(10, 5))

            #stamp_display = []
            #fig.add_subplot(1,1,1)
            #stamp_display.append(afwDisplay.Display(frame=fig))
            #stamp_display[0].scale('linear', 'zscale')
            #stamp_display[0].mtv(diffexp_cutout.maskedImage)
            #plt.show()

            #data_onedim = np.ndarray.flatten(data[x_lower:x_upper, y_lower:y_upper])
            data_onedim = np.ndarray.flatten(data[50:-50, 50:-50])
            
            plt.hist(data_onedim, bins=500, histtype='step', log=True, color='#7A68A6')
            plt.title('Diffexp histogram', fontsize=15)
            stats['{}'.format(visits[i])] = data_onedim
            plt.show()

        ##############
        
        exp_visit_info = diffexp.getInfo().getVisitInfo()
        
        visit_date_python = exp_visit_info.getDate().toPython()
        visit_date_astropy = Time(visit_date_python)
        print(visit_date_astropy)            
        dates.append(visit_date_astropy.mjd)
        b = np.nan_to_num(np.array(data))
        wcs = diffexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)

        
        if correct_coord and i==0:
            print('before centroid correction: xpix, y_pix: {} {} '.format(x_pix, y_pix))
            #x_pix, y_pix = centroid_sources(data_cal, [x_pix], [y_pix], box_size=bs, centroid_func=centroid_com)

            sub_data = data_cal[int(y_pix)-box:box+int(y_pix),int(x_pix)-box:box+int(x_pix)]
            print(sub_data.shape)
            #print(sub_data)
            sub_data = sub_data.copy(order='C')
            objects = sep.extract(sub_data, 1000, minarea=5)
            
            obj = Select_largest_flux(sub_data, objects)
            ox = objects[:]['x']
            xc = float(obj['x'][0])
            yc = float(obj['y'][0])
            
            x_pix = xc + int(x_pix)-box #_pix[0]
            y_pix = yc + int(y_pix)-box #_pix[0]
            print('after centroid correction: xpix, y_pix: {} {} '.format(x_pix, y_pix))
            ra_cor, dec_cor =  wcs.pixelToSkyArray([x_pix], [y_pix], degrees=True)
            ra = ra_cor[0]
            dec = dec_cor[0]

      
            
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        psf = diffexp.getPsf()
        
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm * pixel_to_arcsec     
        
        if r == 'seeing':             
            r = seeing * factor

        

        print('Aperture radii: {} px'.format(r))
        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], r, var = np.asarray(diffexp.variance.array, dtype='float'))
       
        
        print('Coords: ra = {}, dec = {}'.format(ra,dec))
        print('visit : {}'.format(visits[i]))
        if show_stamps:
            Calib_and_Diff_plot_cropped(repo, collection_diff, collection_calexp, ra, dec, [visits[i]], ccd_num, s=r, cutout=cutout)
        #fluxes_under_aperture = values_under_aperture(data, x_pix, y_pix, r)
        #print(fluxes_under_aperture) 
        #error = np.sqrt(np.mean(fluxes_under_aperture**2)* len(fluxes_under_aperture)) 
        
        fluxes.append(flux[0])
        fluxes_err.append(fluxerr[0])
        
        print('flux: ', flux[0])
        print('fluxerr: ', fluxerr[0])
        flags.append(flags)
        print('------------------------------------------')

    if do_lc_stars == True:
        columns_stars = np.ndarray.flatten(np.array([['star_{}_f'.format(i+1), 'star_{}_ferr'.format(i+1)] for i in range(nstars)]))
        stars = pd.DataFrame(columns=columns_stars)
        py = 2048 - 100
        px = 4096 - 100
        width = px*pixel_to_arcsec
        height = py*pixel_to_arcsec
        print('width: {} , height : {}'.format(width, height))
        Table = Find_stars(ra_center, dec_center, width, height, n=nstars, seed=[do_lc_stars, seedstars])
        RA = np.array(Table['RA_ICRS'])
        DEC = np.array(Table['DE_ICRS'])
        #print(Table)
        
        
        for i in range(len(visits)):
            diffexp = butler.get('goodSeeingDiff_differenceExp',visit=visits[i], detector=ccd_num , collections=collection_diff, instrument='DECam')

            wcs = diffexp.getWcs()
            data = np.asarray(diffexp.image.array, dtype='float' )            
            
            flux_stars_and_errors = []
            star_aperture = 2 # arcsec 
            star_aperture/=pixel_to_arcsec # transform it to pixel values 
            saturated_stars = []
            for i in range(nstars):
                
                ra_star = RA[i]
                dec_star = DEC[i]
                print('⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂ (◕ᴥ◕) -. * STAR {} *.- (◕ᴥ◕) ⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑ '.format(i+1))
                
                obj_pos_lsst_star = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
                x_star, y_star = wcs.skyToPixel(obj_pos_lsst_star)  
                print('x_pix : {}  y_pix : {}'.format(x_star, y_star))
                
                if show_star_stamps:
                    Calib_and_Diff_plot_cropped(repo, collection_diff, collection_calexp, ra_star, dec_star, [visits[i]], ccd_num, s=star_aperture, cutout=80)
                
                
                f, f_err, fg = sep.sum_circle(data, [x_star], [y_star], star_aperture, var = np.asarray(diffexp.variance.array, dtype='float'))
                
                if f>2000:
                    
                    saturated_stars.append(i+1)
                    
                flux_stars_and_errors.append(f[0])
                flux_stars_and_errors.append(f_err[0])
                
                print('Flux star: {} Error flux: {}'.format(f[0], f_err[0]))
                
                print('꒰✩ ’ω`ૢ✩꒱ -------------------- ⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂⁑⁂')
            
            stars.loc[len(stars.index)] = flux_stars_and_errors            
    
    #print(stars)
    #plt.show()
        saturated_stars = np.unique(np.array(saturated_stars))
        plt.figure(figsize=(10,6))
        stars['dates'] = dates - min(dates)
        stars = stars.sort_values(by='dates')
        for i in range(nstars):

            f_star = np.array(stars['star_{}_f'.format(i+1)])
            f_star_err = np.array(stars['star_{}_ferr'.format(i+1)])
            Dates = np.array(stars['dates'])
            #norm = 0#np.mean(f_star)
            #std = np.std(f_star)
            new_dates = dates - min(dates)
            j, = np.where(saturated_stars==i+1)
            
            norm = matplotlib.colors.Normalize(vmin=0,vmax=32)
            c_m = matplotlib.cm.plasma
            # create a ScalarMappable and initialize a data structure
            s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            T = np.linspace(0,27,nstars)
            if len(j)==0:
                plt.errorbar(Dates, f_star, yerr= f_star_err, capsize=4, fmt='*-', label = 'star {}'.format(i+1), color = s_m.to_rgba(T[i]), alpha=0.8)
                

        #plt.yscale('log')
        plt.title('Stars LCs from {} and {} with Aperture {}"'.format(collection_diff[9:], ccd_name[ccd_num], star_aperture*pixel_to_arcsec))               
        plt.xlabel('MJD', fontsize=15)
        plt.ylabel('Excess Flux in arbitrary units', fontsize=15)
        plt.legend(loc=9, ncol=5)
    #plt.ylim(-1000,1000)
    if save_lc_stars:
        plt.savefig('light_curves/eridanus_{}_{}_random_Stars.pdf'.format(collection_diff[9:], ccd_name[ccd_num]), bbox_inches='tight')
    #plt.show()    
    
    if do_lc_stars==False:      
        plt.figure(figsize=(10,6))
        
    
    

    #plt.errorbar(dates - min(dates), fluxes/std - norm/std, yerr=new_err/std, capsize=4, fmt='s', ecolor='blue', color='orange', label ='LSST Science Pipelines -RMS error-')
    
   
    plt.figure(figsize=(10,6))   

    #plt.set_cmap("cool")
    if do_zogy:
        zogy = zogy_lc(repo, collection_calexp, collection_coadd, ra, dec, ccd_num, visits, r, instrument = 'DECam', plot_diffexp=plot_zogy_stamps, plot_coadd = plot_coadd, cutout=cutout)
        print(zogy)

        z_flux = zogy.flux - np.mean(zogy.flux)
        z_flux_norm = np.linalg.norm(np.array(z_flux))
        z_flux/=z_flux_norm
        z_ferr = zogy.flux_err
        z_ferr/=z_flux_norm
        plt.errorbar(zogy.dates, z_flux, yerr=z_ferr, capsize=4, fmt='s', label ='ZOGY Cáceres-Burgos', color='orange', ls ='dotted')

    if SIBLING!=None and type(SIBLING)==str:
        Jorge_LC = pd.read_csv(SIBLING, header=5)
        Jorge_LC = Jorge_LC[Jorge_LC['mjd']<57072] 
        
        if factor==0.5:
            mean = np.mean(Jorge_LC.aperture_flx_0)
            fluxes = Jorge_LC.aperture_flx_0 - mean
            #norm = np.linalg.norm(fluxes)
            norm = np.linalg.norm(np.array(Jorge_LC.aperture_flx_0))
            fluxes /= norm
            
            #
            #std = np.norm(Jorge_LC.aperture_flx_0)
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_0 - mean, yerr=Jorge_LC.aperture_flx_err_0,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            plt.errorbar(Jorge_LC.mjd- min(Jorge_LC.mjd), fluxes, yerr=Jorge_LC.aperture_flx_err_0/norm,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
        if factor==0.75:

            mean = np.mean(Jorge_LC.aperture_flx_1)
            fluxes = Jorge_LC.aperture_flx_1 - mean
            #norm = np.linalg.norm(fluxes)
            norm = np.linalg.norm(np.array(Jorge_LC.aperture_flx_1))
            fluxes /= norm
            #
            #std = np.std(Jorge_LC.aperture_flx_1)
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_1 - mean, yerr=Jorge_LC.aperture_flx_err_1,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), fluxes, yerr=Jorge_LC.aperture_flx_err_1/norm,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            
        if factor==1:
            mean = np.mean(Jorge_LC.aperture_flx_2)
            fluxes = Jorge_LC.aperture_flx_2 - mean
            norm = np.linalg.norm(np.array(Jorge_LC.aperture_flx_2))
            #norm = np.linalg.norm(fluxes)
            fluxes /= norm
            #
            #std = np.std(Jorge_LC.aperture_flx_2)
            plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), fluxes, yerr=Jorge_LC.aperture_flx_err_2/norm,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_2 - mean, yerr=Jorge_LC.aperture_flx_err_2,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
        if factor==1.25:
            #std = np.std(Jorge_LC.aperture_flx_3)
            mean = np.mean(Jorge_LC.aperture_flx_3)
            norm = np.linalg.norm(np.array(Jorge_LC.aperture_flx_3))
            fluxes = Jorge_LC.aperture_flx_3 - mean
            norm = np.linalg.norm(fluxes)
            fluxes /= norm
            #
            plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), fluxes, yerr=Jorge_LC.aperture_flx_err_3/norm,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_3 - mean, yerr=Jorge_LC.aperture_flx_err_3,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
        if factor==1.5:
            #std = np.std(Jorge_LC.aperture_flx_4)
            norm = np.linalg.norm(np.array(Jorge_LC.aperture_flx_4))
            mean = np.mean(Jorge_LC.aperture_flx_4)
            fluxes = Jorge_LC.aperture_flx_4 - mean
            #norm = np.linalg.norm(fluxes)
            fluxes /= norm
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_4 - mean, yerr=Jorge_LC.aperture_flx_err_4,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
            plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), fluxes, yerr=Jorge_LC.aperture_flx_err_4/norm,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
    
    sub_fluxes = fluxes - np.mean(fluxes)
    norm_sub_fluxes = sub_fluxes/np.linalg.norm(np.array(fluxes))
    print('norm: ',np.linalg.norm(fluxes) )
    #mean = np.mean(fluxes)
    #norm = np.linalg.norm(fluxes)
    source_of_interest = pd.DataFrame()
    source_of_interest['dates'] = dates - min(dates)
    source_of_interest['flux'] = norm_sub_fluxes #fluxes/norm - mean/norm
    source_of_interest['flux_err'] = fluxes_err/np.linalg.norm(np.array(fluxes)) #norm
    source_of_interest = source_of_interest.sort_values(by='dates')
    
    plt.errorbar(source_of_interest.dates, source_of_interest.flux, yerr=source_of_interest.flux_err, capsize=4, fmt='s', label ='AL Cáceres-Burgos', color='#0827F5', ls ='dotted')
    plt.ylabel('Excess Flux in arbitrary units', fontsize=15 )
    plt.xlabel('MJD', fontsize=15)

    plt.title('Aperture radii: {}", source {}'.format(r_in_arcsec, title), fontsize=15)
    plt.legend(ncol=5)
    
    if sparse_obs:
        plt.xscale('log')

    if save and save_as=='':
        plt.savefig('light_curves/eridanus_ra_{}_dec_{}.pdf'.format(ra,dec), bbox_inches='tight')
    
    if save and save_as!='':
        plt.savefig('light_curves/{}.pdf'.format(save_as), bbox_inches='tight')
    
    plt.show()
    
    
    if hist:
        for key in stats:
            plt.hist(stats[key], bins=500, histtype='step', log=True, label=key, alpha=0.4)
        plt.legend()
    

        
        
    return

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
    
    Table = result[0][random_indexes]
    return Table

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

def Select_largest_flux(data_sub,objects, na=6):
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






