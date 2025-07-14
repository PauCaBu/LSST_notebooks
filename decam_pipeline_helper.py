import lsst_pyhelper as lp
from lsst.daf.butler import Butler
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u 
import pandas as pd
import pyvo as vo
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from photutils import Background2D, MedianBackground
import sep
from matplotlib.colors import BoundaryNorm
import matplotlib
from matplotlib.cm import get_cmap
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')

detector_nomenclature= {'S29':1, 'S30':2, 'S31':3, 'S28':7, 'S27':6, 'S26':5, 'S25':4, 'S24':12, 
                        'S23':11, 'S22':10, 'S21':9, 'S20':8, 'S19':18, 'S18':17, 'S17':16, 
                        'S16':15, 'S15':14, 'S14':13, 'S13':24, 'S12':23, 'S11':22, 'S10':21, 
                        'S9':20,'S8':19, 'S7':31, 'S6':30, 'S5':29, 'S4':28, 'S3':27, 'S2':26, 
                        'S1':25, 'N29':60, 'N30':61, 'N31':62, 'N28':59, 'N27':58, 'N26':57, 
                        'N25':56, 'N24':55, 'N23':54, 'N22':53, 'N21':52, 'N20':51, 'N19':50, 
                        'N18':49, 'N17':48, 'N16':47, 'N15':46, 'N14':45, 'N13':44, 'N12':43, 
                        'N11':42, 'N10':41, 'N9':40,'N8':39, 'N7':38, 'N6':37, 'N5':36, 'N4':35, 
                        'N3':34, 'N2':33, 'N1':32 }

ccd_name = dict(zip(detector_nomenclature.values(), detector_nomenclature.keys()))



def retrieve_stars_common(image, x_pix_stars, y_pix_stars, worst_image, x_pix_stars_worst, y_pix_stars_worst, expimage_worst, expimage, maskimage, maskimage_worst):
    
    stars_in_common = pd.DataFrame()

    Updated_x_pix_stars = []
    Updated_y_pix_stars = []
    Updated_x_pix_stars_worst = []
    Updated_y_pix_stars_worst = []
    
    cutout = 20
    
    for i, (x,y,xw,yw) in enumerate(zip(x_pix_stars, y_pix_stars, x_pix_stars_worst, y_pix_stars_worst)):
        star_mask_ima = maskimage[round(y)-cutout:round(y)+cutout, round(x)-cutout:round(x)+cutout]
        star_mask_wima = maskimage_worst[round(yw)-cutout:round(yw)+cutout, round(xw)-cutout:round(xw)+cutout]
        
        if np.sum(star_mask_ima)>0 or np.sum(star_mask_wima)>0:
            continue
        else:
            Updated_x_pix_stars.append(x)
            Updated_y_pix_stars.append(y)
            Updated_x_pix_stars_worst.append(xw)
            Updated_y_pix_stars_worst.append(yw)
    
    stars_in_common['base_SdssCentroid_x_{}'.format(expimage)] = Updated_x_pix_stars
    stars_in_common['base_SdssCentroid_y_{}'.format(expimage)] = Updated_y_pix_stars
    stars_in_common['base_SdssCentroid_x_{}'.format(expimage_worst)] = Updated_x_pix_stars_worst
    stars_in_common['base_SdssCentroid_y_{}'.format(expimage_worst)] = Updated_y_pix_stars_worst
        
    return stars_in_common


arcsec_to_pixel=0.27

def get_light_curve(images, fwhm, wcs_im, ra, dec, dates, visits, weights, masks, magzeros, exptimes, 
                    airmass, ccd_num, field='', cutout=20, save=False, title='', hist=False, 
                    sparse_obs=False, SIBLING=None, save_as='', do_lc_stars = False, nstars=10, 
                    seedstars=200, save_lc_stars = False, show_stamps=True, show_star_stamps=False, 
                    r_star = 6, correct_coord=False, correct_coord_after_conv=False, do_zogy=False, 
                    collection_coadd=None, plot_zogy_stamps=False, plot_coadd=False, instrument='DECam', 
                    sfx='flx', save_stamps=False, well_subtracted=False, verbose=False, tp='after_ID',
                    area=None, thresh=None, mfactor=1, do_convolution=True, mode='Eridanus',
                    name_to_save='', type_kernel = 'mine', show_coord_correction=False, 
                    stars_from='lsst_pipeline', how_centroid = 'sep', 
                    path_to_folder= '/home/pcaceres/LSST_notebooks/Results/HiTS/SIBLING/', 
                    check_convolution=True,minarea=np.pi * (0.5/arcsec_to_pixel)**2, flux_thresh=None, 
                    ap_radii = np.array([0.5, 0.75, 1, 1.25, 1.5]), jname=None):
    
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
    
    # Here we find the image with the worst seeing, and the associated visit number
    # We retrieve the image that has the worst seeing
    
    round_magnitudes =  np.array([16, 17, 18, 19, 20, 21, 22])

    idx_worst = np.argmax(fwhm)
    worst_fwhm = fwhm[idx_worst]
    worst_wcs = wcs_im[idx_worst]
    worst_ima = images[idx_worst]
    worst_mask  = masks[idx_worst]
    
    #worst_bkg = sep.Background(worst_ima, mask=worst_mask)
    
    sigma_clip = SigmaClip(sigma=3., maxiters=2)
    bkg_estimator = MedianBackground()
    worst_bkg = Background2D(worst_ima, (100, 100), sigma_clip=sigma_clip, mask=worst_mask,
                           bkg_estimator=bkg_estimator, edge_method='pad')
    
    worst_ima = worst_ima - worst_bkg.background
    worst_ima = np.ascontiguousarray(worst_ima)
    
    
    worst_visit = visits[idx_worst]
    
    path_to_results = Path(path_to_folder)
    folder_field = path_to_results.joinpath('{}/'.format(field))
    folder_field.mkdir(parents=True, exist_ok=True)
    subfolder_detector_number = folder_field.joinpath('{}/'.format(ccd_name[ccd_num]))
    subfolder_detector_number.mkdir(parents=True, exist_ok=True)
    subsubfolder_source = subfolder_detector_number.joinpath('ra_{}_dec_{}/'.format(round(ra,3), round(dec,3)))
    subsubfolder_source.mkdir(parents=True, exist_ok=True)
    
    #if os.path.exists(subsubfolder_source / 'galaxy_LCs_dps.csv'):
    #    print('Already made a light curve so I skip it')
    #    return
    
    
    # Here we query or retrieve the stars we will measure the flux from
    if do_lc_stars:
        
        # ra dec of the center of the image... still dont remember what for 
        coords_center = worst_wcs.pixel_to_world(px/2, py/2)
        ra_center = coords_center.ra.value
        dec_center = coords_center.dec.value
        print('ra_center: ', ra_center)
        print('dec_center: ', dec_center)
    
        coords_corner = worst_wcs.pixel_to_world([px - 40.0, 40.0], [py - 40.0, 40.0])
        ra_corner = coords_corner.ra.value
        dec_corner = coords_corner.dec.value
        
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
        
        #print(TAP_results.columns)
        
        RA_stars = np.array(TAP_results['RAMean'])
        DEC_stars = np.array(TAP_results['DecMean'])
        psf_gmag_stars = np.array(TAP_results['gMeanPSFMag'])
        
        idx = np.where((RA_stars>ra_corner.min()) & (RA_stars<ra_corner.max()) & (DEC_stars>dec_corner.min()) & (DEC_stars < dec_corner.max()))
        
        RA_stars = RA_stars[idx]
        DEC_stars = DEC_stars[idx]
        psf_gmag_stars = psf_gmag_stars[idx]
        
        ra_inds_sort = RA_stars.argsort()
        RA_stars = RA_stars[ra_inds_sort[::-1]]
        DEC_stars = DEC_stars[ra_inds_sort[::-1]]
        psf_gmag_stars = psf_gmag_stars[ra_inds_sort[::-1]]
        
        print('number of stars: ', len(RA_stars))
        
        stars_within_mags, = np.where((psf_gmag_stars>=15) & (psf_gmag_stars<=22)) 
            
        RA_stars = RA_stars[stars_within_mags]
        DEC_stars = DEC_stars[stars_within_mags]
        psf_gmag_stars = psf_gmag_stars[stars_within_mags]
        
        print('number of stars within mags 15.5 and 20: ', len(RA_stars))
        
    ###### Here we set the pandas dataframe were the lightcurves are stored
    # flux_ImagDiff_nJy_0.5wseeing
    #print('worst seeing visit: ', worst_seeing_visit)
    #name_columns_imagdiff = ['flux_ImagDiff_nJy_{}sigmaPsf'.format(f) for f in ap_radii]
    #name_columns_imagdiffErr = ['fluxErr_ImagDiff_nJy_{}sigmaPsf'.format(f) for f in ap_radii]
    name_columns_convdown = ['flux_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_convdownErr = ['fluxErr_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_inst_convdown = ['instflux_ConvDown_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_inst_convdownErr = ['instfluxErr_ConvDown_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_inst = ['instflux_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_instErr = ['instfluxErr_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_mconvdown = ['mag_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    name_columns_mconvdownErr = ['magErr_ConvDown_nJy_{}_arcsec'.format(f) for f in ap_radii]
    
    columns = name_columns_convdown + name_columns_convdownErr + name_columns_mconvdown + name_columns_mconvdownErr + name_columns_inst + name_columns_instErr
    source_of_interest = pd.DataFrame(columns = columns)
    source_of_interest['dates'] = dates
    
    ##############
    
    coords_stars_worst = SkyCoord(RA_stars * u.deg, DEC_stars * u.deg, frame='icrs')
    pixels_stars_worst = worst_wcs.world_to_pixel(coords_stars_worst)
    X_pix_stars_worst = pixels_stars_worst[0]
    Y_pix_stars_worst = pixels_stars_worst[1]
    
    wim = check_psf(worst_ima, X_pix_stars_worst, Y_pix_stars_worst, worst_wcs, worst_bkg, 
                    worst_mask, plot=False)
    
    dates_to_plot = []
    # Here we loop over the images 
    visits_used = []
    
    for i in range(len(images)):

        data_cal = images[i]
        var_data_cal = 1/weights[i]
        fwhm_im = fwhm[i]
        mask_im = masks[i]
        mgzpt = magzeros[i] 
        texp = exptimes[i]
        m, s = np.mean(data_cal), np.std(data_cal)
        #bkg = sep.Background(data_cal, mask=mask_im)
        
        sigma_clip = SigmaClip(sigma=3., maxiters=2)
        bkg_estimator = MedianBackground()
        bkg_science = Background2D(data_cal, (100, 100), sigma_clip=sigma_clip, mask=mask_im,
                           bkg_estimator=bkg_estimator, edge_method='pad')
        data_cal = data_cal - bkg_science.background
        data_cal = np.ascontiguousarray(data_cal)
        #data_cal = data_cal - bkg # .background
        
        Data_science[visits[i]] = data_cal
        wcs_cal = wcs_im[i]
        
        print('-------- Looking at visit: ', visits[i])
        coord = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
        x_pix, y_pix = wcs_cal.world_to_pixel(coord) #wcs.skyToPixel(obj_pos_lsst)
        x_pix, y_pix = float(x_pix), float(y_pix)
        
        print('before == x_pix, y_pix: ', x_pix, y_pix)
        
        if (x_pix>px) or (x_pix<0) or (y_pix > py) or (y_pix<0):
            print('pixels are out of this image... skipping this observation')
            continue 
            
        if (x_pix<cutout) or (x_pix>px-cutout) or (y_pix<cutout) or (y_pix> py- cutout):
            cutout = int(np.min(np.array([x_pix, y_pix, px - x_pix, py - y_pix]))) - 1
            print('updating cutout value to: ', cutout)
            pass
            
        
        
        if correct_coord:
            try:
                x_pix, y_pix = lp.centering_coords(data_cal, x_pix, y_pix, cutout, show_coord_correction, how=how_centroid, minarea=minarea)
            except IndexError:
                x_pix, y_pix = [x_pix], [y_pix]
                pass
            
            new_coords = wcs_cal.pixel_to_world(x_pix, y_pix)
            ra = new_coords.ra.value
            dec = new_coords.dec.value
            
        else:
            x_pix, y_pix = [x_pix], [y_pix]
        
        print('x_pix, y_pix: ', x_pix, y_pix)
        
        coords_stars = SkyCoord(RA_stars * u.deg, DEC_stars * u.deg, frame='icrs')
        pixels_stars = wcs_cal.world_to_pixel(coords_stars)
        X_pix_stars = pixels_stars[0]
        Y_pix_stars = pixels_stars[1]        
        coords_science[visits[i]] = [x_pix, y_pix]
        
        
        
        
        #objects = sep.extract(data_cal, 10, minarea=3)
        #m, s = np.mean(data_cal), np.std(data_cal)
        if show_stamps:
            plt.figure(figsize=(6,6))
            plt.imshow(data_cal[round(y_pix[0])-cutout:round(y_pix[0])+cutout+1, round(x_pix[0])-cutout:round(x_pix[0])+cutout+1])
            plt.colorbar()
            plt.scatter(x_pix[0] - round(x_pix[0]) + cutout, y_pix[0] - round(y_pix[0]) + cutout, marker='x', color='r')
            plt.title('science image, with corrected centering')
            plt.show()
            #plt.figure(figsize=(10,6))
            #plt.imshow(data_cal)#[round(y_pix[0])-cutout:round(y_pix[0])+cutout+1, round(x_pix[0])-cutout:round(x_pix[0])+cutout+1])
            #plt.colorbar()
            #plt.title('science image, with corrected centering')
            #plt.show()
            
        print('sum of values within the cutout of the science image: ', np.sum(data_cal[round(y_pix[0])-cutout:round(y_pix[0])+cutout+1, round(x_pix[0])-cutout:round(x_pix[0])+cutout+1]))
        
        if np.sum(data_cal[round(y_pix[0])-cutout:round(y_pix[0])+cutout+1, round(x_pix[0])-cutout:round(x_pix[0])+cutout+1]) < 1:
            print('the stamp at the galaxy position is empty!')
            continue

        if do_convolution:

            im = check_psf(data_cal, X_pix_stars, Y_pix_stars, wcs_cal, bkg_science, mask_im, plot=False)
            
            # For the convolution we mask stars that are brighter than 20 mag
            stars_above_mag_20 = psf_gmag_stars < 20
            
            stars_in_common = retrieve_stars_common(data_cal, X_pix_stars[stars_above_mag_20], 
                                                    Y_pix_stars[stars_above_mag_20], worst_ima, 
                                                    X_pix_stars_worst[stars_above_mag_20],
                                                    Y_pix_stars_worst[stars_above_mag_20],
                                                    worst_visit, visits[i], mask_im, worst_mask)
            
            
            
            if len(stars_in_common)<1:
                print('length stars in common: ', len(stars_in_common))
                print('.... skipping this observation')
                KERNEL[visits[i]] = np.zeros(np.shape(im))
                Data_convol[visits[i]] = np.zeros(np.shape(data_cal)) # calConv_image
                coords_convol[visits[i]] = [np.array(x_pix), np.array(y_pix)]
                continue
            
            dates_to_plot.append(dates[i])

            calConv_image, calConv_variance, kernel = lp.do_convolution_image(data_cal, var_data_cal, im, 
                                                                              wim, mode=mode, 
                                                                              type_kernel=type_kernel, 
                                                                              visit=visits[i], 
                                                                              worst_visit=worst_visit, 
                                                                              stars_in_common=stars_in_common,
                                                                              worst_calexp=worst_ima)#, im=im, wim=wim)
            
            TOTAL_convolved_counts = np.sum(np.sum(calConv_image))
            #print('fraction of Flux lost after convolution: ',1-TOTAL_convolved_counts/TOTAL_counts)

            KERNEL[visits[i]] = kernel
            Data_convol[visits[i]] = calConv_image
           
            if correct_coord_after_conv:
                
                x_pix_conv, y_pix_conv = lp.centering_coords(calConv_image, x_pix[0], y_pix[0], cutout, 
                                                             True,  how=how_centroid, minarea=minarea)
                coords_convol[visits[i]] = [np.array(x_pix_conv), np.array(y_pix_conv)]
                
                if show_stamps:
                    plt.figure(figsize=(6,6))
                    plt.title('convolved image, with new centering')
                    plt.imshow(calConv_image[round(y_pix_conv[0])-cutout:round(y_pix_conv[0])+cutout+1, round(x_pix_conv[0])-cutout:round(x_pix_conv[0])+cutout+1])
                    plt.colorbar()
                    plt.scatter(x_pix_conv[0] - round(x_pix_conv[0]) + cutout, y_pix_conv[0] - round(y_pix_conv[0]) + cutout, marker='x', color='r')
                    plt.show()
            else:
                x_pix_conv = x_pix
                y_pix_conv = y_pix
                coords_convol[visits[i]] = [np.array(x_pix), np.array(y_pix)]
            print('-----------------')

            #if check_convolution:
        visits_used.append(visits[i])
        # return    
        Data_diff[visits[i]] = data_cal
        Data_coadd[visits[i]] = data_cal
        coords_coadd[visits[i]] = [x_pix_conv, y_pix_conv]
         
        flux_sci, fluxerr_sci, flag_sci = sep.sum_circle(data_cal, [x_pix], [y_pix], ap_radii / arcsec_to_pixel, var = var_data_cal) # fixed aperture 
        
        source_of_interest.loc[i,name_columns_inst] = flux_sci.flatten()
        source_of_interest.loc[i,name_columns_instErr] = fluxerr_sci.flatten()

        print('Aperture radii in px: ', ap_radii / arcsec_to_pixel)
        print('Aperture radii in arcsec: ', ap_radii)
        
        if do_convolution:
            
            dx_stamp = 75
            if (x_pix_conv[0]<dx_stamp) or (x_pix_conv[0]>px-dx_stamp) or (y_pix_conv[0]<dx_stamp) or (y_pix_conv[0]> py - dx_stamp):
                dx_stamp = int(np.min(np.array([x_pix_conv[0], y_pix_conv[0], px - x_pix_conv[0], py - y_pix_conv[0]]))) - 1
            
            convolved_stamp = calConv_image[round(y_pix_conv[0])-dx_stamp:round(y_pix_conv[0])+dx_stamp+1,round(x_pix_conv[0])-dx_stamp:round(x_pix_conv[0])+dx_stamp+1].copy(order='C')
            variance_stamp = calConv_variance[round(y_pix_conv[0])-dx_stamp:round(y_pix_conv[0])+dx_stamp+1,round(x_pix_conv[0])-dx_stamp:round(x_pix_conv[0])+dx_stamp+1].copy(order='C')
            
            try:
                sigma_clip = SigmaClip(sigma=3., maxiters=2)
                bkg_estimator = MedianBackground()
                bkg = Background2D(convolved_stamp, (10, 10), sigma_clip=sigma_clip,
                       bkg_estimator=bkg_estimator, edge_method='pad')
                data_image_wout_bkg = convolved_stamp - bkg.background
                
            except ValueError:
                
                data_image_wout_bkg = convolved_stamp
                print('background subtraction failed')    
                
            x_pix_stamp = x_pix_conv[0] - round(x_pix_conv[0]) + dx_stamp
            y_pix_stamp = y_pix_conv[0] - round(y_pix_conv[0]) + dx_stamp
                
            #print(np.shape(data_image_wout_bkg))
            #print(np.shape(convolved_stamp))
            #plt.imshow(data_image_wout_bkg)
            #plt.show()
            
            try:
                flux_conv, fluxerr_conv, flux_convFlag = sep.sum_circle(data_image_wout_bkg, [x_pix_stamp], 
                                                                    [y_pix_stamp], ap_radii / arcsec_to_pixel, 
                                                                    var=variance_stamp, gain=4.0)
            
                magsConv = [-2.5*np.log10(f) + 2.5 * np.log10(texp) + mgzpt for f in flux_conv]
                magsConv_err = [2.5 * ferr / (f * np.log(10)) for ferr, f in zip(fluxerr_conv, flux_conv)]

                source_of_interest.loc[i,name_columns_mconvdown] = np.array(magsConv).flatten()
                source_of_interest.loc[i,name_columns_mconvdownErr] = np.array(magsConv_err).flatten()
            
            
            except IndexError:
                
                source_of_interest.loc[i,name_columns_mconvdown] = np.empty(len(ap_radii))
                source_of_interest.loc[i,name_columns_mconvdownErr] = np.empty(len(ap_radii))
                
             #np.zeros(len(name_columns_mconvdownErr))
            
            
            #source_of_interest.loc[i,name_columns_mconvdown] = np.empty(len(ap_radii))
            #source_of_interest.loc[i,name_columns_mconvdownErr] = np.empty(len(ap_radii))
            #flux_conv, fluxerr_conv, flux_convFlag = 0, 0, 0
                
            #    data_image_wout_bkg = convolved_stamp
            #    bkg_estimator = MedianBackground()
            #    bkg = Background2D(convolved_stamp, sigma_clip=sigma_clip,
            #           bkg_estimator=bkg_estimator)
            #    print('....doing a less detailed background removal...')
            #    data_image_wout_bkg = convolved_stamp - bkg.background
                            
            #print('magsConv: ',  np.array(magsConv).flatten())
 
        #prof = lp.flux_profile_array(calConv_image, x_pix_conv, y_pix_conv, 0.05, 6)
        
        #profiles['{}'.format(visits[i])] = prof/max(prof)
        
        if do_lc_stars == True:
            
                
            nstars = len(X_pix_stars)
            print('Number of stars: ', nstars)
                
            # If its the first image loop, we create the pandas df
            star_aperture = r_star # arcseconds #r_star * fwhm/2
            star_aperture/=arcsec_to_pixel # transform it to pixel values 
            
            if i==0:
                #columns_stars_imagDiff_fluxes = list(np.ndarray.flatten(np.array([ 'star_{}_ImagDiff_fnJy'.format(i+1) for i in range(nstars)])))
                #columns_stars_imagDiff_fluxesErr = list(np.ndarray.flatten(np.array([ 'star_{}_ImagDiff_fnJy_err'.format(i+1) for i in range(nstars)])))
                
                columns_stars_convDown_instfluxes = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_instflx'.format(i+1) for i in range(nstars)])))
                columns_stars_convDown_instfluxesErr = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_instflx'.format(i+1) for i in range(nstars)])))
                
                #columns_stars_convDown_fluxes = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_fnJy'.format(i+1) for i in range(nstars)])))
                #columns_stars_convDown_fluxesErr = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_fnJy_err'.format(i+1) for i in range(nstars)])))
                
                columns_stars_convDown_mag = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_mag'.format(i+1) for i in range(nstars)])))
                columns_stars_convDown_magErr = list(np.ndarray.flatten(np.array([ 'star_{}_convDown_magErr'.format(i+1) for i in range(nstars)])))
                # columns_stars_convDown_fluxes + columns_stars_convDown_fluxesErr +
                stars_calc_byme = pd.DataFrame(columns= columns_stars_convDown_mag + columns_stars_convDown_magErr)
            
            saturated_star = []
            
            if do_convolution:
                
                # the centering I do not do it, but maybe I will have tooooo!
                fluxConv_nJy = []
                fluxerrConv_nJy = []
                
                magsConv_nJy = []
                magserrConv_nJy = []
                
                for k in range(nstars):
                    
                    x_pix_1star = X_pix_stars[k]
                    y_pix_1star = Y_pix_stars[k]
                    mask_star_cutout = mask_im[round(round(y_pix_1star)-star_aperture):round(round(y_pix_1star)+star_aperture), round(round(x_pix_1star)-star_aperture):round(round(x_pix_1star)+star_aperture)]#calexp.getCutout(obj_pos_lsst_star, size=lsst.geom.Extent2I(star_aperture, star_aperture))
                    
                    number_of_sat_pixels = np.sum(mask_star_cutout) #um(sum(np.array([[str_digit in str(element) for element in row] for row in calexp_star_cutout.getMask().array])))
                    
                    if number_of_sat_pixels > 0:
                        print('star with bad pixels, so we skip...')
                        fluxConv_nJy.append(np.nan)
                        fluxerrConv_nJy.append(np.nan)
                        magsConv_nJy.append(np.nan)
                        magserrConv_nJy.append(np.nan)
                        saturated_star.append(k)
                        continue
                    
                    # cutout_star = 23 // before
                    x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_1star, y_pix_1star, 13, show_stamps=False, how='sep', minarea=3)
                    
                    f_conv, ferr_conv, f_convFlag = sep.sum_circle(calConv_image, x_pix_new, y_pix_new, star_aperture, var=calConv_variance, gain=4.0)
                                                            
                    magsConv_nJy.append(-2.5*np.log10(f_conv[0]) + 2.5 * np.log10(texp) + mgzpt)
                    # [(2.5 * ferr / (f * np.log(10)))**(0.5) for ferr, f in zip(fluxerr_conv, flux_conv)]

                    magserrConv_nJy.append(2.5 * ferr_conv[0] / (f_conv[0] * np.log(10)))
                    
                    #if show_star_stamps:# and magsConv_to_nJy.value<=17.5:
                    #    
                    #    plt.imshow(np.arcsinh(calConv_image))
                    #    plt.xlim(x_pix_new - 23, x_pix_new + 23)
                    #    plt.ylim(y_pix_new - 23, y_pix_new + 23)
                    #    
                    #    plt.title('star number {} in convolved image'.format(k+1))
                    #    plt.colorbar()
                    #    plt.show()
                        
                        #plt.imshow(np.arcsinh(diffexp_calib_array))
                        #plt.xlim(x_pix_1star - 23, x_pix_1star + 23)
                        #plt.ylim(y_pix_1star - 23, y_pix_1star + 23)
                        #
                        #plt.title('star number {} in difference image'.format(k+1))
                        #plt.colorbar()
                        #plt.show()

                stars_calc_byme.loc[i,columns_stars_convDown_mag] = np.array(magsConv_nJy).flatten()
                stars_calc_byme.loc[i,columns_stars_convDown_magErr] = np.array(magserrConv_nJy).flatten()
                 
            if show_star_stamps:
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 42))
                ax1.axis('off')
                ax2.axis('off')
                #ax3.axis('off')
                m, s = np.mean(data_cal), np.std(data_cal)
                im1 = ax1.imshow(data_cal, vmin = m-s, vmax = m+s)
                im2 = ax2.imshow(calConv_image, vmin = m-s, vmax = m+s)
                ax1.set_title('Calibrated Image')
                ax2.set_title('Convolved Image')
                
                #md, sd = np.mean(diffexp_calib_array), np.std(diffexp_calib_array)
                #im3 = ax3.imshow(diffexp_calib_array, vmin=md-sd, vmax = md+sd)
                #ax3.set_title('Difference Image')
                
                for s in range(nstars):
                    x_star, y_star = X_pix_stars[s], Y_pix_stars[s]
                    
                    ax1.add_patch(plt.Circle((x_star, y_star), radius=star_aperture, color="red", fill=False))
                    ax1.text(x_star+star_aperture, y_star, '{}'.format(s+1), color="red")
                    
                    ax1.add_patch(plt.Circle((x_pix, y_pix), radius=star_aperture, color='m', fill=False))
                    ax1.text(x_pix+star_aperture, y_pix, 'Galaxy', color='m')
                    
                    ax2.add_patch(plt.Circle((x_star, y_star), radius=star_aperture, color="red", fill=False))
                    ax2.text(x_star+star_aperture, y_star, '{}'.format(s+1), color="red")
                    
                    ax2.add_patch(plt.Circle((x_pix_conv, y_pix_conv), radius=star_aperture, color='m', fill=False))
                    ax2.text(x_pix_conv+star_aperture, y_pix_conv, 'Galaxy', color='m')
                    #ax3.add_patch(plt.Circle((x_star, y_star), radius=star_aperture, color="green", fill=False))
                    #ax3.text(x_star+star_aperture, y_star, '{}'.format(s+1), color="green")
                    #
                    #ax3.add_patch(plt.Circle((x_pix, y_pix), radius=star_aperture, color='m', fill=False))
                    #ax3.text(x_pix+star_aperture, y_pix, 'Galaxy', color='m')
                plt.show()
    
    
    #print(profiles_stars)
    # plotting airmas, seeing & calibration factor ############################################

    plt.figure(figsize=(10,6))

    plt.plot(dates, np.array(airmass), 'o', color='magenta', linestyle='--', label='Airmass')

    plt.plot(dates, fwhm * arcsec_to_pixel, 'o', color='black', linestyle='--', label='FWHM (arcsec)')
     
    plt.xlabel('MJD', fontsize=17)
    plt.ylabel('In their given scale', fontsize=17)
    plt.legend(frameon=False)
    if save:
        plt.savefig(subsubfolder_source / 'airmass_seeing.jpeg', bbox_inches='tight')
    plt.show()
    
    ######################################################################################
    ######################################################################################
    ######################################################################################
    
    
    norm = matplotlib.colors.Normalize(vmin=0,vmax=nstars)
    c_m = matplotlib.cm.plasma

    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    magsconv_star_mean = np.array([np.nanmean(stars_calc_byme['star_{}_convDown_mag'.format(i+1)], dtype='float64') for i in range(nstars)])
    magsconv_star_std = np.array([np.nanstd(stars_calc_byme['star_{}_convDown_mag'.format(i+1)], dtype='float64') for i in range(nstars)])
    
    
    # We only take the stars that were correctly subtracted in the image difference 
    # filtering stars ! ¨good stars¨ do not increase their flux to more than half their median maximally
    
    # keep_stars, = np.where(fluxconv_star_max/fluxconv_star_median < 1.5)
    
    # saturated_star
    
    # ids_good_stars = np.array([item for item in range(nstars) if item not in saturated_star])
    
    # print('saturated star: ', saturated_star)
    ##################################################################################
    
    #mags_good_stars = magsconv_star_mean[ids_good_stars]
    #print('mags good stars: ',  mags_good_stars)
    #print('indexes where there is nan: ', np.where(np.isnan(mags_good_stars)))
    
    idx_nan_mags, = np.where(np.isnan(magsconv_star_mean))
    idx_var_mags, = np.where(magsconv_star_std > 1)
    print('these stars are >1 mag variable: ', idx_var_mags)
    id_good_stars = np.array([item for item in range(nstars) if item not in idx_nan_mags])
    id_good_stars = np.array([item for item in id_good_stars if item not in idx_var_mags])
    id_good_stars = np.array([item for item in id_good_stars if item not in saturated_star])
    
    mags_good_stars = magsconv_star_mean[id_good_stars]
    psf_gmag_stars_good = psf_gmag_stars[id_good_stars]
    RA_stars = RA_stars[id_good_stars]
    DEC_stars = DEC_stars[id_good_stars]
    
     
    
    closest_indices = np.array([np.argmin(np.abs(mags_good_stars - mag)) for mag in round_magnitudes])
    column_w_mags = 'mag_ConvDown_nJy_{}_arcsec'.format(1.0)
    mean_conv_mag = np.nanmean(np.array(source_of_interest[column_w_mags]), dtype='float64')
    ids_stars_within_gal_mag, = np.where((magsconv_star_mean >= mean_conv_mag-0.5) & (magsconv_star_mean < mean_conv_mag+0.5))
    ids_stars_within_gal_mag = np.array([item for item in ids_stars_within_gal_mag if item in id_good_stars])
    
    #print('magnitude within galaxy mag: ', mags_good_stars[ids_stars_within_gal_mag])
    n_stars_within_gal_mag = len(ids_stars_within_gal_mag)
    
    print('n_stars_within_gal_mag : ',n_stars_within_gal_mag )
    print('ids_stars_within_gal_mag : ',ids_stars_within_gal_mag )
    print('mean conv mag of GALAXY: ', mean_conv_mag)
    
    print('len(RA_stars): ', len(RA_stars))
    print('len(DEC_stars): ', len(DEC_stars))
    #print('len(psf_gmag_stars_good): ', len(psf_gmag_stars_good))
    print('closest_indeces: ', closest_indices)
    
    stars_science_mag_columns = ['star_{}_convDown_mag'.format(i+1) for i in ids_stars_within_gal_mag]
    stars_science_mag = stars_calc_byme[stars_science_mag_columns] #.dropna(axis=1, how='any')
    print('stars science magnitude df: ')
    print('stars within gal magnitude : ', ids_stars_within_gal_mag + 1)
    print('average magnitude: ', stars_science_mag.mean())
    stars_science_mag -= stars_science_mag.mean(skipna=True)
    #print('stars_calc_byme: ', stars_calc_byme)
    print('stars_science_mag : ', stars_science_mag)
    stars_science_disp = stars_science_mag.std(axis=1, skipna=True) #np.array([float(np.std(np.array(stars_science_mag.loc[i]))) for i in range(len(stars_science_mag))])
    

    for ii in range(len(images)):
        # de 0 a 6
        wcs_cal = wcs_im[ii]
        try:
            calConv_image = Data_convol[visits[ii]]
        except KeyError:
            continue
            
        idx_star = closest_indices[0]
        coord_star = SkyCoord(RA_stars[idx_star] * u.deg, DEC_stars[idx_star] * u.deg, frame='icrs')
        pixel_star = wcs_cal.world_to_pixel(coord_star)
        x_pix_star = pixel_star[0]
        y_pix_star = pixel_star[1]
        x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_star, y_pix_star, 23, show_stamps=False, how='sep', minarea=3)        
        prof = lp.flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[0], visits[ii])] = prof/max(prof)
        
        idx_star = closest_indices[1] 
        coord_star = SkyCoord(RA_stars[idx_star] * u.deg, DEC_stars[idx_star] * u.deg, frame='icrs')
        pixel_star = wcs_cal.world_to_pixel(coord_star)
        x_pix_star = pixel_star[0]
        y_pix_star = pixel_star[1]
        x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_star, y_pix_star, 23, show_stamps=False, how='sep', minarea=3)
        prof = lp.flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[1], visits[ii])] = prof/max(prof)
        
        
        idx_star = closest_indices[2] 
        coord_star = SkyCoord(RA_stars[idx_star] * u.deg, DEC_stars[idx_star] * u.deg, frame='icrs')
        pixel_star = wcs_cal.world_to_pixel(coord_star)
        x_pix_star = pixel_star[0]
        y_pix_star = pixel_star[1]
        x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_star, y_pix_star, 23, show_stamps=False, how='sep', minarea=3)
        prof = lp.flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[2], visits[ii])] = prof/max(prof)
        
        
        idx_star = closest_indices[3]
        coord_star = SkyCoord(RA_stars[idx_star] * u.deg, DEC_stars[idx_star] * u.deg, frame='icrs')
        pixel_star = wcs_cal.world_to_pixel(coord_star)
        x_pix_star = pixel_star[0]
        y_pix_star = pixel_star[1]
        x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_star, y_pix_star, 23, show_stamps=False, how='sep', minarea=3)
        prof = lp.flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[3], visits[ii])] = prof/max(prof)
        
        idx_star = closest_indices[4]
        coord_star = SkyCoord(RA_stars[idx_star] * u.deg, DEC_stars[idx_star] * u.deg, frame='icrs')
        pixel_star = wcs_cal.world_to_pixel(coord_star)
        x_pix_star = pixel_star[0]
        y_pix_star = pixel_star[1]
        x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_star, y_pix_star, 23, show_stamps=False, how='sep', minarea=3)
        prof = lp.flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[4], visits[ii])] = prof/max(prof)
        
        
        idx_star = closest_indices[5]
        coord_star = SkyCoord(RA_stars[idx_star] * u.deg, DEC_stars[idx_star] * u.deg, frame='icrs')
        pixel_star = wcs_cal.world_to_pixel(coord_star)
        x_pix_star = pixel_star[0]
        y_pix_star = pixel_star[1]
        x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_star, y_pix_star, 23, show_stamps=False, how='sep', minarea=3)
        prof = lp.flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[5], visits[ii])] = prof/max(prof)
                
        idx_star = closest_indices[6]
        coord_star = SkyCoord(RA_stars[idx_star] * u.deg, DEC_stars[idx_star] * u.deg, frame='icrs')
        pixel_star = wcs_cal.world_to_pixel(coord_star)
        x_pix_star = pixel_star[0]
        y_pix_star = pixel_star[1]
        x_pix_new, y_pix_new = lp.centering_coords(calConv_image, x_pix_star, y_pix_star, 23, show_stamps=False, how='sep', minarea=3)
        prof = lp.flux_profile_array(calConv_image,  x_pix_new[0], y_pix_new[0], 0.05, r_star)
        profiles_stars['mag{}_{}'.format(round_magnitudes[6], visits[ii])] = prof/max(prof)
    
    print('good_stars: ', id_good_stars)
    
    # To plot the variability of the fluxes of stars in the convolved image :
    Results_star = pd.DataFrame(columns = ['date', 'stars_scienceMag_1sigmaupp_byEpoch', 'stars_scienceMag_2sigmaupp_byEpoch', 'stars_scienceMag_1sigmalow_byEpoch', 'stars_scienceMag_2sigmalow_byEpoch'])
    
    #print('stars science dispersion: ', stars_science_disp)
    
    stars_sciencemag_1sigmalow_byEpoch = - stars_science_disp
    stars_sciencemag_1sigmaupp_byEpoch = stars_science_disp
    stars_sciencemag_2sigmalow_byEpoch = - 2 * stars_science_disp
    stars_sciencemag_2sigmaupp_byEpoch = 2 * stars_science_disp
    
    Results_star['date'] = dates
    
    Results_star['stars_scienceMag_1sigmalow_byEpoch'] = stars_sciencemag_1sigmalow_byEpoch
    Results_star['stars_scienceMag_1sigmaupp_byEpoch'] = stars_sciencemag_1sigmaupp_byEpoch
    Results_star['stars_scienceMag_2sigmalow_byEpoch'] = stars_sciencemag_2sigmalow_byEpoch
    Results_star['stars_scienceMag_2sigmaupp_byEpoch'] = stars_sciencemag_2sigmaupp_byEpoch
    
    print(Results_star)
    
    #########################################
    
    ## Here we plot the mags light curves of stars 
    
    plt.figure(figsize=(10,6))
    plt.title('stars LCs of mags in convolved science image from {} and {} with ap. radii of {}", within mags {} $\pm$ 0.5'.format(field, ccd_name[ccd_num], star_aperture*arcsec_to_pixel, round(mean_conv_mag,2)))
    
    norm = matplotlib.colors.Normalize(vmin=min(magsconv_star_mean),vmax=max(magsconv_star_mean))
    c_m = matplotlib.cm.plasma

    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    for i in ids_stars_within_gal_mag:
        print('plotting mags of stars: ', i+1)
        
        fs_star = (np.array(stars_calc_byme['star_{}_convDown_mag'.format(i+1)])).flatten()
        fs_star_err = np.ndarray.flatten(np.array(stars_calc_byme['star_{}_convDown_magErr'.format(i+1)]))
        
        fs_star = np.array([float(x) if str(x) != 'nan' else np.nan for x in fs_star])
        fs_star_err = np.array([float(x) if str(x) != 'nan' else np.nan for x in fs_star_err])
        
        mask = ~np.isnan(fs_star) & ~np.isnan(fs_star_err)
        
        if np.sum(mask) < 2:
            continue  # skip stars with too few valid points

        stars_yarray = fs_star[mask] - np.nanmean(fs_star, dtype=np.float64)
        mean_mag = np.nanmean(fs_star, dtype=np.float64)

        plt.errorbar(np.array(dates_to_plot)[mask], stars_yarray, yerr=fs_star_err[mask],
                     capsize=4, fmt='s', ls='solid', 
                     color=s_m.to_rgba(mean_mag), label=f'Star {i+1}')

        for w, label in enumerate(np.ones(np.sum(mask))*(int(i+1))):
            plt.annotate(str(int(label)), (np.array(dates_to_plot)[mask][w], stars_yarray[w]), color='green')
        
        
        #stars_yarray = np.array(fs_star - np.nanmean(fs_star, dtype=np.float64))
        #plt.errorbar(dates_to_plot, stars_yarray, yerr= fs_star_err, capsize=4, fmt='s', ls='solid', color = s_m.to_rgba(np.nanmean(fs_star, dtype=np.float64)))
        #marker_labels = np.ones(len(dates_to_plot))*(int(i+1))
        #for w, label in enumerate(marker_labels):
        #    plt.annotate(str(int(label)), (dates_to_plot[w], stars_yarray[w]), color='green')
    
    plt.xlabel('MJD', fontsize=15)
    plt.ylabel('$mag$ [AB] - mean(mag)', fontsize=15)
    plt.colorbar(s_m, label = 'Mean mag [AB]')
    #plt.legend()
    
    if save:
        plt.savefig(subsubfolder_source / 'stars_mags_convDown_LCs.jpeg', bbox_inches='tight')    
    # Here we plot the std deviation of stars"
    plt.show()
    
    
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Average dispersion of magnitude as a function of magnitude', fontsize=15)
    
    evaluate_mags = np.array([14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5])
    stars_at_given_mag = [id_good_stars[np.where((mags_good_stars>=evaluate_mags[k]) & (mags_good_stars<evaluate_mags[k+1]))] for k in range(len(evaluate_mags)-1)]
    
    # mags_good_stars
    std_at_given_mag = []
    eval_mag = []
    n_stars_mag = []
    try:
        for i in range(len(evaluate_mags)-1):
            eval_mag.append((evaluate_mags[i]+evaluate_mags[i+1])*0.5)
            stars_science_given_mag_columns = ['star_{}_convDown_mag'.format(j+1) for j in stars_at_given_mag[i]]
            n_stars_mag.append(len(stars_at_given_mag[i]))
            stars_science_at_given_mag = stars_calc_byme[stars_science_given_mag_columns]
            stars_science_at_given_mag -= stars_science_at_given_mag.mean()
            stars_sciencemag_1sigma_disp_byEpoch = np.array(stars_science_at_given_mag.std()) #np.array([np.nanstd(np.array(stars_science_at_given_mag.loc[i])) for i in range(len(stars_science_at_given_mag))])
            std_at_given_mag.append(np.mean(stars_sciencemag_1sigma_disp_byEpoch))
            ax.text(eval_mag[i]+0.2, 0, str(len(stars_at_given_mag[i])))

        ax.errorbar(eval_mag, np.zeros(len(eval_mag)), yerr=std_at_given_mag, xerr=1.0, color='m', capsize=3, fmt='o', ls=' ')
        ax.set_xlabel('$m_g$', fontsize=15)
        ax.set_ylabel('$Average \Delta m_g$', fontsize=15)
        plt.show()
    except:
        pass
    
    ########################################################################################
    ####################### Conv downgrade LCs ############################################
    ########################################################################################
    
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Convolution downgrade (mags) Light curves', fontsize=17)  
    
    norm = matplotlib.colors.Normalize(vmin=0.5,vmax=1.5)
    c_m = matplotlib.cm.magma

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    max_sep = 0 
    
    for fa in ap_radii:
        if do_convolution:
            
            column_to_take = 'mag_ConvDown_nJy_{}_arcsec'.format(fa)
            column_err_to_take = 'magErr_ConvDown_nJy_{}_arcsec'.format(fa)
            
            conv_mag = np.array(source_of_interest[column_to_take])
            conv_magerr = np.array(source_of_interest[column_err_to_take])
            #print('conv_mag: ', conv_mag)
            mean_mag = np.nanmean(conv_mag, dtype='float64')
            
            ax.errorbar(source_of_interest.dates, conv_mag - mean_mag, yerr=conv_magerr, capsize=4, fmt='o', color=s_m.to_rgba(fa), ls='-')
            
            if SIBLING!=None:
                x, y, yerr = lp.compare_to(SIBLING, sfx='mag', factor=fa, beforeDate=57073)
                ax.errorbar(x, y -  np.mean(y), yerr=yerr,  capsize=4, fmt='^', color=s_m.to_rgba(fa), markeredgewidth=3, markerfacecolor='None', ls ='dotted')
            
            #print(np.max(conv_mag - mean_mag))
            
            if max_sep <= max([np.max(np.abs(conv_mag - mean_mag)), np.max(np.abs(y -  np.mean(y)))]):
                max_sep = max([np.max(np.abs(conv_mag - mean_mag)), np.max(np.abs(y -  np.mean(y)))])
            
    # This plots are for the labels
    if SIBLING!=None:
        ax.errorbar([-4,-4],[0,0],capsize=4, fmt='^', color='k', markeredgewidth=3, markerfacecolor='None', ls ='dotted', label='Martinez-Palomera et al. 2020')
    ax.errorbar([-4,-4],[0,0],capsize=4, fmt='o', color='k', markeredgewidth=3, ls ='-', label='Caceres-Burgos in prep')
    
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    cbar = plt.colorbar(s_m, cax=cbar_ax,  boundaries =np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1),
                        ticks = np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1)[:-1] + np.diff(np.linspace(min(ap_radii),max(ap_radii),len(ap_radii)+1))/2)
    cbar.set_ticklabels(ap_radii.astype('str'))
    
    if do_lc_stars:
        ax.fill_between(dates_to_plot, stars_sciencemag_1sigmalow_byEpoch, stars_sciencemag_1sigmaupp_byEpoch, alpha=0.1, color='b', label = '1-2 $\sigma$ dev of {} $\pm$ 0.5 mag'.format(round(mean_conv_mag,2))) #
        ax.fill_between(dates_to_plot, stars_sciencemag_2sigmalow_byEpoch, stars_sciencemag_2sigmaupp_byEpoch, alpha=0.1, color='b')  
        source_of_interest['stars_dev_within_0p5_galmag'] = stars_sciencemag_1sigmalow_byEpoch
        
    ax.axhline(0, ls='--', color='gray')
    ax.set_xlabel('MJD', fontsize=15)
    ax.set_ylabel('m_g - mean(m_g)', fontsize=15)
    ax.set_xlim(min([min(source_of_interest.dates), min(x)]) - 0.05, max([max(source_of_interest.dates), max(x)]) + 0.05)
    ax.set_ylim(-max_sep - 0.05, max_sep + 0.05)
    
    ax.legend(frameon=False, ncol=2, loc='upper left')
    if save:
        plt.savefig(subsubfolder_source / 'Convolution_downgrade_LCs.jpeg', bbox_inches='tight')
    
    ##### Calculate Excess Variance ###########
    
    sigma_rms_sq, errsigma_rms_sq, sigma_rms_subtracted = lp.Excess_variance(np.array(source_of_interest['mag_ConvDown_nJy_1.0_arcsec']), np.array(source_of_interest['magErr_ConvDown_nJy_1.0_arcsec']))
    
    print('sigma_rms_sq, errsigma_rms_sq, sigma_rms_subtracted: ', sigma_rms_sq, errsigma_rms_sq, sigma_rms_subtracted)
    
    if (mode=='HiTS' or mode=='HITS') and SIBLING is not None and jname is not None:
        
        x, m, merr = lp.compare_to(SIBLING, sfx='mag', factor=0.75)
        sibling_dataset.loc[jname, 'Excess_variance_pcb_wkernel_{}'.format(type_kernel)] = sigma_rms_sq
        sibling_dataset.loc[jname, 'Excess_variance_e_pcb_wkernel_{}'.format(type_kernel)] = errsigma_rms_sq
        sibling_dataset.loc[jname, 'Excess_variance_cor_pcb_wkernel_{}'.format(type_kernel)] = sigma_rms_subtracted    
        sigma_rms_sq_jge, errsigma_rms_sq_jge, sigma_rms_subtracted_jge = Excess_variance(y, yerr)
        sibling_dataset.loc[jname, 'Excess_variance_jge'] = sigma_rms_sq_jge
        sibling_dataset.loc[jname, 'Excess_variance_e_jge'] = errsigma_rms_sq_jge
        sibling_dataset.loc[jname, 'Excess_variance_cor_jge'] = sigma_rms_subtracted_jge
        sibling_dataset.to_csv('SIBLING_sources_usingMPfilter_andPCB_comparison.csv')
    
    ########################################################################################
    ###############################  Profiles ##############################################
    ########################################################################################
    
    #fig = plt.figure(figsize=(10,6))
    #ax = fig.add_subplot(111)

    #norm = matplotlib.colors.Normalize(vmin=min(fwhm),vmax=max(fwhm))
    #c_m = matplotlib.cm.magma

    ## create a ScalarMappable and initialize a data structure
    #s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    #s_m.set_array([])
    #
    #for i, v in enumerate(visits):
    #    ax.plot(np.linspace(0.05, 6 , 15), profiles['{}'.format(v)]/profiles['{}'.format(worst_visit)], label=str(round(min(dates),2)) + ' + {}'.format(dates[i] - min(dates)), color=s_m.to_rgba(fwhm[i]))
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
    #    plt.savefig(subsubfolder_source / 'curve_growth_of_galaxy.jpeg', bbox_inches='tight')

    #plt.show()
    
    lp.plot_star_profiles(profiles_stars, round_magnitudes, visits,  np.linspace(0.05, 6, 15), worst_visit, fwhm * arcsec_to_pixel, save_as = subsubfolder_source / 'curve_growth_stars.jpeg' )
    
    print('coords convol: ', coords_convol)
    
    lp.stamps_instcal(Data_science, Data_convol, coords_science, coords_convol, source_of_interest,
                      Results_star, visits_used, KERNEL, fwhm, SIBLING = SIBLING, cut_aux=cutout, 
                      r_diff = fwhm_im * np.array([0.75]), r_science=worst_fwhm * np.array([0.75]),  
                      field='', name='', first_mjd = 58810, folder=subsubfolder_source)        

    if save:
        source_of_interest.to_csv(subsubfolder_source / 'galaxy_LCs_dps.csv')
                                  
    print(source_of_interest)
    return source_of_interest 

def check_psf(data_cal, stars_x_pix, star_y_pix, wcs_cal, bkg, mask, plot=False, cutout=24, cutout_for_psf=12):
   
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
    
    arcsec_to_pixel = 0.27 #arcsec/pixel

    #print('number of available stars: ', len(stars_table))
    
    RAs = []
    DECs = []
    starn = 0
    collected_stars = {}
    centroid_stars = {}
    sum_stars = np.zeros((cutout_for_psf*2+1, cutout_for_psf*2+1))
    sum_stars_conv = np.zeros((cutout_for_psf*2+1, cutout_for_psf*2+1))
    
    number_stars = 0
    
    for x, y in zip(stars_x_pix, star_y_pix):
        
        #print('looking at star in position: ', x, y)
        number_stars+=1
        x_pix_new, y_pix_new = lp.centering_coords(data_cal, x, y, int(cutout/2), show_stamps=False, how='sep', minarea=3, flux_thresh=1.6)
        calexp_cutout_to_use = data_cal[round(y_pix_new[0])-cutout_for_psf:round(y_pix_new[0])+cutout_for_psf+1, round(x_pix_new[0])-cutout_for_psf:round(x_pix_new[0])+cutout_for_psf+1]
        
        mask_of_star_stamp = mask[round(y_pix_new[0])-cutout_for_psf:round(y_pix_new[0])+cutout_for_psf+1, round(x_pix_new[0])-cutout_for_psf:round(x_pix_new[0])+cutout_for_psf+1]
        #flux_diff, fluxerr_diff, flag_diff = sep.sum_circle(diffexp_calib_array, [x_pix], [y_pix], seeing * sigma2fwhm * ap_radii, var = np.asarray(diffexp_calibrated.variance.array, dtype='float')) 
        
        # before I was using sep way of estimatin the background: bkg.globalrms
        f, ferr, flag = sep.sum_circle(data_cal, [x], [y], 10, err=np.ascontiguousarray(bkg.background_rms), gain=4)
        #print('Signal to noise: ',  f/ferr)
        
        if f/ferr<20 or np.sum(mask_of_star_stamp)>0:
            continue
        
        
        sum_stars += calexp_cutout_to_use
        #print('suma masks: ', np.sum(mask_of_star_stamp))
        
        if plot:

            fig = plt.figure(figsize=(8, 5))
            m, s = np.mean(calexp_cutout_to_use), np.std(calexp_cutout_to_use)
            plt.imshow(calexp_cutout_to_use, cmap='rocket', origin='lower', vmin=m, vmax=m+2*s)
            plt.colorbar()
            plt.scatter(cutout_for_psf, cutout_for_psf, color='green', marker='x', linewidth=3, label='rounded center')
            plt.title('star number {} in sicence after centering'.format(number_stars), fontsize=15)
            plt.tight_layout()
            plt.legend()
            plt.show()
    
    stars_sum_norm = sum_stars/np.sum(np.sum(sum_stars))
        
    return stars_sum_norm
