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
from photutils.utils._convolution import _filter_data
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import HuberRegressor
from numpy.polynomial.chebyshev import chebvander2d, chebval2d
import time
import argparse

start_load = time.time()

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
main_root='/home/pcaceres'
sibling_allcand = pd.read_csv('{}/HiTS_data/SIBLING_allcand.csv'.format(main_root), index_col=0)



def magzero_func(x, mzero):
    return x + mzero

def huber_loss(params, x, y, delta=0.5):
    
    mzero = params[0]
    residuals = y - magzero_func(x, mzero)
    abs_res = np.abs(residuals)
    
    # Huber loss definition
    loss = np.where(abs_res <= delta,
                    0.5 * residuals**2,
                    delta * (abs_res - 0.5 * delta))
    return np.sum(loss)

#df_one_field = Vdata[(Vdata['field']=='Blind15A_05')]

def estimate_magzero(DF):
    for index, row in DF.iterrows():

        ra = row['raMedian']
        dec = row['decMedian']

        internalID = row['internalID']

        ccd = row['ccds']
        field = row['field']

        camp_2015 = row['camp_2015']
        camp_2014 = row['camp_2014']

        ccd_num = detector_nomenclature[ccd]

        print(ra, dec, field, ccd, ccd_num)

        images = []
        time_obs = []
        magzero = []
        airmass = []
        fwhm = [] # Median FWHM in pixels  
        wwcs = []
        visits = []
        exptimes = []
        coord = SkyCoord(ra * u.deg, dec * u.deg,frame='icrs') 

        if camp_2015==1:

            tag = 'v1'
            path_images_at_field = '/home/pcaceres/LSST_notebooks/Instcal_images_Bright_and_ExcessVar/' + field

        if camp_2014 ==1:

            tag = 'ls11'
            path_images_at_field = '/home/pcaceres/LSST_notebooks/Instcal_images_2014HiTS/' + field

        if not os.path.exists(path_images_at_field):

            print("This field is not downloaded. ")

            continue

        if os.path.exists('/home/pcaceres/LSST_notebooks/magzero_estimations/magzero_{}_{}_tag_{}.csv'.format(field, ccd, tag)):

            print("This magnitude is already estimated")

            continue

        for i in glob.glob(path_images_at_field + '/*_ooi_g_{}*'.format(tag)):

            prefix = i.split('_')[-3]
            print(i)
            try:
                hdul = fits.open(i)
                data = hdul[ccd_num-1].data

                images.append(data)
                time_obs.append(hdul[0].header['MJD-OBS'])
                airmass.append(hdul[0].header['AIRMASS'])
                fwhm.append(hdul[ccd_num-1].header['FWHM'])
                visits.append(hdul[0].header['EXPNUM'])
                exptimes.append(hdul[0].header['EXPTIME'])
                w = WCS(hdul[ccd_num-1].header)
                wwcs.append(w)
                c = w.pixel_to_world(40,40)
            except:
                print('bad image, so we skip it')
                pass

        weights = []
        date_weights = []
        for i in glob.glob(path_images_at_field+'/*_oow_g_{}*'.format(tag)):

            prefix = i.split('_')[-3]
            hdul = fits.open(i)
            data = hdul[ccd_num-1].data
            date_weights.append(hdul[0].header['MJD-OBS'])
            weights.append(data)

        masks = []
        date_masks = []
        for i in glob.glob(path_images_at_field+'/*_ood_g_{}*'.format(tag)):

            prefix = i.split('_')[-3]
            print(i)
            hdul = fits.open(i)
            data = hdul[ccd_num-1].data  
            date_masks.append(hdul[0].header['MJD-OBS'])
            masks.append(data)

        path_to_folder = '/home/pcaceres/LSST_notebooks/Results/DECam_pipeline/' + field + '_instcal'


        try:

            idx_sorted = np.argsort(np.array(time_obs))
            images = np.array(images)[idx_sorted]
            airmass = np.array(airmass)[idx_sorted]
            fwhm = np.array(fwhm)[idx_sorted]
            time_obs = np.array(time_obs)[idx_sorted]
            wwcs = np.array(wwcs)[idx_sorted]
            visits = np.array(visits)[idx_sorted]
            exptimes = np.array(exptimes)[idx_sorted]
            idx_masks = np.argsort(np.array(date_masks))
            masks = np.array(masks)[idx_masks]
            idx_weights = np.argsort(np.array(date_weights))
            weights = np.array(weights)[idx_weights]

        except IndexError:

            print('we dont have enough information, maybe the images necessary are not downloaded')

            continue


        if len(images) == 0:
            continue

        # Step 2: Align all by common dates
        common_dates = sorted(set(time_obs) & set(date_masks) & set(date_weights))

        # Mapping from date to index
        idx_time_map = {d: i for i, d in enumerate(time_obs)}
        idx_mask_map = {d: i for i, d in enumerate(date_masks)}
        idx_weight_map = {d: i for i, d in enumerate(date_weights)}

        # Build aligned arrays

        aligned_images = np.array([images[idx_time_map[d]] for d in common_dates])
        aligned_airmass = np.array([airmass[idx_time_map[d]] for d in common_dates])
        aligned_fwhm = np.array([fwhm[idx_time_map[d]] for d in common_dates])
        aligned_time_obs = np.array(common_dates)
        aligned_wwcs = np.array([wwcs[idx_time_map[d]] for d in common_dates])
        aligned_visits = np.array([visits[idx_time_map[d]] for d in common_dates])
        aligned_exptimes = np.array([exptimes[idx_time_map[d]] for d in common_dates])
        aligned_masks = np.array([masks[idx_mask_map[d]] for d in common_dates])
        aligned_weights = np.array([weights[idx_weight_map[d]] for d in common_dates])    

        # Step 3: Apply the date cut
        date_cut = aligned_time_obs < 57073
        px = 2048.0
        py = 4096.0

        info = pd.DataFrame(columns=['dates', 'magzero', 'nstars', 'visit'])

        for i in range(len(date_cut)):

            im = aligned_images[i]        
            ma = aligned_masks[i]
            wcs = aligned_wwcs[i]
            texp = aligned_exptimes[i]
            va = 1/aligned_weights[i]
            vi = aligned_visits[i]
            ny, nx = im.shape
            y = np.arange(ny)
            x = np.arange(nx)
            im_original = im
            im_original[ma>0] = 0 
            im_original[im_original<0] = 0
            sigma_clip_bkg = SigmaClip(sigma=3, maxiters=3)
            bkg_estimator = MedianBackground()
            bkg_science = Background2D(im_original, (100, 100), sigma_clip=sigma_clip_bkg, mask=ma,
                               bkg_estimator=bkg_estimator, edge_method='pad')

            background_model = bkg_science.background # chebval2d(X, Y, c)
            np.save("/home/pcaceres/LSST_notebooks/background_models/background_model_{}_{}_{}.npy".format(field, ccd, vi), background_model)

            im -= background_model
            im = np.ascontiguousarray(im)

            coords_center = wcs.pixel_to_world(px/2, py/2)
            ra_center = coords_center.ra.value
            dec_center = coords_center.dec.value
            print('ra_center: ', ra_center)
            print('dec_center: ', dec_center)

            coords_corner = wcs.pixel_to_world([px - 40.0, 40.0], [py - 40.0, 40.0])
            ra_corner = coords_corner.ra.value
            dec_corner = coords_corner.dec.value

            TAP_service = vo.dal.TAPService("https://mast.stsci.edu/vo-tap/api/v0.1/ps1dr2/")

            job = TAP_service.run_async("""
                SELECT objID, RAMean, DecMean, nDetections, ng, nr, ni, nz, ny, gPSFMag, gApMag, rPSFMag, iPSFMag, zPSFMag, yPSFMag, gKronMag, iKronMag, ipsfLikelihood
                FROM dbo.StackObjectView
                WHERE
                CONTAINS(POINT('ICRS', RAMean, DecMean), CIRCLE('ICRS', {}, {}, .13))=1
                AND nDetections >= 3
                AND ng > 0 AND ni > 0
                AND iPSFMag < 21 AND iPSFMag - iKronMag < 0.05 AND iPSFMag > 14
                  """.format(ra_center, dec_center))

            TAP_results = job.to_table()

            RA_stars = np.array(TAP_results['RAMean'])
            DEC_stars = np.array(TAP_results['DecMean'])
            log_abs_iPSF_likelihood = np.log10(np.abs(np.array(TAP_results['ipsfLikelihood'])))

            psf_gmag_stars = np.array(TAP_results['gPSFMag'])
            Ap_gmag_stars = np.array(TAP_results['gApMag'])

            idx = np.where((RA_stars>ra_corner.min()) & (RA_stars<ra_corner.max()) & (DEC_stars>dec_corner.min()) & (DEC_stars < dec_corner.max()))

            RA_stars = RA_stars[idx]
            DEC_stars = DEC_stars[idx]
            psf_gmag_stars = psf_gmag_stars[idx]
            Ap_gmag_stars = Ap_gmag_stars[idx]
            ra_inds_sort = RA_stars.argsort()
            RA_stars = RA_stars[ra_inds_sort[::-1]]
            DEC_stars = DEC_stars[ra_inds_sort[::-1]]
            psf_gmag_stars = psf_gmag_stars[ra_inds_sort[::-1]]
            Ap_gmag_stars = Ap_gmag_stars[ra_inds_sort[::-1]]

            stars_within_mags, = np.where((psf_gmag_stars>=15) & (psf_gmag_stars<=22)) 

            RA_stars = RA_stars[stars_within_mags]
            DEC_stars = DEC_stars[stars_within_mags]
            Ap_gmag_stars = Ap_gmag_stars[stars_within_mags]
            psf_gmag_stars = psf_gmag_stars[stars_within_mags]

            coords_stars = SkyCoord(RA_stars * u.deg, DEC_stars * u.deg, frame='icrs')
            pixels_stars = wcs.world_to_pixel(coords_stars)
            X_pix_stars = pixels_stars[0]
            Y_pix_stars = pixels_stars[1]        
            nstars = len(X_pix_stars)

            objects  = sep.extract(im, 10.6, var=va, mask=ma, minarea=3)

            x_pix_sep = objects['x']
            y_pix_sep = objects['y']

            x_pix_star_sep = []
            y_pix_star_sep = []
            Ap_gmag_stars_found = []

            for j in range(nstars):

                xps1 = X_pix_stars[j]
                yps1 = Y_pix_stars[j]

                d = ((xps1 - x_pix_sep)**2 + (yps1-y_pix_sep)**2)**(1/2)

                idx_star_pos = np.where(d<3)

                if len(idx_star_pos[0])==1:
                    x_pix_star_sep.append(x_pix_sep[idx_star_pos[0]])
                    y_pix_star_sep.append(y_pix_sep[idx_star_pos[0]])
                    Ap_gmag_stars_found.append(Ap_gmag_stars[j])

            ap_radii=5 # arcseconds
            arcsec_to_pixel = 0.27

            flux, fluxerr, flux_Flag = sep.sum_circle(im, x_pix_star_sep, y_pix_star_sep, 
                                                      ap_radii / arcsec_to_pixel, 
                                                      var=va, gain=4.0)

            Mags = np.array([-2.5*np.log10(f[0]) + 2.5 * np.log10(texp) for f in flux])

            #Mags_err = np.array([2.5 * ferr / (f * np.log(10)) for ferr, f in zip(fluxerr, flux)])

            idx_wout_nans_and_snabove5 = np.where((~np.isnan(Mags)) & ((flux/fluxerr).flatten() > 5) )

            Mags = Mags[idx_wout_nans_and_snabove5]

            #Mags_err = Mags_err[idx_wout_nans_and_snabove5].flatten()

            Ap_gmag_stars_found = np.array(Ap_gmag_stars_found)[idx_wout_nans_and_snabove5]         

            m_to_plot = np.linspace(Mags.min(), Mags.max(), 3)

            X = Mags.reshape(-1, 1) # sklearn expects 2D X
            y = Ap_gmag_stars_found

            res = minimize(huber_loss, x0=[0.0], args=(Mags, Ap_gmag_stars_found, 0.02))
            mzero_huber = res.x[0]

            new_row = {'dates':aligned_time_obs[i], 'magzero':mzero_huber, 'nstars':len(Ap_gmag_stars_found), 'visit':vi}   
            info.loc[len(info)] = new_row

        info.to_csv('/home/pcaceres/LSST_notebooks/magzero_estimations/magzero_{}_{}_tag_{}.csv'.format(field, ccd, tag), index=False)
        return 
    
if (__name__ == "__main__"):
    
    start = time.time()
    parser = argparse.ArgumentParser()
    Vdata = pd.read_csv('/home/pcaceres/spectral_search/SIBLING_nb/SIBLING_variable_galaxies_with_Jorges_criteria.csv')

    parser.add_argument('-fd', '--field', type=str, help='Field to estimate the zero point and store background', required = True)
    parser.add_argument('-ccd', '--detector_name', type=str, help='CCD we want to estimate the zero point and store background', required=False, default=None)
    
    args = parser.parse_args()
    
    print('on my way to estimate the magzeros for ', args.field)
    
    if args.detector_name is not None:
        DF_field = Vdata[(Vdata['field']==args.field) & (Vdata['ccds']==args.detector_name)]
    else:
        DF_field = Vdata[Vdata['field']==args.field]
    
    try:
        estimate_magzero(DF_field)
    except:
        print('this failed for some reason')
        pass
    
    tnew = time.time() 
    print('time that took to estimate the magzero of this field: ',tnew-start)
