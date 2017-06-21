'''
This take the power spectra cross with CMB lensign and auto of different surveys ans spits out their rho as in Shwerwin Smithful
'''

import numpy as np
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import pickle


def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      - beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             - maximum multipole.
    """
    ls = np.arange(0, lmax + 1)
    return np.exp(-(fwhm_arcmin * np.pi / 180. / 60.)**2 / (16. * np.log(2.)) * ls * (ls + 1.))


def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
          * noise_uK_arcmin - map noise level in uK.arcmin
          * fwhm_arcmin     - beam full-width-at-half-maximum (fwhm) in arcmin.
          * lmax            - maximum multipole.
    """
    return (noise_uK_arcmin * np.pi / 180. / 60.)**2 / bl(fwhm_arcmin, lmax)**2


def main(labels, cmb):

    # cosmosis_dir = '/home/manzotti/cosmosis/'
    # inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
    # Config_ini = ConfigParser.ConfigParser()
    # values = ConfigParser.ConfigParser()
    # Config_ini.read(inifile)
    # values_file = Config_ini.get('pipeline', 'values')
    output_dir = '../Data/'
    # values.read(cosmosis_dir + values_file)
    cls = pickle.load(open('../Data/limber_spectra_delens.pkl', 'rb'))
    lbins = np.load('../Data/ells.npy')

    ckk = cls['kk']
    # initialize
    rho_comb = np.zeros((np.size(lbins)))
    surveys = labels
    # surveys = ['wise', 'k', 'euclid', 'des_weak', 'lsst', 'ska10', 'ska01',
    #            'ska5', 'ska1', 'cib', 'desi', 'des']
    cl_cross_k = {}
    cl_auto = {}
    rho = {}
    for label in surveys:
        cl_cross_k[label] = np.array(cls['k' + label])
        cl_auto[label] = np.array(cls[label + label])
        if label == 'wise':
            cl_cross_k[label][np.where(lbins < 100)] = 0.
            #  from Simone Our conservative masking leaves f sky = 0.47 and about
            # 50 million galaxies.
            steradians_on_sphere = 4 * np.pi
            fsky = 0.447
            n_gal = 50e6
            gal_per_ster = n_gal / (steradians_on_sphere * fsky)
            cl_auto[label] += 1 / gal_per_ster * np.ones_like(cl_auto[label])

        if label == 'cib':
            cl_auto[label] = np.array([3500. * (1. * l / 3000.)**(-1.25) for l in lbins])
            # cl_auto[label] += 1000*cl_auto[label][100] * (lbins / lbins[100])**-4.6

        rho[label] = cl_cross_k[label] / np.sqrt(ckk[:] * cl_auto[label])
        if label == 'wise':
            rho[label][np.where(lbins < 100)] = 0.
        if label == 'cib':
            rho[label][np.where(lbins < 100)] = 0.

    # single survey save
    for label in surveys:
        np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format(label),
                   np.vstack((lbins, rho[label])).T)

    cgk = np.zeros((len(labels) + 1, np.size(lbins)))
    cgg = np.zeros((len(labels) + 1, len(labels) + 1, np.size(lbins)))

    for i in np.arange(0, len(labels)):
        cgk[i, :] = np.array(cls['k' + labels[i]])

        for j in np.arange(i, len(labels)):
            cgg[i, j, :] = np.array(cls[labels[i] + labels[j]])

            if (labels[i] == 'cib' and labels[j] == 'cib'):
                cgg[i, j, :] = np.array([3500. * (1. * l / 3000.)**(-1.25) for l in lbins])

            cgg[j, i, :] = cgg[i, j, :]

    if cmb == 'Planck':

        noise_phi = np.loadtxt('./quicklens_data/nlkk.dat')
        # noise_cmb = nl(noise, beam, lmax=4000)
        noise_fun = interp1d(noise_phi[:, 0], noise_phi[:, 1], bounds_error=False, fill_value=1e10)
        ckk_noise = np.zeros_like(ckk)
        ckk_noise = noise_fun(lbins)

    if cmb == 'LiteBird':
        noise = 2.0
        beam = 30
        noise_phi = np.loadtxt(
            './quicklens_data/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
        noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
        # noise_cmb = nl(noise, beam, lmax=4000)
        noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
        ckk_noise = np.zeros_like(ckk)
        ckk_noise = noise_fun(lbins)

    if cmb == 'S3':
        noise = 3.0
        beam = 1
        noise_phi = np.loadtxt(
            './quicklens_data/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
        noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
        # noise_cmb = nl(noise, beam, lmax=4000)
        noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
        ckk_noise = np.zeros_like(ckk)
        ckk_noise = noise_fun(lbins)

    elif cmb == 'S4':
        noise = 1.0
        beam = 1
        noise_phi = np.loadtxt(
            './quicklens_data/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
        noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
        # noise_cmb = nl(noise, beam, lmax=4000)
        noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
        ckk_noise = np.zeros_like(ckk)
        ckk_noise = noise_fun(lbins)

    elif cmb == 'now':
        noise = 9.
        beam = 1
        noise_phi = np.loadtxt(
            './quicklens_data/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
        noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
        # noise_cmb = nl(noise, beam, lmax=4000)

        noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
        ckk_noise = np.zeros_like(ckk)
        ckk_noise = noise_fun(lbins)

    # add cmb lensing
    cgk[-1, :] = ckk
    cgg[-1, :, :] = cgk[:, :]
    cgg[:, -1, :] = cgg[-1, :, :]
    # add noise
    cgg[-1, -1, :] = ckk + ckk_noise
    rho_cmb = np.sqrt(ckk / (ckk + ckk_noise))
    rho_comb = np.zeros_like(lbins)
    rho_gals = np.zeros_like(lbins)
    # See eq A9 of Sherwin CIB
    for i, ell in enumerate(lbins):
        if ell < 108 and 'cib' in labels and 'wise' in labels:
            remove_wise_cib_idx = [labels.index('cib'), labels.index('wise')]
            cgki = np.delete(cgk[:, i], remove_wise_cib_idx)
            cggi = np.delete(
                np.delete(cgg[:, :, i], remove_wise_cib_idx, 0), remove_wise_cib_idx, 1)
        else:
            cgki = cgk[:, i]
            cggi = cgg[:, :, i]

        rho_comb[i] = np.sqrt(np.dot(cgki, np.dot(
            np.linalg.inv(cggi), cgki)) / ckk[i])
        rho_gals[i] = np.sqrt(np.dot(
            cgki[:-1], np.dot(np.linalg.inv(cggi[:-1, :-1]), cgki[:-1])) / ckk[i])

    np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format('comb'),
               np.vstack((lbins, rho_comb)).T)
    np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format('gals'),
               np.vstack((lbins, rho_gals)).T)
    np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format('cmb_' + cmb),
               np.vstack((lbins, rho_cmb)).T)

    print('done computing rhos')
    return lbins, rho, rho_comb, rho_gals, rho_cmb


if __name__ == "__main__":
    main()
