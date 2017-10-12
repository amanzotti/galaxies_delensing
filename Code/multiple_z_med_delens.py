'''
This take the power spectra cross with CMB lensign and auto of different surveys ans spits out their rho as in Shwerwin Smithful
'''

import numpy as np
# import scipy.integrate
from scipy.interpolate import interp1d
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


def compute_rho_sigle_tracer(surveys, lbins, cls, ckk, output_dir, galaxies_steradians=1e15):
    cl_cross_k = {}
    cl_auto = {}
    rho = {}

    for label in surveys:
        cl_cross_k[label] = np.array(cls['k' + label])
        cl_auto[label] = np.array(cls[label + label]) + \
            (galaxies_steradians / (0.000290888)**2)**(-1)
        rho[label] = cl_cross_k[label] / np.sqrt(ckk[:] * cl_auto[label])
        np.savetxt(output_dir + '/limber_spectra/rho_test_' + label + '_ngal_{:.1f}.txt'.format(galaxies_steradians),
                   np.vstack((lbins, rho[label])).T)
    return rho


def compute_rho_single_tracer_plus_CMB(surveys, lbins, cls, ckk, ckk_noise, output_dir, galaxies_steradians=1e15):
    rho_comb_dict = {}
    # return lbins, rho
    for label in surveys:
        rho_comb_array = np.zeros_like(lbins)
        # print(label)
        cgk = np.zeros((2, np.size(lbins)))
        cgg = np.zeros((2, 2, np.size(lbins)))

        cgk[0, :] = np.array(cls['k' + label])
        cgg[0, 0, :] = np.array(cls[label + label]) + \
            (galaxies_steradians / (0.000290888)**2)**(-1)
        # add cmb lensing
        cgk[-1, :] = ckk
        cgg[-1, :, :] = cgk[:, :]
        cgg[:, -1, :] = cgg[-1, :, :]
        # add noise
        cgg[-1, -1, :] = ckk + ckk_noise

        rho_cmb = np.sqrt(ckk / (ckk + ckk_noise))

        # See eq A9 of Sherwin CIB
        for i, ell in enumerate(lbins):
            cgki = cgk[:, i]
            cggi = cgg[:, :, i]
            rho_comb_array[i] = np.sqrt(np.dot(cgki, np.dot(
                np.linalg.inv(cggi), cgki)) / ckk[i])

        rho_comb_dict[label] = rho_comb_array
        np.savetxt(output_dir + '/limber_spectra/rho_test_comb_' + label + '_ngal_{:.1f}.txt'.format(galaxies_steradians),
                   np.vstack((lbins, np.array(rho_comb_dict[label]))).T)

    return rho_comb_dict, rho_cmb


def set_CMB_lensing_noise(lbins, cmb, ckk):

    if cmb == 'Planck':
        noise_phi = np.loadtxt('./quicklens_data/nlkk.dat')
        # noise_cmb = nl(noise, beam, lmax=4000)
        noise_fun = interp1d(
            noise_phi[:, 0], noise_phi[:, 1], bounds_error=False, fill_value=1e10)
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
        noise = 2.0
        beam = 2
        noise_phi = np.loadtxt(
            './quicklens_data/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
        noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
        # noise_cmb = nl(noise, beam, lmax=4000)
        noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
        ckk_noise = np.zeros_like(ckk)
        ckk_noise = noise_fun(lbins)

    elif cmb == 'S4':
        noise = 0.3
        beam = 2
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
    return ckk_noise


def build_average_matrix():
    pass


def main(cmb, z_means, gal_nums):

    output_dir = '/home/manzotti/galaxies_delensing/Data/'
    cls = pickle.load(open('../Data/limber_spectra_delens_test_zm.pkl', 'rb'))
    lbins = np.load('../Data/ells.npy')

    ckk = cls['kk']
    # initialize
    labels = []
    for z in z_means:
        # labels.append('z_median_{:.3}'.format(z_mean))
        labels.append('z_median_{:.3}'.format(z))

    surveys = labels

    rho = compute_rho_sigle_tracer(
        surveys, lbins, cls, ckk, output_dir, galaxies_steradians=1.)

    ckk_noise = set_CMB_lensing_noise(lbins, cmb, ckk)

    rho_comb_dict, rho_cmb = compute_rho_single_tracer_plus_CMB(
        surveys, lbins, cls, ckk, ckk_noise, output_dir, galaxies_steradians=1.)

    for gal_num in gal_nums:

        rho = compute_rho_sigle_tracer(
            surveys, lbins, cls, ckk, output_dir, galaxies_steradians=gal_num)

        ckk_noise = set_CMB_lensing_noise(lbins, cmb, ckk)

        rho_comb_dict, rho_cmb = compute_rho_single_tracer_plus_CMB(
            surveys, lbins, cls, ckk, ckk_noise, output_dir, galaxies_steradians=gal_num)

    print('done computing rhos')
    return lbins, rho, rho_comb_dict, rho_cmb


if __name__ == "__main__":
    main('S4')
