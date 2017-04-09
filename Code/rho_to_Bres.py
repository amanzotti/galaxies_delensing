# coding: utf-8
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import sys
import configparser as ConfigParser
import numpy as np
from joblib import Parallel, delayed
import scipy.integrate as integrate


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


def compute_res_parallel(rho_filename):

    print('start integration')

    if rho_filename == 'test':
        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2)
    else:
        rho = np.loadtxt(rho_filename)
        lbins = np.loadtxt('lbins.txt')
        rho_fun = InterpolatedUnivariateSpline(
            lbins, np.nan_to_num(rho), ext='raise')

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell) ** 2)

    clbb_res_ell = [integrate.dblquad(
        integrand, 4, 1500, lambda x: 0, lambda x: 2. * np.pi, args=(L,), epsabs=1.49e-08, epsrel=1.49e-07)[0] for L in np.arange(10, 700, 10)]

    np.savetxt(rho_filename.split('.txt')[0] + 'Cbb_res.txt', clbb_res_ell)
    np.savetxt(datadir + 'limber_spectra/cbb_res_ls.txt', np.arange(10, 700, 10))

    return clbb_res_ell


def load_res(labels):
    datadir = '/home/manzotti/cosmosis/modules/limber/galaxies_delens/'

    res_list = []
    for label in labels:
        res_list.append(np.loadtxt(datadir + 'limber_spectra/cbb_res_' + label + 'test3.txt'))
    return res_list


cosmosis_dir = '/home/manzotti/cosmosis/'
inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')

datadir = output_dir

clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
ells_cmb = np.loadtxt(datadir + 'cmb_cl/ell.txt')

clee *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
clpp = clpp * 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))


lbins = np.logspace(1, 3.5, 190)


clbb_th = np.loadtxt(
    output_dir + 'cmb_cl/bb.txt')
clbb_th *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

surveys = ['test', 'cib', 'des', 'comb_des_cib', 'comb_des_cib_cmb',
           'ska10', 'ska5', 'ska1', 'ska01', 'lsst', 'euclid', 'rho_comb', 'rho_cmbS4', 'rho_cmbS3']

surveys = ['rho_cmbS3', 'rho_cmbS4']
# generating noise in E-modes
nle = nl(1, 1, lmax=ells_cmb[-1])[2:]

# B_res = Parallel(n_jobs=len(surveys), verbose=50)(delayed(
#     compute_res)(i) for i in surveys)

clee_fun = InterpolatedUnivariateSpline(
    ells_cmb[:5000], clee[:5000], ext=2)
clpp_fun = InterpolatedUnivariateSpline(
    ells_cmb[:5000], clpp[:5000], ext='zeros')
nle_fun = InterpolatedUnivariateSpline(
    ells_cmb[:5000], nle[:5000], ext=2)

rho_names = ['rho_cib.txt', 'rho_des.txt', 'rho_cmb_current.txt', 'rho_gals_current.txt', 'rho_comb_current.txt', 'rho_cib.txt',
             'rho_cmb_S3.txt', 'rho_gals_S3.txt', 'rho_comb_S3.txt', 'rho_cmb_S4.txt', 'rho_gals_S4.txt', 'rho_comb_S4.txt']

# for label in surveys:
#     B_res2 = compute_res_2(label, clee_fun, clpp_fun, nle_fun)

# for label in surveys:
#     compute_res_3(label, clee_fun, clpp_fun, nle_fun)
# compute_res_parallel('rho_cmbS4')

B_res3 = Parallel(n_jobs=6, verbose=500)(delayed(
    compute_res_parallel)(i) for i in rho_names)


# def compute_res(label_survey):
#     lbins = np.logspace(1, 3.5, 190)

#     if label_survey == 'test':
#         clbb_res = lensing.utils.calc_lensed_clbb_first_order(
#             lbins, clee, clpp, lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
#     else:
#         rho = np.loadtxt(
#             '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
#         rho_fun = interp1d(rho[:, 0], rho[:, 1], bounds_error=False, fill_value=0.)
#         clbb_res = lensing.utils.calc_lensed_clbb_first_order(
#             lbins, clee, clpp * (1. - (clee / (clee + nle)) * rho_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
#     np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + '.txt',
#                np.array(clbb_res.specs['cl'], dtype=float))
#     return np.array(clbb_res.specs['cl'], dtype=float)


# def compute_res_2(label_survey, clee, clpp, nle):

#     if label_survey == 'test':
#         def integrand(ell):
#             return ell**5 / 4. / np.pi * clpp(ell) * clee(ell)
#     else:
#         rho = np.loadtxt(
#             '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
#         rho_fun = interp1d(rho[:, 0], rho[:, 1], bounds_error=False, fill_value=0.)

#         def integrand(ell):
# return ell**5 / 4. / np.pi * clpp(ell) * clee(ell) * (1. - (clee(ell) /
# (clee(ell) + nle(ell))) * rho_fun(ell) ** 2)

#     clbb_res = integrate.quad(integrand, 4, 2500, limit=100, epsrel=1.49e-09)
#     np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey +
#                'test2.txt', clbb_res[0] * np.ones(2000))
#     return clbb_res[0] * np.ones(2000)

# def compute_res_3(label_survey, clee_fun, clpp_fun, nle_fun):
#     rho = np.loadtxt(
#         '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
#     rho_fun = interp1d(rho[:1000, 0], rho[:1000, 1], bounds_error=False, fill_value=0.)
#     print('start integration')

#     @jit
#     def integrand(theta, ell, L):
#         #     print(ell,theta)
#         clee = clee_fun(ell)
# return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 *
# clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee *
# (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) *
# rho_fun(ell) ** 2)

#     clbb_res_ell = [integrate.dblquad(
#         integrand, 4, 3000, lambda x: 0, lambda x: 2. * np.pi, args=(L,))[0] for L in np.arange(4, 500, 10)]
#     np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + 'test3.txt', clbb_res_ell)
#     return clbb_res_ell
