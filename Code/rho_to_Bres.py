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


def compute_res_parallel(rho_filename, output_dir, clee_fun, clpp_fun, nle_fun):

    print('start integration')

    if rho_filename == 'test':
        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2)
    else:
        rho_filename = output_dir + 'limber_spectra/' + rho_filename
        print(rho_filename)
        rho = np.loadtxt(rho_filename)[:, 1]
        lbins = np.loadtxt(rho_filename)[:, 0]
        rho_fun = InterpolatedUnivariateSpline(
            lbins, np.nan_to_num(rho), ext='raise')

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell) ** 2)

    clbb_res_ell = [integrate.dblquad(
        integrand, 8, 1500, lambda x: 0, lambda x: 2. * np.pi, args=(L,), epsabs=1.49e-08, epsrel=1.49e-07)[0] for L in np.arange(4, 2500, 10)]

    np.savetxt(rho_filename.split('.txt')[0] + 'Cbb_res.txt', clbb_res_ell)
    np.savetxt(output_dir + 'limber_spectra/cbb_res_ls.txt', np.arange(4, 2500, 10))

    return clbb_res_ell


def compute_res(rho_filename, noise_pol=2., fwhm_beam=30.):

    cosmosis_dir = '/home/manzotti/cosmosis/'
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    Config_ini = ConfigParser.ConfigParser()
    # values = ConfigParser.ConfigParser()
    Config_ini.read(inifile)
    # values_file = Config_ini.get('pipeline', 'values')
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
    nle = nl(noise_pol, fwhm_beam, lmax=ells_cmb[-1])[2:]

    clee_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clee[:5000], ext=2)
    clpp_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clpp[:5000], ext='zeros')
    nle_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], nle[:5000], ext=2)

    print('start integration')

    if rho_filename == 'test':
        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2)
    else:
        rho = np.loadtxt(rho_filename)[:, 1]
        lbins = np.loadtxt(rho_filename)[:, 0]
        rho_fun = InterpolatedUnivariateSpline(lbins, np.nan_to_num(rho), ext='raise')

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell) ** 2)

    clbb_res_ell = [integrate.dblquad(
        integrand, 4, 3000, lambda x: 0, lambda x: 2. * np.pi, args=(L,), epsabs=1.49e-08, epsrel=1.49e-07)[0] for L in np.arange(4, 1500, 10)]

    np.savetxt(rho_filename.split('.txt')[0] + 'Cbb_res.txt', clbb_res_ell)
    np.savetxt(datadir + 'limber_spectra/cbb_res_ls.txt', np.arange(4, 1500, 10))

    return clbb_res_ell


def load_res(labels):
    datadir = '/home/manzotti/cosmosis/modules/limber/galaxies_delens/'

    res_list = []
    for label in labels:
        res_list.append(np.loadtxt(datadir + 'limber_spectra/cbb_res_' + label + 'test3.txt'))
    return res_list


def main(rho_names, nle):
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

    # surveys = ['test', 'cib', 'des', 'comb_des_cib', 'comb_des_cib_cmb',
    #            'ska10', 'ska5', 'ska1', 'ska01', 'lsst', 'euclid', 'rho_comb', 'rho_cmbS4', 'rho_cmbS3']

    # surveys = ['rho_cmbS3', 'rho_cmbS4']
    # generating noise in E-modes
    # nle = nl(1, 1, lmax=ells_cmb[-1])[2:]

    # B_res = Parallel(n_jobs=len(surveys), verbose=50)(delayed(
    #     compute_res)(i) for i in surveys)

    clee_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clee[:5000], ext=2)
    clpp_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clpp[:5000], ext='zeros')
    nle_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], nle[:5000], ext=2)

    return Parallel(n_jobs=len(rho_names), verbose=0)(delayed(compute_res_parallel)(i,output_dir, clee_fun, clpp_fun, nle_fun) for i in rho_names)




if __name__ == "__main__":
    main(rho_names, nle)
