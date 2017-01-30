# coding: utf-8
from scipy.interpolate import RectBivariateSpline, interp1d
import lensing
import sys
import ConfigParser
import numpy as np
from joblib import Parallel, delayed
import scipy.integrate as integrate
noise = '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/my_projects/CIB_sherwin/compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt'


# call external Eichiro code

def generate_CMB_rho():
    '''
    todo: to make it more realistic add beam here. an extenstion to more observable TE and mean variance would be good
    '''

    clpp_th = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/galaxies_delens/cmb_cl/pp.txt')

    ells_cmb = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/galaxies_delens/cmb_cl/ell.txt')

    # noise is a 2d array first column is ell second is cl
    noise_cl = np.loadtxt(
        '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/my_projects/CIB_sherwin/compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt')
    # check the first ell is the same
    assert (ells_cmb[0] == noise_cl[0, 0])
    noise_cl[:, 1] = noise_cl[:, 1] * noise_cl[:, 0] ** 4
    # cut one of the 2 at the proper lmax if needed
    lmax = np.min((np.shape(noise_cl)[0], np.shape(clpp_th)[0]))
    # try to generate them at the same ells
    return np.sqrt(clpp_th[:lmax] / (clpp_th[:lmax] + noise_cl[:lmax, 1]))


def compute_res(label_survey):
    rho = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
    rho_fun = interp1d(rho[:, 0], rho[:, 1])
    clbb_res = lensing.utils.calc_lensed_clbb_first_order(
        lbins, clee, clpp * (1. - (clee / (clee + nle)) * rho_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
    np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + '.txt',
               np.array(clbb_res.specs['cl'], dtype=float))


def compute_res_2(label_survey, clee, clpp, nle):
    rho = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
    rho_fun = interp1d(rho[:, 0], rho[:, 1], bounds_error=False, fill_value=0.)

    def integrand(ell):
        # print(ell**5 / 4. / np.pi * clpp(ell) * clee(ell))
        return ell**5 / 4. / np.pi * clpp(ell) * clee(ell) * (1. - (clee(ell) / (clee(ell) + nle(ell))) * rho_fun(ell) ** 2)

    clbb_res = integrate.quad(integrand, 4, 2500, limit=100, epsrel=1.49e-09)
    np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + 'test2.txt', clbb_res[0] * np.ones_like(rho[:, 0]))


def compute_res_3(label_survey, clee_fun, clpp_fun, nle_fun):
    rho = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
    rho_fun = interp1d(rho[:, 0], rho[:, 1], bounds_error=False, fill_value=0.)
    print('start integration')
    def integrand(theta, ell, L):
        #     print(ell,theta)
        clee = clee_fun(ell)
        return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell) ** 2)

    clbb_res_ell = [integrate.dblquad(integrand, 4, 3000, lambda x: 0, lambda x: 2. * np.pi, args=(L,))[0] for L in np.arange(4, 500, 10)]
    np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + 'test3.txt', clbb_res_ell)


def compute_res_3_parallel(label_survey):
    rho = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
    rho_fun = interp1d(rho[:, 0], rho[:, 1], bounds_error=False, fill_value=0.)
    print('start integration')

    def integrand(theta, ell, L):
        #     print(ell,theta)
        clee = clee_fun(ell)
        return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell) ** 2)

    clbb_res_ell = [integrate.dblquad(integrand, 4, 3000, lambda x: 0, lambda x: 2. * np.pi, args=(L,))[0] for L in np.arange(4, 500, 10)]
    np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + 'test3.txt', clbb_res_ell)


lmin = 60

datadir = '/home/manzotti/cosmosis/modules/limber/galaxies_delens/'

clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
ells_cmb = np.loadtxt(datadir + 'cmb_cl/ell.txt')

clee *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
clpp = clpp * 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))


lbins = np.logspace(1, 3.5, 190)


clbb_th = np.loadtxt(
    '/home/manzotti/cosmosis/modules/limber/galaxies_delens/cmb_cl/bb.txt')
clbb_th *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

surveys = ['cib', 'des', 'comb_des_cib', 'comb_des_cib_cmb', 'ska10', 'ska5', 'ska1', 'ska01', 'lsst', 'euclid']

nle = lensing.utils.nl(1, 1, lmax=ells_cmb[-1])[2:]

# B_res = Parallel(n_jobs=len(surveys), verbose=50)(delayed(
#     compute_res)(i) for i in surveys)

clee_fun = interp1d(ells_cmb, clee, bounds_error=False, fill_value=0.)
clpp_fun = interp1d(ells_cmb, clpp, bounds_error=False, fill_value=0.)
nle_fun = interp1d(ells_cmb, nle, bounds_error=False, fill_value=10e8)

for label in surveys:
    compute_res_2(label, clee_fun, clpp_fun, nle_fun)

# for label in surveys:
#     compute_res_3(label, clee_fun, clpp_fun, nle_fun)

B_res = Parallel(n_jobs=len(surveys), verbose=50)(delayed(
    compute_res_3_parallel)(i) for i in surveys)

# clbb = lensing.utils.calc_lensed_clbb_first_order(
#     lbins, clee, clpp, lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)

# np.savetxt(datadir + 'limber_spectra/cbb.txt',
#            np.array(clbb.specs['cl'], dtype=float))
# np.savetxt(
#     datadir + 'limber_spectra/cbb_res_ls.txt', clbb.ls)
