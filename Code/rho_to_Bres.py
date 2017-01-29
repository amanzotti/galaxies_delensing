# coding: utf-8
from scipy.interpolate import RectBivariateSpline, interp1d
import lensing
import sys
import ConfigParser
import numpy as np
from joblib import Parallel, delayed

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

B_res = Parallel(n_jobs=len(surveys), verbose=50)(delayed(
    compute_res)(i) for i in surveys)


# clbb = lensing.utils.calc_lensed_clbb_first_order(
#     lbins, clee, clpp, lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)

# np.savetxt(datadir + 'limber_spectra/cbb.txt',
#            np.array(clbb.specs['cl'], dtype=float))
# np.savetxt(
#     datadir + 'limber_spectra/cbb_res_ls.txt', clbb.ls)
