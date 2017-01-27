# coding: utf-8
from scipy.interpolate import RectBivariateSpline, interp1d
import lensing
import sys
import ConfigParser
import numpy as np

noise = '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/my_projects/CIB_sherwin/compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt'


# call external Eichiro code

def generate_CMB_rho():
    '''
    todo: to make it more realistic add beam here. an extenstion to more observable TE and mean variance would be good
    '''

    clpp_th = np.loadtxt(
        '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/cmb_cl/pp.txt')

    ells_cmb = np.loadtxt(
        '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/cmb_cl/ell.txt')

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


lmin = 60

datadir = '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/'

clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
ells_cmb = np.loadtxt(datadir + 'cmb_cl/ell.txt')

clee *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
clpp = clpp / (ells_cmb.astype(float)) ** 4


lbins = np.logspace(1, 3.2, 90)

clbb = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp, lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_th = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/cmb_cl/bb.txt')
clbb_th *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))


rho_cib = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/limber_spectra/rho_cib.txt')
rho_des = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/limber_spectra/rho_des.txt')
rho_des_cib = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/limber_spectra/rho_comb_des_cib.txt')

rho_des_cib_cmb = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/cib_des_delens/limber_spectra/rho_comb_des_cib_cmb.txt')

for i, l in enumerate(rho_cib[:, 0]):
    if l < lmin:
        rho_cib[i, 1] = 0.0

rho_cib_fun = interp1d(rho_cib[:, 0], rho_cib[:, 1])
rho_des_fun = interp1d(rho_des[:, 0], rho_des[:, 1])
rho_des_cib_fun = interp1d(rho_des_cib[:, 0], rho_des_cib[:, 1])
rho_des_cib_cmb_fun = interp1d(rho_des_cib_cmb[:, 0], rho_des_cib_cmb[:, 1])

# print ells_cmb,ells_desi,ells_des

clbb_cib = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_cib_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_des = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_des_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_des_cib = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_des_cib_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_des_cib_cmb = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_des_cib_cmb_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)

np.savetxt(datadir + 'limber_spectra/cbb_res_cib.txt',
           np.array(clbb_cib.specs['cl'], dtype=float))

np.savetxt(datadir + 'limber_spectra/cbb_res_des.txt',
           np.array(clbb_des.specs['cl'], dtype=float))

np.savetxt(datadir + 'limber_spectra/cbb_res_des_cib.txt',
           np.array(clbb_des_cib.specs['cl'], dtype=float))

np.savetxt(datadir + 'limber_spectra/cbb_res_des_cib_cmb.txt',
           np.array(clbb_des_cib_cmb.specs['cl'], dtype=float))

np.savetxt(datadir + 'limber_spectra/cbb.txt',
           np.array(clbb.specs['cl'], dtype=float))

np.savetxt(
    datadir + 'limber_spectra/cbb_res_ls.txt', clbb.ls)
