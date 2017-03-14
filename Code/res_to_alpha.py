# coding: utf-8

'''
Testing different method to compute a 2d B_res.

Use method 3 after the test for best accuracy

'''



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

surveys = ['cib', 'des', 'comb_des_cib', 'comb_des_cib_cmb', 'ska10', 'ska5', 'ska1', 'ska01']

rho = {}
nlb = lensing.utils.nl(1, 1, lmax=ells_cmb[-1])
nlbb_th_fun = interp1d(ells_cmb[:], nlb[2:])

clbb_res = {}
cbb_res_ls = np.loadtxt(
    '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/cbb_res_ls.txt')
clbb_th_fun = interp1d(ells_cmb[:], clbb_th[:])


for label in surveys:
    clbb_res[label] = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/cbb_res_' + label + 'test_parallel.txt')

alpha = {}
for label in surveys:
    clbb_res_fun = interp1d(cbb_res_ls[:], clbb_res[label][:])
    alpha[label] = np.mean([(clbb_th_fun(i) + nlbb_th_fun(i)) / (clbb_res_fun(i) + nlbb_th_fun(i)) for i in np.arange(50, 200)])
    print(alpha[label])
