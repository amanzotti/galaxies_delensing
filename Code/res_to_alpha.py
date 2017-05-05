# coding: utf-8

'''
Testing different method to compute a 2d B_res.

Use method 3 after the test for best accuracy

'''



from scipy.interpolate import RectBivariateSpline, interp1d
# import lensing
import sys
import configparser as ConfigParser
import numpy as np

noise = '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/my_projects/CIB_sherwin/compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt'


# call external Eichiro code

# def generate_CMB_rho():
#     '''
#     todo: to make it more realistic add beam here. an extenstion to more observable TE and mean variance would be good
#     '''

#     clpp_th = np.loadtxt(
#         '/home/manzotti/cosmosis/modules/limber/galaxies_delens/cmb_cl/pp.txt')

#     ells_cmb = np.loadtxt(
#         '/home/manzotti/cosmosis/modules/limber/galaxies_delens/cmb_cl/ell.txt')

#     # noise is a 2d array first column is ell second is cl
#     noise_cl = np.loadtxt(
#         '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/my_projects/CIB_sherwin/compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt')
#     # check the first ell is the same
#     assert (ells_cmb[0] == noise_cl[0, 0])
#     noise_cl[:, 1] = noise_cl[:, 1] * noise_cl[:, 0] ** 4
#     # cut one of the 2 at the proper lmax if needed
#     lmax = np.min((np.shape(noise_cl)[0], np.shape(clpp_th)[0]))
#     # try to generate them at the same ells
#     return np.sqrt(clpp_th[:lmax] / (clpp_th[:lmax] + noise_cl[:lmax, 1]))

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

lmin = 60

datadir = '/home/manzotti/galaxies_delensing/Data/'

clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
ells_cmb = np.loadtxt(datadir + 'cmb_cl/ell.txt')

clee *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
clpp = clpp * 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))


lbins = np.logspace(1, 3.5, 190)

clbb_th = np.loadtxt(
    '/home/manzotti/galaxies_delensing/Data/cmb_cl/bb.txt')
clbb_th *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

surveys = ['cib', 'des', 'comb_des_cib', 'comb_des_cib_cmb', 'ska10', 'ska5', 'ska1', 'ska01']

rho = {}
cmb= 'S3'
if cmb == 'S3':
    noise = 7.0
    beam = 3



elif cmb == 'S4':
    noise = 1.
    beam = 1


nlb = nl(beam, noise, lmax=ells_cmb[-1])
nlbb_th_fun = interp1d(ells_cmb[:], nlb[2:])

clbb_res = {}
cbb_res_ls = np.loadtxt(
    '/home/manzotti/galaxies_delensing/Data/limber_spectra/cbb_res_ls.txt')
clbb_th_fun = interp1d(ells_cmb[:], clbb_th[:])


for label in surveys:
    clbb_res[label] = np.loadtxt(
        '/home/manzotti/galaxies_delensing/Data/limber_spectra/cbb_res_' + label + '.txt')

alpha = {}
for label in surveys:
    clbb_res_fun = interp1d(cbb_res_ls[:], clbb_res[label][:])
    alpha[label] = np.mean([(clbb_th_fun(i) + nlbb_th_fun(i)) / (clbb_res_fun(i) + nlbb_th_fun(i)) for i in np.arange(50, 200)])
    print(label,alpha[label])
