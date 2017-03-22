'''
This take the power spectra cross with CMB lensign and auto of different surveys ans spits out their rho as in Shwerwin Smithful
'''

import numpy as np
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import configparser as ConfigParser


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


cosmosis_dir = '/home/manzotti/cosmosis/'
inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
n_slices = 2


Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')

values.read(cosmosis_dir + values_file)


lbins = np.loadtxt(output_dir + '/limber_spectra/ells__delens.txt')


ckk = np.loadtxt(output_dir + '/limber_spectra/cl_kk_delens.txt')

# noise_cl = np.loadtxt(
#     './compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt')
# check the first ell is the same
# assert (ells_cmb[0] == noise_cl[0, 0])
# CMB Specs

noise = 5.0
beam = 3
noise_phi = np.loadtxt('./quicklens/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
# noise_cmb = nl(noise, beam, lmax=4000)

# noise_cl[:, 1] = noise_cl[:, 1] * noise_cl[:, 0] ** 4 / 4.  # because
# the power C_kk is l^4/4 C_phi
noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
ckk_noise = np.zeros_like(ckk)
ckk_noise = noise_fun(lbins)

# initialize
rho_comb = np.zeros((np.size(lbins)))
rho_cib_des = np.zeros((np.size(lbins)))
rho_des = np.zeros((np.size(lbins)))
rho_cib = np.zeros((np.size(lbins)))


surveys = ['k', 'euclid', 'des_weak', 'lsst', 'ska10', 'ska01',
          'ska5', 'ska1', 'cib', 'desi', 'des']
cl_cross_k = {}
cl_auto = {}
rho = {}
for label in surveys:
    cl_cross_k[label] = np.loadtxt(output_dir + '/limber_spectra/cl_k' + label + '_delens.txt')
    cl_auto[label] = np.loadtxt(output_dir +
                                '/limber_spectra/cl_' + label + label + '_delens.txt')
    if label == 'cib':
        cl_auto[label] = np.array([3500. * (1. * l / 3000.)**(-1.25) for l in lbins])

    rho[label] = cl_cross_k[label] / np.sqrt(ckk[:] * cl_auto[label])


# single survey save
for label in surveys:
    np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format(label),
               np.vstack((lbins, rho[label])).T)


# Multiple surveys

# labels = ['lsst', 'cib', 'desi', 'des']
# cmb = 'S4'

# labels = ['cib', 'des']
# cmb = 'now'


cgk = np.zeros((len(labels) + 1, np.size(lbins)))
cgg = np.zeros((len(labels) + 1, len(labels) + 1, np.size(lbins)))

for i in np.arange(0, len(labels)):
    cgk[i, :] = np.loadtxt(output_dir +
                           '/limber_spectra/cl_k' + labels[i] + '_delens.txt')

    for j in np.arange(i, len(labels)):
        cgg[i, j, :] = np.loadtxt(output_dir +
                                  '/limber_spectra/cl_' + labels[i] + labels[j] + '_delens.txt')
        if (labels[i] == 'cib' and labels[j] == 'cib'):
            print('here')
            cgg[i, j, :] = np.array([3500. * (1. * l / 3000.)**(-1.25) for l in lbins])

        cgg[j, i, :] = cgg[i, j, :]

if cmb == 'S3':
    noise = 5.0
    beam = 1
    noise_phi = np.loadtxt('./quicklens/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
    noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
    # noise_cmb = nl(noise, beam, lmax=4000)
    noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
    ckk_noise = np.zeros_like(ckk)
    ckk_noise = noise_fun(lbins)

elif cmb == 'S4':
    noise = 1.0
    beam = 1
    noise_phi = np.loadtxt('./quicklens/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
    noise_phi *= np.arange(0, len(noise_phi))**4. / 4.
    # noise_cmb = nl(noise, beam, lmax=4000)
    noise_fun = interp1d(np.arange(0, len(noise_phi)), noise_phi)
    ckk_noise = np.zeros_like(ckk)
    ckk_noise = noise_fun(lbins)


elif cmb == 'now':
    noise = 9.
    beam = 1
    noise_phi = np.loadtxt('./quicklens/min_var_noise_{}muk_{}beam.txt'.format(noise, beam))
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
    rho_comb[i] = np.sqrt(np.dot(cgk[:, i], np.dot(
        np.linalg.inv(cgg[:, :, i]), cgk[:, i])) / ckk[i])
    rho_gals[i] = np.sqrt(np.dot(
        cgk[:-1, i], np.dot(np.linalg.inv(cgg[:-1, :-1, i]), cgk[:-1, i])) / ckk[i])

np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format('rho_comb'),
           np.vstack((lbins, rho_comb)).T)
np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format('rho_gals'),
           np.vstack((lbins, rho_gals)).T)
np.savetxt(output_dir + '/limber_spectra/rho_{}.txt'.format('rho_cmb' + cmb),
           np.vstack((lbins, rho_cmb)).T)
