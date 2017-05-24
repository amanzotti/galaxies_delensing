
# # Fisher Analysis

# !module load gcc/6.1

import numpy as np
import matplotlib.pyplot as plt
try:
    import functools32
except ImportError:
    import functools as functools32

import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
import multiple_survey_delens
import configparser as ConfigParser
import rho_to_Bres
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline

from colorama import init
from colorama import Fore, Back, Style

init(autoreset=True)
plt.rcParams["figure.figsize"] = [16, 10]

np.seterr(divide='ignore', invalid='ignore')
# ============================================
# SET LATEX
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# ============================================


font_size = 24.
plt.rcParams['font.size'] = font_size

plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.linewidth'] = font_size / 22.
plt.rcParams['axes.titlesize'] = font_size * 1.3
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size / 1.2
plt.rcParams['ytick.labelsize'] = font_size / 1.2
plt.rcParams['axes.color_cycle'] = '#e41a1c,#377eb8,#4daf4a,#984ea3,#ff7f00,#ffff33,#a65628'


# ============================================

# Simplify paths by removing "invisible" points, useful for reducing
# file size when plotting a large number of points
plt.rcParams['path.simplify'] = False
# ============================================

# ============================================

# Have the legend only plot one point instead of two, turn off the
# frame, and reduce the space between the point and the label

plt.rcParams['axes.linewidth'] = 1.0


# ============================================
# LEGEND
# ============================================
# OPTION
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.handletextpad'] = 0.3
# ============================================


# COSMOSIS VALUES!!!!!!

pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars.set_cosmology(H0=70, ombh2=0.0226, omch2=0.112,
                   mnu=0.029, omk=0, tau=0.079)
pars.InitPower.set_params(ns=0.96, r=0., nt=0)
pars.set_for_lmax(5000, lens_potential_accuracy=3)
# pars.set_for_lmax?

pars.AccurateBB = True
pars.OutputNormalization = False
pars.WantTensors = True
pars.DoLensing = True
pars.max_l_tensor = 3000
pars.max_eta_k_tensor = 3000.

# print(pars) # if you want to test parasm
results = camb.get_results(pars)
tens_cl = results.get_tensor_cls(3000)


# ### Remember there is a 7.4e12 missing and CAMB always give you $ \ell (\ell +1)/2\pi  $

@functools32.lru_cache(maxsize=64)
def clbb(r=0.1, nt=None, lmax=3000):
    inflation_params = initialpower.InitialPowerParams()
    if nt is None:
        nt = -r / 8.
    inflation_params.set_params(As=2.1e-9, r=r, nt=nt)
    results.power_spectra_from_transfer(inflation_params)
    return results.get_total_cls(lmax)[:, 2] * 7.42835025e12


@functools32.lru_cache(maxsize=64)
def clbb_tens(r=0.1, nt=None, lmax=3000):
    inflation_params = initialpower.InitialPowerParams()
    if nt is None:
        nt = -r / 8.
    inflation_params.set_params(As=2.1e-9, r=r, nt=nt)
    results.power_spectra_from_transfer(inflation_params)
    return results.get_tensor_cls(lmax)[:, 2] * 7.42835025e12


clbb_tens.cache_clear()
clbb.cache_clear()


def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      - beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             - maximum multipole.
    """
    import numpy as np
    ls = np.arange(0, lmax + 1)
    return np.exp(-(fwhm_arcmin * np.pi / 180. / 60.)**2 /
                  (16. * np.log(2.)) * ls * (ls + 1.))


def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
          * noise_uK_arcmin - map noise level in uK.arcmin
          * fwhm_arcmin     - beam full-width-at-half-maximum (fwhm) in arcmin.
          * lmax            - maximum multipole.
    """
    import numpy as np

    return (noise_uK_arcmin * np.pi / 180. / 60.)**2 / bl(fwhm_arcmin, lmax)**2


def fisher_r_nt(r_fid=0.2, fid=None,
                lmin=10,
                lmax=2000,
                noise_uK_arcmin=4.5,
                fwhm_arcmin=4.,
                clbb_cov=None,
                fsky=0.5
                ):

    nlb = nl(noise_uK_arcmin, fwhm_arcmin, lmax=lmax)
    ell_nlb = np.arange(0, len(nlb))
    nlb = nlb * ell_nlb * (ell_nlb + 1.) / 2. / np.pi

    if fid is None:
        #         print('n_t fis in None set consistency relation')
        fid = -r_fid / 8.

    if clbb_cov is None:
        clbb_cov = clbb(r_fid, fid, lmax=lmax)

    Cov = np.sqrt(2. / (fsky * (2. * np.arange(0, len(nlb)) + 1.))
                  ) * (clbb_cov + nlb)

#     print(r_fid, fid,Cov)

    dx = r_fid * 0.02 + 0.01
    dBl_dr = (-clbb(r_fid + 2. * dx, fid, lmax=lmax) + 8. * clbb(r_fid + dx, fid, lmax=lmax) -
              8. * clbb(r_fid - dx, fid, lmax=lmax) + clbb(r_fid - 2. * dx, fid, lmax=lmax)) / (12. * dx)
#     print(dBl_dr)

    dx = fid * 0.05 + 0.04
    nt_deriv = (-clbb(r_fid, fid + 2 * dx, lmax=lmax) + 8. * clbb(r_fid, fid + dx, lmax=lmax) -
                8. * clbb(r_fid, fid - dx, lmax=lmax) + clbb(r_fid, fid - 2 * dx, lmax=lmax)) / (12. * dx)
#     print(nt_deriv)
#     print(dBl_dr, nt_deriv)

    Frr = np.sum(np.nan_to_num(dBl_dr**2 / Cov**2)[lmin:lmax])
    Fnn = np.sum(np.nan_to_num(nt_deriv**2 / Cov**2)[lmin:lmax])
    Fnr = np.sum(np.nan_to_num(nt_deriv * dBl_dr / Cov**2)[lmin:lmax])

    F_matrix = np.zeros((2, 2))
    F_matrix[0, 0] = Frr
    F_matrix[1, 0] = Fnr
    F_matrix[0, 1] = Fnr
    F_matrix[1, 1] = Fnn

#     print(F_matrix)
    sigma_r = np.sqrt(np.linalg.inv(F_matrix)[0, 0])
    sigma_nt = np.sqrt(np.linalg.inv(F_matrix)[1, 1])
#     print(sigma_r,sigma_nt)
    return (sigma_r, sigma_nt, np.sqrt(1 / Frr), np.sqrt(1 / Fnn))


cosmosis_dir = '/home/manzotti/cosmosis/'
inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')
print('')
print(Fore.RED + 'PLANCK + DES + CIB + WISE')
print('')


# =====================================
# =====================================


labels = ['wise', 'cib', 'des']
cmb = 'Planck'
multiple_survey_delens.main(labels, cmb)
ells_cmb = np.loadtxt(output_dir + 'cmb_cl/ell.txt')
rho_names = ['rho_cib.txt', 'rho_des.txt', 'rho_gals.txt',
             'rho_wise.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
# BICEP level 3 muK and 30 arcmin beam
nle = nl(3., 30., lmax=ells_cmb[-1])[2:]
B_test = rho_to_Bres.main(['test'], nle)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_lensed = InterpolatedUnivariateSpline(
    lbins, lbins * (lbins + 1.) * np.nan_to_num(B_test) / 2. / np.pi, ext='extrapolate')
B_res3 = rho_to_Bres.main(rho_names, nle)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')

clbb_res = {}
for i, probe in enumerate(rho_names):
    if probe == 'test':
        print(i, probe)
        clbb_res[probe] = InterpolatedUnivariateSpline(
            lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')
    else:
        print(i, probe.split('.txt')[0].split('rho_')[1])
        clbb_res[probe.split('.txt')[0].split('rho_')[1]] = InterpolatedUnivariateSpline(
            lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[1:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))),
          'ell<500=', 1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) /
                                   clbb_lensed(np.arange(4, 500, 75))),
          'ell<1000=', 1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(np.arange(4, 1000, 100))
                                    ), 'ell<1500=', 1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(np.arange(4, 1500, 100)))
          )
    print('')

print('')
print('')

# sys.exit()
lmax = 500
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 3.
fwhm_arcmin = 30.
r_fid = 0.
fsky = 0.06

print('')
print(Fore.YELLOW + 'r=0')
print('')
for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
        r_fid=r_fid,
        lmin=50,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))
                                 ) + clbb_tens(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
        r_fid=r_fid,
        lmin=50,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    print('After delensing % errors', sigma_r_1 * 1e2)
    print('gain', (probe, 'gain = ', sigma_r_1 / sigma_r,
                   sigma_nt_1 / sigma_nt, sigr_1 / sigr, sigmant_1 / sigmant))

print('')
print('')

# ### r=0.07
print(Fore.YELLOW + 'r=0.12')
print('')
print('')

# In[271]:

clbb.cache_clear()
clbb_tens.cache_clear()

lmax = 500
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 3.
fwhm_arcmin = 30.
r_fid = 0.07
fsky = 0.06


for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        fsky=fsky,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(r_fid, lmax=lmax)))
                                 ) + clbb_tens(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)

    print('After delensing % errors', sigma_r_1, sigma_nt)
    print(probe, 'gain = ', sigma_r_1 / sigma_r)


print(Fore.RED + 'Actual scenario High res SPT-pol')
# In[274]:

labels = ['wise', 'cib', 'des']
cmb = 'now'
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_cib.txt', 'rho_des.txt', 'rho_gals.txt',
             'rho_wise.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
nle = nl(9, 1, lmax=ells_cmb[-1])[2:]
B_res3 = rho_to_Bres.main(rho_names, nle)


# In[275]:

lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[1]] = InterpolatedUnivariateSpline(
        lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')


print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[1:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))),
          'ell<500=', 1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) /
                                   clbb_lensed(np.arange(4, 500, 75))),
          'ell<1000=', 1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(np.arange(4, 1000, 100))
                                    ), 'ell<1500=', 1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(np.arange(4, 1500, 100)))
          )
    print('')

print('')
print('')


# In[276]:

lmax = 500
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 3.
fwhm_arcmin = 30.
r_fid = 0.
f_sky = 0.06
print('')
print('')

# ### r=0.07

print('')
print(Fore.YELLOW + 'r=0')
print('')
print('')
print('')


for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
        r_fid=r_fid,
        lmin=50,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))
                                 ) + clbb_tens(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
        r_fid=r_fid,
        lmin=50,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    print('After delensing % errors', sigma_r_1 * 1e2)
    print('gain', (probe, 'gain = ', sigma_r_1 / sigma_r,
                   sigma_nt_1 / sigma_nt, sigr_1 / sigr, sigmant_1 / sigmant))


# ### r=0.12

# In[277]:

clbb.cache_clear()
clbb_tens.cache_clear()

lmax = 500
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 3.
fwhm_arcmin = 30.
r_fid = 0.12
fsky = 0.06
print('')
print('r=0.12')
print('')

for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        fsky=fsky,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(r_fid, lmax=lmax)))
                                 ) + clbb_tens(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)

    print('After delensing % errors', sigma_r_1, sigma_nt)
    print(probe, 'gain = ', sigma_r_1 / sigma_r)


# ## CMB S3

# In[278]:

# possible names orders matters = ['k', 'euclid', 'des_weak', 'lsst',
# 'ska10',            'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
labels = ['wise', 'cib', 'desi', 'des']
cmb = 'S3'
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_cib.txt', 'rho_des.txt', 'rho_desi.txt',
             'rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
nle = nl(3, 1, lmax=ells_cmb[-1])[2:]
B_res3 = rho_to_Bres.main(rho_names, nle)


# In[279]:

lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[1]] = InterpolatedUnivariateSpline(
        lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')


print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[1:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))),
          'ell<500=', 1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) /
                                   clbb_lensed(np.arange(4, 500, 75))),
          'ell<1000=', 1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(np.arange(4, 1000, 100))
                                    ), 'ell<1500=', 1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(np.arange(4, 1500, 100)))
          )
    print('')

print('')
print('')


# ### r=0

# In[288]:

# noise_uK_arcmin=4.5,
# fwhm_arcmin=4.,
lmax = 2000
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 3
fwhm_arcmin = 1.
r_fid = 0.0001
fsky = 0.06

print('')
print(Fore.YELLOW + 'r=0')
print('')
for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        fsky=fsky,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(r_fid, lmax=lmax)))
                                 ) + clbb_tens(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)

    print('After delensing % errors', sigma_r, sigma_nt)
    print(probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt)


# ### r=0.12

# In[307]:

# noise_uK_arcmin=4.5,
# fwhm_arcmin=4.,
lmax = 2000
noise_uK_arcmin = 3.
fwhm_arcmin = 1.
r_fid = 0.12
fsky = 0.06
print('')
print('')

# ### r=0.07
print(Fore.YELLOW + 'r=0.12')
print('')
print('')

for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        fsky=fsky,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(r_fid, lmax=lmax)))
                                 ) + clbb_tens(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb(r_fid, lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)

    print('After delensing % errors', sigma_r_1, sigma_nt, sigma_nt_1)
    print(probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt)


# ## CMB S4

# In[310]:

#  'wise', 'euclid', 'des_weak', 'lsst', 'ska10',
# #              'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
labels = ['wise', 'euclid', 'des_weak', 'lsst', 'ska10',
          'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
cmb = 'S4'
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_euclid.txt', 'rho_lsst.txt', 'rho_ska10.txt', 'rho_ska5.txt',
             'rho_ska1.txt', 'rho_ska01.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
nle = nl(1, 1, lmax=ells_cmb[-1])[2:]
B_res3 = rho_to_Bres.main(rho_names, nle)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[1]] = InterpolatedUnivariateSpline(
        lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')


print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[1:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))),
          'ell<500=', 1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) /
                                   clbb_lensed(np.arange(4, 500, 75))),
          'ell<1000=', 1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(np.arange(4, 1000, 100))
                                    ), 'ell<1500=', 1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(np.arange(4, 1500, 100)))
          )
    print('')

print('')
print('')
# ### r=0

# In[312]:

# noise_uK_arcmin=4.5,
# fwhm_arcmin=4.,
lmax = 3000
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 1
fwhm_arcmin = 1.
r_fid = 0.

print('')
print(Fore.YELLOW + 'r=0')
print('')
for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb(0., lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    print((probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt))


# ### r=0.07

# In[313]:

# noise_uK_arcmin=4.5,
# fwhm_arcmin=4.,
lmax = 3000
noise_uK_arcmin = 1
fwhm_arcmin = 1.
r_fid = 0.12

print('')
print('')

# ### r=0.07
print(Fore.YELLOW + 'r=0.12')
print('')
print('')


for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb(0., lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    print((probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt))


# ## SKA ONLY

# In[ ]:

#  'wise', 'euclid', 'des_weak', 'lsst', 'ska10',
# #              'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
labels = ['ska10', 'ska01', 'ska5', 'ska1']
cmb = 'S4'
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_ska10.txt', 'rho_ska5.txt', 'rho_ska1.txt', 'rho_ska01.txt',
             'rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
nle = nl(1, 1, lmax=ells_cmb[-1])[2:]
B_res3 = rho_to_Bres.main(rho_names, nle)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[1]] = InterpolatedUnivariateSpline(
        lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')


print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[1:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))),
          'ell<500=', 1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) /
                                   clbb_lensed(np.arange(4, 500, 75))),
          'ell<1000=', 1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(np.arange(4, 1000, 100))
                                    ), 'ell<1500=', 1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(np.arange(4, 1500, 100)))
          )
    print('')

print('')
print('')


# In[ ]:

# noise_uK_arcmin=4.5,
# fwhm_arcmin=4.,
lmax = 3000
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 1
fwhm_arcmin = 1.
r_fid = 0.
print('')
print(Fore.YELLOW + 'r=0')
print('')
for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb(0., lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    print((probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt))


# ## LSS minus SKA

# In[ ]:

#  'wise', 'euclid', 'des_weak', 'lsst', 'ska10',
# #              'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
labels = ['wise', 'euclid', 'lsst', 'cib', 'desi', 'des']
cmb = 'S4'
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_euclid.txt', 'rho_lsst.txt',
             'rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
nle = nl(1, 1, lmax=ells_cmb[-1])[2:]
B_res3 = rho_to_Bres.main(rho_names, nle)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[1]] = InterpolatedUnivariateSpline(
        lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')


print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[1:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))),
          'ell<500=', 1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) /
                                   clbb_lensed(np.arange(4, 500, 75))),
          'ell<1000=', 1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(np.arange(4, 1000, 100))
                                    ), 'ell<1500=', 1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(np.arange(4, 1500, 100)))
          )
    print('')

print('')
print('')

# noise_uK_arcmin=4.5,
# fwhm_arcmin=4.,
lmax = 3000
# This needs to be Bicep like, the value of the deep exp
noise_uK_arcmin = 1
fwhm_arcmin = 1.
r_fid = 0.
print('')
print(Fore.YELLOW + 'r=0')
print('')
for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    sigma_r, sigma_nt, sigr, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    sigma_r_1, sigma_nt_1, sigr_1, _ = fisher_r_nt(
        r_fid=r_fid,
        lmin=4,
        lmax=lmax,
        clbb_cov=clbb(0., lmax=lmax),
        noise_uK_arcmin=noise_uK_arcmin,
        fwhm_arcmin=fwhm_arcmin)
    print((probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt))


# ---
sys.exit('CRoss checks are done after this')
#  - - -

# # Cross check literature results

# ### Kimmy

# In[46]:

pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars.set_cosmology(
    H0=67.11, ombh2=0.022, omch2=0.12, mnu=0.085, omk=0, tau=0.0925)
pars.InitPower.set_params(As=2.21e-9, ns=0.9624, r=0.2, nt=0)
pars.set_for_lmax(2000, lens_potential_accuracy=2)
# pars.set_for_lmax?

pars.AccurateBB = True
pars.OutputNormalization = False
pars.WantTensors = True
pars.DoLensing = True
pars.max_l_tensor = 1000
pars.max_eta_k_tensor = 1200.0 * 2
# print(pars) # if you want to test parasm
results = camb.get_results(pars)
results = camb.get_transfer_functions(pars)
tens_cl = results.get_tensor_cls(1000)


# In[45]:

clbb.cache_clear()
clbb_tens.cache_clear()


# In[46]:

lmin = 10
lmax = 500
sigma_lens = []
sigma_no_lens = []
noise_104 = {}
noise_105 = {}
noise_106 = {}
fsky = 0.25
noise_104[fsky] = 3.34 * np.sqrt(2)
noise_105[fsky] = 1.06 * np.sqrt(2)
noise_106[fsky] = 0.33 * np.sqrt(2)

fsky = 0.5
noise_104[fsky] = 4.73 * np.sqrt(2)
noise_105[fsky] = 1.5 * np.sqrt(2)
noise_106[fsky] = 0.47 * np.sqrt(2)

fsky = 0.75
noise_104[fsky] = 5.79 * np.sqrt(2)
noise_105[fsky] = 1.83 * np.sqrt(2)
noise_106[fsky] = 0.58 * np.sqrt(2)

# 10^4 detects

for r in [0.2, 0.1, 0.05, 0.02]:
    for fsky in [0.25, 0.5, 0.75]:
        print(('Ndet = 10^4', 'fsky =', fsky, 'r = ', r))
        #         sigma,sigmant = fisher_r_nt(lmin=10,lmax=3000,clbb_cov = clbb_tens(r),r_fid =r ,noise_uK_arcmin = noise_104[fsky],fid=0.,fsky=fsky)
        #         sigma_no_lens.append(sigmant)
        sigma, sigmant, _, _ = fisher_r_nt(
            lmin=lmin,
            lmax=lmax,
            r_fid=r,
            noise_uK_arcmin=noise_104[fsky],
            fid=0.,
            fsky=fsky)
        print((sigma, sigmant))
        sigma_lens.append(sigmant)

# 10^5 detects

for r in [0.2, 0.1, 0.05, 0.02]:
    for fsky in [0.25, 0.5, 0.75]:
        print(('Ndet = 10^5', 'fsky =', fsky, 'r = ', r))
        #         sigma,sigmant = fisher_r_nt(lmin=10,lmax=3000,clbb_cov = clbb_tens(r),r_fid =r ,noise_uK_arcmin = noise_105[fsky] ,fid=0.,fsky=fsky);
        #         sigma_no_lens.append(sigmant)
        sigma, sigmant, _, _ = fisher_r_nt(
            lmin=lmin,
            lmax=lmax,
            r_fid=r,
            noise_uK_arcmin=noise_105[fsky],
            fid=0.,
            fsky=fsky)

        print((sigma, sigmant))

        sigma_lens.append(sigmant)

# 10^6 detects

for r in [0.2, 0.1, 0.05, 0.02]:
    for fsky in [0.25, 0.5, 0.75]:
        print(('Ndet = 10^6', 'fsky =', fsky, 'r = ', r))
        #         sigma,sigmant = fisher_r_nt(lmin=10,lmax=3000,clbb_cov = clbb_tens(r),r_fid =r ,noise_uK_arcmin = noise_106[fsky] ,fid=0.,fsky=fsky);
        #         sigma_no_lens.append(sigmant)
        sigma, sigmant, _, _ = fisher_r_nt(
            lmin=lmin,
            lmax=lmax,
            r_fid=r,
            noise_uK_arcmin=noise_106[fsky],
            fid=0.,
            fsky=fsky)

        print(sigma, sigmant)

        sigma_lens.append(sigmant)


# ## Seems like it is working!

# ### Simard et al.

# In[50]:

pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
# They used Planck 2013 for their parameters. See https://arxiv.org/abs/1303.5076
pars.set_cosmology(
    H0=67.5, ombh2=0.022, omch2=0.12, mnu=0.00, omk=0, tau=0.089)
pars.InitPower.set_params(As=2.2e-9, ns=0.962, r=0.2, nt=-0.2 / 8.)
pars.set_for_lmax(3100, lens_potential_accuracy=2)
# pars.set_for_lmax?
pars.AccurateBB = True
pars.OutputNormalization = False
pars.WantTensors = True
pars.DoLensing = True
pars.max_l_tensor = 1500
pars.max_eta_k_tensor = 1200.0 * 3
# print(pars) # if you want to test parasm
results = camb.get_results(pars)
results = camb.get_transfer_functions(pars)


# In[ ]:


# #### Look at Fig 3.

# In[51]:

clbb.cache_clear()
clbb_tens.cache_clear()


# In[62]:

sigma_lens = []
sigma_no_lens = []
for r in np.linspace(0.0, 0.25, 20):
    print(r)
    sigma, sigmant, _, _ = fisher_r_nt(
        lmin=10,
        lmax=3000,
        clbb_cov=clbb_tens(r, -r / 8.),
        r_fid=r,
        noise_uK_arcmin=1)
    sigma_no_lens.append(sigmant)
    sigma, sigmant, _, _ = fisher_r_nt(
        lmin=10, lmax=3000, r_fid=r, noise_uK_arcmin=1)
    sigma_lens.append(sigmant)


# In[65]:

plt.plot(np.linspace(0.01, 0.25, 20), sigma_no_lens)
plt.plot(np.linspace(0.01, 0.25, 20), sigma_lens)
plt.ylim(0, 0.2)
plt.xlim(0.02,)
plt.xlabel(r'$r_{fid}$')
plt.ylabel(r'$\sigma(n_{T})$')


# In[60]:

sigma_lens = []
sigma_no_lens = []
for noise_p in np.linspace(0.1, 7, 20):
    print(noise_p)
    sigma, sigmant, _, _ = fisher_r_nt(
        lmin=10, lmax=3000, clbb_cov=clbb_tens(0.2), r_fid=0.2, noise_uK_arcmin=noise_p)
    sigma_no_lens.append(sigmant)
    sigma, sigmant, _, _ = fisher_r_nt(lmin=10, lmax=3000, r_fid=0.2, noise_uK_arcmin=noise_p)
    sigma_lens.append(sigmant)


# In[61]:

plt.plot(np.linspace(0.1, 7, 20), sigma_no_lens)
plt.plot(np.linspace(0.1, 7, 20), sigma_lens)
plt.ylim(0, 0.1)
plt.xlabel(r'$\Delta_{P}$')
plt.ylabel(r'$\sigma(n_{T})$')


# ## Cross Check with delensing works

# - - -

# Toshiya SKA?

# In[145]:

pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
# They used Planck 2013 for their parameters. See https://arxiv.org/abs/1303.5076
pars.set_cosmology(
    H0=67.5, ombh2=0.022, omch2=0.12, mnu=0.00, omk=0, tau=0.089)
pars.InitPower.set_params(As=2.2e-9, ns=0.962, r=0.2, nt=-0.2 / 8.)
pars.set_for_lmax(3100, lens_potential_accuracy=2)
# pars.set_for_lmax?
pars.AccurateBB = True
pars.OutputNormalization = False
pars.WantTensors = True
pars.DoLensing = True
pars.max_l_tensor = 1500
pars.max_eta_k_tensor = 1200.0 * 3
# print(pars) # if you want to test parasm
results = camb.get_results(pars)
results = camb.get_transfer_functions(pars)


# In[146]:

clbb.cache_clear()
clbb_tens.cache_clear()


# In[187]:

labels = ['ska10', 'ska01', 'ska5', 'ska1']
# surveys = ['k', 'euclid', 'des_weak', 'lsst', 'ska10', 'ska01',
#            'ska5', 'ska1', 'cib', 'desi', 'des']
cmb = 'S3'
get_ipython().magic(u'run -i multiple_survey_delens.py')


# In[189]:

plt.plot(lbins, rho['ska01']**2, label='01')
plt.plot(lbins, rho['ska5']**2, label='5')
plt.plot(lbins, rho['ska10']**2, label='10')
plt.plot(lbins, rho['ska1']**2, label='1')
plt.xlim(0, 3000)
plt.ylim(0, 1)
plt.grid()
plt.legend()


# In[90]:

plt.plot(lbins, rho['ska01']**2, label='01')
plt.plot(lbins, rho['ska5']**2, label='5')
plt.plot(lbins, rho['ska10']**2, label='10')
plt.plot(lbins, rho['ska1']**2, label='1')
plt.xlim(0, 400)
plt.ylim(0.5, 1)
plt.grid()
plt.legend()


# In[191]:

rho_ska_5 = rho_to_Bres.compute_res(
    '/home/manzotti/galaxies_delensing/Data/limber_spectra/rho_ska5.txt')
rho_ska_10 = rho_to_Bres.compute_res(
    '/home/manzotti/galaxies_delensing/Data/limber_spectra/rho_ska10.txt')
rho_ska_1 = rho_to_Bres.compute_res(
    '/home/manzotti/galaxies_delensing/Data/limber_spectra/rho_ska1.txt')
rho_ska_01 = rho_to_Bres.compute_res(
    '/home/manzotti/galaxies_delensing/Data/limber_spectra/rho_ska01.txt')


# In[192]:

clbb_test = rho_to_Bres.compute_res('test')


# In[193]:

ells = np.arange(4, 1500, 10)


# In[194]:

plt.plot(ells, np.array(rho_ska_5) / np.array(clbb_test), '.-', label='ska 5')
plt.plot(ells, np.array(rho_ska_10) / np.array(clbb_test), label='ska 10')
plt.plot(ells, np.array(rho_ska_01) / np.array(clbb_test), label='ska 01')
plt.plot(ells, np.array(rho_ska_1) / np.array(clbb_test), label='ska 1')
plt.legend()
plt.xlim(0, 200)
plt.ylim(0, 1)


# Alpha calcualtion

# In[162]:

plt.plot(clbb_res_fun_ska5(np.arange(0, len(clbb(0.0, lmax=1000)))))
plt.plot(clbb(0.0, 0., lmax=1000))


# In[195]:

clbb_res_fun_ska5 = scipy.interpolate.UnivariateSpline(
    ells, ells * (ells + 1) * np.array(rho_ska_5) / 2. / np.pi, k=3, s=0, ext=3)
clbb_res_fun_ska1 = scipy.interpolate.UnivariateSpline(
    ells, ells * (ells + 1) * np.array(rho_ska_1) / 2. / np.pi, k=3, s=0, ext=3)
clbb_res_fun_ska01 = scipy.interpolate.UnivariateSpline(
    ells, ells * (ells + 1) * np.array(rho_ska_01) / 2. / np.pi, k=3, s=0, ext=3)
clbb_res_fun_ska10 = scipy.interpolate.UnivariateSpline(
    ells, ells * (ells + 1) * np.array(rho_ska_10) / 2. / np.pi, k=3, s=0, ext=3)
clbb_res_fun_test = scipy.interpolate.UnivariateSpline(
    ells, ells * (ells + 1) * np.array(clbb_test) / 2. / np.pi, k=3, s=0, ext=3)


# In[197]:

sigma_r5, _, sigr, _ = fisher_r_nt(r_fid=0, fid=0., lmin=4, lmax=100, noise_uK_arcmin=2.,
                                   fwhm_arcmin=30., clbb_cov=clbb_res_fun_ska5(
                                       np.arange(0, len(clbb(0.0, lmax=100)))))


sigma_r1, _, sigr, _ = fisher_r_nt(r_fid=0, fid=0., lmin=4, lmax=100, noise_uK_arcmin=2.,
                                   fwhm_arcmin=30., clbb_cov=clbb_res_fun_ska1(
                                       np.arange(0, len(clbb(0.0, lmax=100)))))

sigma_r01, _, sigr, _ = fisher_r_nt(r_fid=0, fid=0., lmin=4, lmax=100, noise_uK_arcmin=2.,
                                    fwhm_arcmin=30., clbb_cov=clbb_res_fun_ska01(
                                        np.arange(0, len(clbb(0.0, lmax=100)))))

sigma_r10, _, sigr, _ = fisher_r_nt(r_fid=0, fid=0., lmin=4, lmax=100, noise_uK_arcmin=2.,
                                    fwhm_arcmin=30., clbb_cov=clbb_res_fun_ska10(
                                        np.arange(0, len(clbb(0.0, lmax=100)))))

sigma_r_1, _, sigr_1, _ = fisher_r_nt(r_fid=0, fid=0., lmin=4, lmax=100, noise_uK_arcmin=2.,
                                      fwhm_arcmin=30., clbb_cov=clbb_res_fun_test(np.arange(0, len(clbb(0.0, lmax=100)))))

print(('gain ska5= ', sigma_r_1 / sigma_r5))
print(('gain ska1 = ', sigma_r_1 / sigma_r1))
print(('gain ska10 = ', sigma_r_1 / sigma_r10))
print(('gain ska01 = ', sigma_r_1 / sigma_r01))


# In[198]:

noise_uK_arcmin = 2.
fwhm_arcmin = 30.
nlb = nl(noise_uK_arcmin, fwhm_arcmin, 100)
ell_nlb = np.arange(0, len(nlb))
nlb = nlb * ell_nlb * (ell_nlb + 1.) / 2. / np.pi


# In[199]:

print((np.mean((clbb_res_fun_test(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb) /
               (clbb_res_fun_ska10(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb))))

print((np.mean((clbb_res_fun_test(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb) /
               (clbb_res_fun_ska5(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb))))

print((np.mean((clbb_res_fun_test(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb) /
               (clbb_res_fun_ska1(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb))))

print((np.mean((clbb_res_fun_test(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb) /
               (clbb_res_fun_ska01(np.arange(0, len(clbb(0.0, lmax=100)))) + nlb))))


# Fairly good agreement

# In[ ]: