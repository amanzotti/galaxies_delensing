
# # Fisher Analysis

import numpy as np
import matplotlib.pyplot as plt
try:
    import functools32
except ImportError:
    import functools as functools32

import sys
# import os
import camb
from camb import model, initialpower
import multiple_z_med_delens
import configparser as ConfigParser
import rho_to_Bres
from scipy.interpolate import InterpolatedUnivariateSpline
from colorama import init
from colorama import Fore

init(autoreset=True)
np.seterr(divide='ignore', invalid='ignore')
# ============================================
# SET LATEX
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# ============================================

# cosmology values!!!!!!

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

# Remember there is a 7.4e12 missing and CAMB always give you $ \ell (\ell +1)/2\pi  $


@functools32.lru_cache(maxsize=64)
def clbb(r=0.1, nt=None, lmax=3000):
    inflation_params = initialpower.InitialPowerParams()
    if nt is None:
        nt = -r / 8.
    inflation_params.set_params(As=2.1e-9, r=r, nt=nt, pivot_tensor=0.01)
    results.power_spectra_from_transfer(inflation_params)
    return results.get_total_cls(lmax)[:, 2] * 7.42835025e12


@functools32.lru_cache(maxsize=64)
def clbb_tens(r=0.1, nt=None, lmax=3000):
    inflation_params = initialpower.InitialPowerParams()
    if nt is None:
        nt = -r / 8.
    inflation_params.set_params(As=2.1e-9, r=r, nt=nt, pivot_tensor=0.01)
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


inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')
# print('')
# print(Fore.RED + '')
# print('')


# # =====================================
# # TEST

# labels = ['wise', 'cib', 'des']
# cmb = 'S3'
# multiple_survey_delens.main(labels, cmb)
# rho_names = ['rho_cmb_' + cmb + '.txt']

# cmb = 'S4'
# multiple_survey_delens.main(labels, cmb)
# rho_names.append('rho_cmb_' + cmb + '.txt')
# # =====================================
rho_names = []
z_meadians = np.linspace(0.3, 2, 20)
gal_nums = np.linspace(0.01, 100, 20)
for z_mean in z_meadians:
    # label = 'z_median_{:.3}'.format(z_mean)
    for gal_num in gal_nums:
        rho_names.append(
            'rho_test_z_median_{:.3}_ngal_{:.1f}.txt'.format(z_mean, gal_num))

cmb = 'Planck'
lbins, rho, rho_comb_dict, rho_cmb = multiple_z_med_delens.main(
    cmb, z_means=z_meadians, gal_nums=gal_nums)
ells_cmb = np.loadtxt(output_dir + 'cmb_cl/ell.txt')
# rho_names = rho.keys()


# deep survey to delens or what is giving you E-mode
# BICEP level 3 muK and 30 arcmin beam

noise_uK_arcmin = 3.
fwhm_arcmin = 30.
# not used right now
ell_range_deep = [20, 400]
ell_range_high = [200, ells_cmb[-1]]
nle_deep = nl(noise_uK_arcmin, fwhm_arcmin, lmax=ells_cmb[-1])[2:]
nle_high = nl(9., 1., lmax=ells_cmb[-1])[2:]
# nle_high[:ell_range_high[0]] = np.inf
# nle_deep[:ell_range_deep[0]] = np.inf
# nle_deep[ell_range_deep[1]:] = np.inf
# nle = 1/(1/nle_high +1/nle_deep)
print(rho_names)

nle = nl(noise_uK_arcmin, fwhm_arcmin, lmax=ells_cmb[-1])[2:]
B_test = rho_to_Bres.main(['test'], nle)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_lensed = InterpolatedUnivariateSpline(
    lbins, lbins * (lbins + 1.) * np.nan_to_num(B_test) / 2. / np.pi, ext='extrapolate')
print(rho_names)
B_res3 = rho_to_Bres.main(rho_names, nle)
# B_res3, lbins = rho_to_Bres.load_res(rho_names)

lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')


clbb_res = {}
for i, probe in enumerate(rho_names):
    if probe == 'test':
        print(i, probe)
        clbb_res[probe] = InterpolatedUnivariateSpline(
            lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')
    else:
        print(i, probe.split('.txt')[0].split('rho_test_')[1])
        clbb_res[probe.split('.txt')[0].split('rho_test_')[1]] = InterpolatedUnivariateSpline(
            lbins, lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi, ext='extrapolate')


b_res_power = np.zeros((len(z_meadians), len(gal_nums)))

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for i, z_mean in enumerate(z_meadians):
    # label = 'z_median_{:.3}'.format(z_mean)
    for j, gal_num in enumerate(gal_nums):
        probe = 'rho_test_z_median_{:.3}_ngal_{:.1f}.txt'.format(
            z_mean, gal_num)
        probe = probe.split('.txt')[0].split('rho_test_')[1]
        b_res_power[i, j] = 1. - np.mean(clbb_res[probe](np.arange(4, 100, 25)
                                                         ) / clbb_lensed(np.arange(4, 100, 25)))
np.save('./b_res_power_matrix', b_res_power)
np.save('./z_meadians', z_meadians)
np.save('./gal_nums', gal_nums)

sys.exit()

print('')

print('')
print('')

# sys.exit()
lmax = 500
# This needs to be Bicep like, the value of the deep exp
r_fid = 0.
fsky = 0.06

print('')
print(Fore.YELLOW + 'r=0, no noise')
print('')
for i, label in enumerate(rho_names):
    probe = label.split('.txt')[0].split('rho_')[1]
    # delensed
    sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
        r_fid=r_fid,
        lmin=50,
        lmax=lmax,
        fsky=fsky,
        clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))
                                 ) + clbb_tens(r_fid, lmax=lmax),
        noise_uK_arcmin=0.,
        fwhm_arcmin=fwhm_arcmin)

    sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
        r_fid=r_fid,
        lmin=50,
        lmax=lmax,
        fsky=fsky,
        # clbb also contains r tensor contributions
        clbb_cov=clbb(r_fid, lmax=lmax),
        noise_uK_arcmin=0.,
        fwhm_arcmin=fwhm_arcmin)
    print('After delensing % errors', sigma_r_1 * 1e2)
    print('gain', (probe, 'gain = ', sigma_r_1 / sigma_r,
                   sigma_nt_1 / sigma_nt, sigr_1 / sigr, sigmant_1 / sigmant))

print('')
print('')


# Fairly good agreement

# In[ ]:
