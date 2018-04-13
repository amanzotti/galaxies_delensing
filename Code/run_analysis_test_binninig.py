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
import multiple_survey_delens
import configparser as ConfigParser
import rho_to_Bres
from scipy.interpolate import InterpolatedUnivariateSpline
from colorama import init
from colorama import Fore
import datetime

sys.stdout = open(
    'run_test_binning' +
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S" + '.txt'), 'w')

init(autoreset=True)
np.seterr(divide='ignore', invalid='ignore')
# ============================================
# SET LATEX
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# ============================================


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


def run_fisher_cases(rho_names, lmin, lmax, deep, fsky=0.06, r_tens_case=0.07):
    print('')
    print(Fore.YELLOW + 'r=0, no noise')
    print('')

    cbb_lensed = clbb_lensed(np.arange(0, len(clbb(0.0, lmax=lmax))))
    r_fid = 0
    for i, label in enumerate(rho_names):
        probe = label.split('.txt')[0].split('rho_')[1]
        # delensed
        sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
            r_fid=r_fid,
            lmin=lmin,
            lmax=lmax,
            fsky=fsky,
            clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))) +
            clbb_tens(r_fid, lmax=lmax),
            noise_uK_arcmin=0.,
            fwhm_arcmin=deep['fwhm_arcmin'])

        sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
            r_fid=r_fid,
            lmin=lmin,
            lmax=lmax,
            fsky=fsky,
            # clbb also contains r tensor contributions
            clbb_cov=cbb_lensed + clbb_tens(r_fid, lmax=lmax),
            noise_uK_arcmin=0.,
            fwhm_arcmin=deep['fwhm_arcmin'])
        print('gain',
              (probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt))

    print('After delensing % errors sigma(r)*1e2', sigma_r * 1e2)
    print('After delensing % errors sigma(nt)', sigma_nt)
    print('')
    print('')

    print('')
    print(Fore.YELLOW + 'r=0')
    print('')
    r_fid = 1e-4

    for i, label in enumerate(rho_names):
        probe = label.split('.txt')[0].split('rho_')[1]
        sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
            r_fid=r_fid,
            lmin=lmin,
            lmax=lmax,
            fsky=fsky,
            clbb_cov=clbb_res[probe](np.arange(0, len(clbb(0.0, lmax=lmax)))) +
            clbb_tens(r_fid, lmax=lmax),
            noise_uK_arcmin=deep['noise_uK_arcmin'],
            fwhm_arcmin=deep['fwhm_arcmin'])
        sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
            r_fid=r_fid,
            lmin=lmin,
            lmax=lmax,
            fsky=fsky,
            clbb_cov=cbb_lensed + clbb_tens(r_fid, lmax=lmax),
            noise_uK_arcmin=deep['noise_uK_arcmin'],
            fwhm_arcmin=deep['fwhm_arcmin'])
        print('gain',
              (probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt))

    print('After delensing % errors sigma(r)*1e2', sigma_r * 1e2)
    print('After delensing % errors sigma(nt)', sigma_nt)
    print('')
    print('')

    # In[271]:

    clbb.cache_clear()
    clbb_tens.cache_clear()
    r_fid = r_tens_case
    fsky = 0.06

    # ### r=0.07
    print(Fore.YELLOW + 'r={}'.format(r_fid))
    print('')
    print('')

    for i, label in enumerate(rho_names):
        probe = label.split('.txt')[0].split('rho_')[1]
        sigma_r, sigma_nt, sigr, sigmant = fisher_r_nt(
            r_fid=r_fid,
            lmin=lmin,
            fsky=fsky,
            lmax=lmax,
            clbb_cov=clbb_res[probe](np.arange(0, len(clbb(r_fid, lmax=lmax))))
            + clbb_tens(r_fid, lmax=lmax),
            noise_uK_arcmin=deep['noise_uK_arcmin'],
            fwhm_arcmin=deep['fwhm_arcmin'])
        sigma_r_1, sigma_nt_1, sigr_1, sigmant_1 = fisher_r_nt(
            r_fid=r_fid,
            lmin=lmin,
            lmax=lmax,
            fsky=fsky,
            clbb_cov=cbb_lensed + clbb_tens(r_fid, lmax=lmax),
            noise_uK_arcmin=deep['noise_uK_arcmin'],
            fwhm_arcmin=deep['fwhm_arcmin'])

        print('gain',
              (probe, 'gain = ', sigma_r_1 / sigma_r, sigma_nt_1 / sigma_nt))

    print('After delensing % errors sigma(r)*1e2', sigma_r * 1e2)
    print('After delensing % errors sigma(nt)', sigma_nt)


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


def fisher_r_nt(r_fid=0.2,
                fid=None,
                lmin=10,
                lmax=2000,
                noise_uK_arcmin=4.5,
                fwhm_arcmin=4.,
                clbb_cov=None,
                fsky=0.5):

    # print('noise', noise_uK_arcmin, 'beam=', fwhm_arcmin)
    nlb = nl(noise_uK_arcmin, fwhm_arcmin, lmax=lmax)
    ell_nlb = np.arange(0, len(nlb))
    nlb = nlb * ell_nlb * (ell_nlb + 1.) / 2. / np.pi

    if fid is None:
        # print('n_t fis in None set consistency relation')
        fid = -r_fid / 8.
        # fid= 0.

    if clbb_cov is None:
        clbb_cov = clbb(r_fid, fid, lmax=lmax)

    Cov = np.sqrt(2. / (fsky *
                        (2. * np.arange(0, len(nlb)) + 1.))) * (clbb_cov + nlb)

    #     print(r_fid, fid,Cov)

    dx = r_fid * 0.02 + 0.03
    dBl_dr = (-clbb(r_fid + 2. * dx, fid, lmax=lmax) + 8. * clbb(
        r_fid + dx, fid, lmax=lmax) - 8. * clbb(r_fid - dx, fid, lmax=lmax) +
              clbb(r_fid - 2. * dx, fid, lmax=lmax)) / (12. * dx)
    #     print(dBl_dr)

    dx = fid * 0.03 + 0.03
    nt_deriv = (-clbb(r_fid, fid + 2 * dx, lmax=lmax) + 8. * clbb(
        r_fid, fid + dx, lmax=lmax) - 8. * clbb(r_fid, fid - dx, lmax=lmax) +
                clbb(r_fid, fid - 2 * dx, lmax=lmax)) / (12. * dx)
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


def combine_deep_high_res(deep_noise,
                          deep_fwhm,
                          highres_noise,
                          highres_fwhm,
                          lmin_deep=20,
                          lmax_deep=400,
                          lmin_highres=2000):
    deep = {}
    deep['noise_uK_arcmin'] = 3.
    deep['fwhm_arcmin'] = 30.
    # high res
    high_res = {}
    high_res['noise_uK_arcmin'] = 9.4
    high_res['fwhm_arcmin'] = 1.

    # not used right now
    ell_range_deep = [lmin_deep, lmax_deep]
    ell_range_high = [lmin_highres, ells_cmb[-1]]
    nle_deep = nl(
        deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
    nle_high = nl(
        deep['noise_uK_arcmin'], high_res['fwhm_arcmin'],
        lmax=ells_cmb[-1])[2:]
    nle_high[:ell_range_high[0]] = np.inf
    nle_deep[:ell_range_deep[0]] = np.inf
    nle_deep[ell_range_deep[1]:] = np.inf
    nle = 1 / (1 / nle_high + 1 / nle_deep)
    nle[np.where(nle == np.inf)] = 1e20
    return deep, nle


# cosmology values!!!!!!

pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars.set_cosmology(
    H0=67.26, ombh2=0.02222, omch2=0.1199, mnu=0.06, omk=0, tau=0.079)
pars.InitPower.set_params(ns=0.96, r=0., nt=0, pivot_tensor=0.01, As=2.1e-9)
pars.set_for_lmax(5000, lens_potential_accuracy=3)
pars.NonLinear = model.NonLinear_both

pars.AccurateBB = True
pars.OutputNormalization = False
pars.WantTensors = True
pars.DoLensing = True
pars.max_l_tensor = 3000
pars.max_eta_k_tensor = 3000.

# print(pars) # if you want to test parasm
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars)
# Remember there is a 7.4e12 missing and CAMB always give you $ \ell (\ell +1)/2\pi  $
totCL = powers['total']
cle = totCL[:, 1]
ells_cmb = np.arange(0, len(cle))
clp = powers['lens_potential'][:, 0] / (ells_cmb *
                                        (ells_cmb + 1))**2 * (2. * np.pi)
cle = cle * 7.42835025e12 / (ells_cmb * (ells_cmb + 1)) * (2. * np.pi)
clpp_fun = InterpolatedUnivariateSpline(
    ells_cmb[:5000], np.nan_to_num(clp[:5000]), ext=2)
clee_fun = InterpolatedUnivariateSpline(
    ells_cmb[:5000], np.nan_to_num(cle[:5000]), ext=2)

inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')

# ==========================================
# ==========================================
# L range to be used in the fisher forecasts
lmin = 50
lmax = 500

print('')
print(Fore.RED + 'PLANCK + DES ')
print('')

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

labels = ['des_bin0', 'des_bin1', 'des_bin2', 'des_bin3']
cmb = 'Planck'
multiple_survey_delens.main(labels, cmb)
ells_cmb = np.loadtxt(output_dir + 'cmb_cl/ell.txt')
rho_names = [
    'rho_cib.txt', 'rho_gals.txt', 'rho_wise.txt', 'rho_comb.txt',
    'rho_cmb_' + cmb + '.txt'
]

# deep survey to delens or what is giving you E-mode
# BICEP level 3 muK and 30 arcmin beam

# deep
deep = {}
deep['noise_uK_arcmin'] = 3.
deep['fwhm_arcmin'] = 30.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 60.
high_res['fwhm_arcmin'] = 7.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)

# ==========================================
#  Compute BB lens with our integral to have a fair comparison.
# Do so once for all here
# ==========================================

B_test = rho_to_Bres.main(['test'], nle_fun, clpp_fun, clee_fun)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_lensed = InterpolatedUnivariateSpline(
    lbins,
    lbins * (lbins + 1.) * np.nan_to_num(B_test) / 2. / np.pi,
    ext='extrapolate')

# # ================================================

B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)

clbb_res = {}
for i, probe in enumerate(rho_names):
    if probe == 'test':
        print(i, probe)
        clbb_res[probe] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')
    else:
        print(i, probe.split('.txt')[0].split('rho_')[1])
        clbb_res[probe.split('.txt')[0].split('rho_')[
            1]] = InterpolatedUnivariateSpline(
                lbins,
                lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
                ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(50, 100, 100)) / clbb_lensed(
              np.arange(50, 100, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))

print('')
print('')
print('')

# sys.exit()

# This needs to be Bicep like, the value of the deep exp

run_fisher_cases(rho_names, lmin, lmax, deep)

print('')
print(Fore.RED + 'PLANCK + DES no binning')
print('')

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

labels = ['des']
cmb = 'Planck'
multiple_survey_delens.main(labels, cmb)
ells_cmb = np.loadtxt(output_dir + 'cmb_cl/ell.txt')
rho_names = [
    'rho_des.txt', 'rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt'
]

# deep survey to delens or what is giving you E-mode
# BICEP level 3 muK and 30 arcmin beam

# deep
deep = {}
deep['noise_uK_arcmin'] = 3.
deep['fwhm_arcmin'] = 30.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 60
high_res['fwhm_arcmin'] = 7.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)

B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)

clbb_res = {}
for i, probe in enumerate(rho_names):
    if probe == 'test':
        print(i, probe)
        clbb_res[probe] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')
    else:
        print(i, probe.split('.txt')[0].split('rho_')[1])
        clbb_res[probe.split('.txt')[0].split('rho_')[
            1]] = InterpolatedUnivariateSpline(
                lbins,
                lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
                ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(50, 100, 100)) / clbb_lensed(
              np.arange(50, 100, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))

print('')
print('')
print('')

# sys.exit()

# This needs to be Bicep like, the value of the deep exp

run_fisher_cases(rho_names, lmin, lmax, deep)

# sys.exit()

print(Fore.RED + 'Actual scenario High res SPT-pol')
# In[274]:

labels = ['des_bin0', 'des_bin1', 'des_bin2', 'des_bin3']
cmb = 'now'
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode

# deep
deep = {}
deep['noise_uK_arcmin'] = 3.
deep['fwhm_arcmin'] = 30.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 9.4
high_res['fwhm_arcmin'] = 1.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)

# In[275]:

lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        10, 100, 25)) / clbb_lensed(np.arange(10, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(10, 500, 75)) / clbb_lensed(
              np.arange(10, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(50, 200, 100)) / clbb_lensed(
              np.arange(50, 200, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(10, 1500, 100)) / clbb_lensed(
              np.arange(10, 1500, 100))))
    print('')

print('')
print('')

# In[276]:

# This needs to be Bicep like, the value of the deep exp

run_fisher_cases(rho_names, lmin, lmax, deep)
# ====================================
# ====================================

print(Fore.RED + 'Actual scenario High res SPT-pol DES No binning')
# In[274]:

labels = ['des']
cmb = 'now'
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_des.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode

# deep
deep = {}
deep['noise_uK_arcmin'] = 3.
deep['fwhm_arcmin'] = 30.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 9.4
high_res['fwhm_arcmin'] = 1.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)

# In[275]:

lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        10, 100, 25)) / clbb_lensed(np.arange(10, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(10, 500, 75)) / clbb_lensed(
              np.arange(10, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(50, 200, 100)) / clbb_lensed(
              np.arange(50, 200, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(10, 1500, 100)) / clbb_lensed(
              np.arange(10, 1500, 100))))
    print('')

print('')
print('')

# In[276]:

# This needs to be Bicep like, the value of the deep exp

run_fisher_cases(rho_names, lmin, lmax, deep)

# CMB S3

# ====================================
# ====================================

# In[278]:
print('')
print(Fore.RED + 'CMB S3 + DESI')
print('')

labels = ['desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']

cmb = 'S3'
print(Fore.RED + 'Tracers:' + '-'.join(labels))
multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
# deep
deep = {}
deep['noise_uK_arcmin'] = 2.
deep['fwhm_arcmin'] = 30.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 3
high_res['fwhm_arcmin'] = 1.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

# deep survey to delens or what is giving you E-mode
nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(
              np.arange(4, 1000, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))
    print('')

print('')
print('')
print('')

run_fisher_cases(rho_names, lmin, lmax, deep)
print('')

# In[278]:
print('')
print(Fore.RED + 'CMB S3 + DESI no binning')
print('')

labels = ['desi']

cmb = 'S3'
print(Fore.RED + 'Tracers:' + '-'.join(labels))
multiple_survey_delens.main(labels, cmb)
rho_names = [
    'rho_desi.txt', 'rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt'
]
# deep survey to delens or what is giving you E-mode
# deep
deep = {}
deep['noise_uK_arcmin'] = 2.
deep['fwhm_arcmin'] = 30.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 3
high_res['fwhm_arcmin'] = 1.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

# deep survey to delens or what is giving you E-mode
nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(
              np.arange(4, 1000, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))
    print('')

print('')
print('')
print('')

run_fisher_cases(rho_names, lmin, lmax, deep)
print('')

# ==============================
# ==============================
# CMB-S4
# ==============================
# ==============================
# This needs to be Bicep like, the value of the deep exp

# deep
deep = {}
deep['noise_uK_arcmin'] = 1.5
deep['fwhm_arcmin'] = 15.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 1.
high_res['fwhm_arcmin'] = 1.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

print('')
print(Fore.RED + 'CMB S4 all binned')
print('')

labels = [
    'des_bin0', 'des_bin1', 'des_bin2', 'des_bin3', 'lsst_bin0', 'lsst_bin1',
    'lsst_bin2', 'lsst_bin3', 'lsst_bin4', 'lsst_bin5', 'lsst_bin6',
    'lsst_bin7', 'lsst_bin8', 'lsst_bin9', 'desi_bin0', 'desi_bin1',
    'desi_bin2', 'desi_bin3'
]
cmb = 'S4'

print(Fore.RED + 'Tracers:' + '-'.join(labels))

multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']

# deep survey to delens or what is giving you E-mode
nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(
              np.arange(4, 1000, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))
    print('')

print('')
print('')

run_fisher_cases(rho_names, lmin, lmax, deep)

# deep
deep = {}
deep['noise_uK_arcmin'] = 1.5
deep['fwhm_arcmin'] = 15.
# high res
high_res = {}
high_res['noise_uK_arcmin'] = 1.
high_res['fwhm_arcmin'] = 1.

# not used right now
ell_range_deep = [20, 800]
ell_range_high = [50, ells_cmb[-1]]
nle_deep = nl(
    deep['noise_uK_arcmin'], deep['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high = nl(
    deep['noise_uK_arcmin'], high_res['fwhm_arcmin'], lmax=ells_cmb[-1])[2:]
nle_high[:ell_range_high[0]] = np.inf
nle_deep[:ell_range_deep[0]] = np.inf
nle_deep[ell_range_deep[1]:] = np.inf
nle = 1 / (1 / nle_high + 1 / nle_deep)
nle[np.where(nle == np.inf)] = 1e20

print('')
print(Fore.RED + 'CMB S4 NO binned')
print('')

labels = ['lsst', 'desi', 'des']
cmb = 'S4'

print(Fore.RED + 'Tracers:' + '-'.join(labels))

multiple_survey_delens.main(labels, cmb)
rho_names = [
    'rho_desi.txt', 'rho_lsst.txt', 'rho_des.txt', 'rho_gals.txt',
    'rho_comb.txt', 'rho_cmb_' + cmb + '.txt'
]

# deep survey to delens or what is giving you E-mode
nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(
              np.arange(4, 1000, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))
    print('')

print('')
print('')

run_fisher_cases(rho_names, lmin, lmax, deep)

print('')
print(Fore.RED + 'CMB S4 + just lsst binned')
print('')

labels = [
    'lsst_bin0', 'lsst_bin1', 'lsst_bin2', 'lsst_bin3', 'lsst_bin4',
    'lsst_bin5', 'lsst_bin6', 'lsst_bin7', 'lsst_bin8', 'lsst_bin9'
]
cmb = 'S4'

print(Fore.RED + 'Tracers:' + '-'.join(labels))

multiple_survey_delens.main(labels, cmb)
rho_names = ['rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']

# deep survey to delens or what is giving you E-mode
nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(
              np.arange(4, 1000, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))
    print('')

print('')
print('')

run_fisher_cases(rho_names, lmin, lmax, deep)
# ## LSS minus SKA

print('')
print(Fore.RED + 'CMB S4 + just lsst NOT binned')
print('')

labels = ['lsst']
cmb = 'S4'

print(Fore.RED + 'Tracers:' + '-'.join(labels))

multiple_survey_delens.main(labels, cmb)
rho_names = [
    'rho_lsst.txt', 'rho_gals.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt'
]

# deep survey to delens or what is giving you E-mode
nle_fun = InterpolatedUnivariateSpline(np.arange(0, len(nle)), nle, ext=2)
B_res3 = rho_to_Bres.main(rho_names, nle_fun, clpp_fun, clee_fun)
lbins = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')
clbb_res = {}
for i, probe in enumerate(rho_names):
    print(i, probe.split('.txt')[0].split('rho_')[1])
    clbb_res[probe.split('.txt')[0].split('rho_')[
        1]] = InterpolatedUnivariateSpline(
            lbins,
            lbins * (lbins + 1.) * np.nan_to_num(B_res3[i]) / 2. / np.pi,
            ext='extrapolate')

print('')
print(Fore.YELLOW + 'Fraction of removed Bmode power')
for probe in rho_names[0:]:
    probe = probe.split('.txt')[0].split('rho_')[1]
    print(probe)
    print('ell<100=', 1. - np.mean(clbb_res[probe](np.arange(
        4, 100, 25)) / clbb_lensed(np.arange(4, 100, 25))), 'ell<500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 500, 75)) / clbb_lensed(
              np.arange(4, 500, 75))), 'ell<1000=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1000, 100)) / clbb_lensed(
              np.arange(4, 1000, 100))), 'ell<1500=',
          1. - np.mean(clbb_res[probe](np.arange(4, 1500, 100)) / clbb_lensed(
              np.arange(4, 1500, 100))))
    print('')

print('')
print('')

run_fisher_cases(rho_names, lmin, lmax, deep)
# ## LSS minus SKA
