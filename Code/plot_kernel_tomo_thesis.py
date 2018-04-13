import scipy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import pyfits
    import pdb

except:
    pass

# try:
#     import palettable

# except Exception as e:
#     pass
import os
import sys
import time
import subprocess
import pyximport
pyximport.install(reload_support=True)
import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
# import kappa_gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import limber_integrals
import pickle as pk
import camb
from camb import model
import configparser
from joblib import Parallel, delayed
# from profiling.sampling import SamplingProfiler
# profiler = SamplingProfiler()
import DESI
from pprint import pprint
import builtins

from matplotlib import cm

# try:
#     cmap = palettable.colorbrewer.get_map('RdBu', 'diverging', 11, reverse=True).mpl_colormap
#     plt.rcParams['image.cmap'] = cmap
# except Exception as e:
#     print('probably palettable not found')


def find_bins(z, dndz, nbins):
    '''
    Function that, given a redshift distribution returns binning with equal number of bins.

    It returns a list of the redshift of the bins.
    '''
    cum = np.cumsum(dndz)
    args = np.hstack(
        (np.array(0.),
         np.searchsorted(
             cum, [cum[-1] / nbins * n
                   for n in np.arange(1, nbins + 1)]))).astype(np.int)
    args[-1] = len(cum) - 1
    return [z[args[i]:args[i + 1]] for i in np.arange(0, len(args) - 1.)], args


def make_tomo_bins(z, dndz, sigmaz, width, nbins, hspline, omegac, h, b=1.):
    '''
    this function takes a full dndz distribution and given the number of bins and their width.sigma z returns a list of different tomographic bins
    '''
    lsst_tomo_bins = []
    for n in np.arange(1, nbins + 1):
        dndz_win = dndz * (scipy.special.erfc(
            (width *
             (n - 1) - z) / (sigmaz * np.sqrt(2))) - scipy.special.erfc(
                 (width * n - z) / (sigmaz * np.sqrt(2))))
        dndzlsst_temp = InterpolatedUnivariateSpline(z, dndz_win, ext='zeros')
        norm = dndzlsst_temp.integral(z[0], z[-1])
        # print(norm, dndz_win, z)
        dndzlsst_temp_fun = InterpolatedUnivariateSpline(
            z, dndz_win / norm * np.sqrt(1. + z), ext='zeros')
        lsst_tomo_bins.append(
            gals_kernel.kern(z, dndzlsst_temp_fun, hspline, omegac, h, b=1.))
    return lsst_tomo_bins


def make_tomo_bins_equal_gals(z, dndz, sigmaz, nbins, hspline, omegac, h,
                              b=1.):
    '''
    this function takes a full dndz distribution and given the number of bins and their width.sigma z returns a list of different tomographic bins
    '''

    z_bins = find_bins(z, dndz, nbins)

    # print('z_bins', z_bins)
    def p_z_ph_z(z_ph, z, sigma_z):
        return np.exp(-(z_ph - z)**2 /
                      (2. * sigma_z**2)) / np.sqrt(2 * np.pi * sigma_z**2)

    # print('print', z, dndz, sigmaz, z_bins)
    lsst_tomo_bins = []
    galaxies_fraction = []
    for n in range(0, len(z_bins[0])):
        # print('n',n,len(z_bins[0]),z[int(z_bins[1][n])], z[int(z_bins[1][n + 1])])
        # print([z_val for i, z_val in enumerate(z)])
        # print(int(z_bins[1][n]),int(z_bins[1][
        #                                          n + 1]))
        photoz_confusion = [
            scipy.integrate.quad(
                p_z_ph_z,
                z[int(z_bins[1][n])],
                z[int(z_bins[1][n + 1])],
                args=(z_val, sigmaz[i]),
                limit=600,
                epsabs=0,
                epsrel=1.49e-03)[0] for i, z_val in enumerate(z)
        ]

        dndz_win = dndz * photoz_confusion
        dndzlsst_temp = InterpolatedUnivariateSpline(z, dndz_win, ext='zeros')
        norm = dndzlsst_temp.integral(z[0], z[-1])
        # print(norm, dndz_win, z)
        galaxies_fraction.append(norm)
        dndzlsst_temp_fun = InterpolatedUnivariateSpline(
            z, dndz_win / norm * np.sqrt(1. + z), ext='zeros')
        lsst_tomo_bins.append(
            gals_kernel.kern(z, dndzlsst_temp_fun, hspline, omegac, h, b=1.))
    return lsst_tomo_bins, np.array(galaxies_fraction)


def make_spec_bins(z, dndz_fun, nbins, hspline, omegac, h, b=1.):
    '''
    this takes a dndz distribution and splits it in nbins of equal lenght in z. this is quite rudimentary
    '''
    spec_bins = []
    galaxies_fraction = []
    z_bins = [
        z[i:i + int(len(z) / nbins) + 1]
        for i in range(0, len(z), int(len(z) / nbins))
    ]
    # +1 is inserted not to have gaps
    for z in z_bins:
        dndz = dndz_fun(z)
        # print(z, dndzlsst)
        dndz_temp = InterpolatedUnivariateSpline(z, dndz, ext='zeros')
        norm = dndz_temp.integral(z[0], z[-1])
        galaxies_fraction.append(norm)
        # print('norm', norm)
        dndz_bin = InterpolatedUnivariateSpline(z, dndz / norm, ext='zeros')
        spec_bins.append(
            gals_kernel.kern(z, dndz_bin, hspline, omegac, h, b=1.))
    # sys.exit()
    return spec_bins, np.array(galaxies_fraction)


# SET UP CAMB
pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars.set_cosmology(
    H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(3500, lens_potential_accuracy=0)
pars.NonLinear = model.NonLinear_both
pars.set_matter_power(redshifts=np.linspace(0., 13, 50), kmax=5.0)
results = camb.get_results(pars)

# P(z,k)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(
    minkh=1e-6, maxkh=5, have_power_spectra=False, npoints=250)
rbs = RectBivariateSpline(kh_nonlin, z_nonlin, pk_nonlin.T)
h = pars.H0 / 100.

# Distances
# =======================
# distance to last scattering surface
xlss = (results.conformal_time(0) - model.tau_maxvis.value) * h
# spline the redshift and the comoving distance
z = np.linspace(0, 15, 100)[::-1]
chispline = InterpolatedUnivariateSpline(
    np.linspace(0, 15, 100),
    results.comoving_radial_distance(np.linspace(0, 15, 100)) * h,
    ext=0)
hspline = InterpolatedUnivariateSpline(
    np.linspace(0, 15, 100), [
        results.hubble_parameter(z_vector) / pars.H0 / 3000.
        for z_vector in np.linspace(0, 15, 100)
    ],
    ext=0)

# GROWTH

growth = InterpolatedUnivariateSpline(
    np.linspace(0, 15, 100),
    np.sqrt((rbs(0.01, np.linspace(0, 15, 100)) / rbs(0.01, 0)))[0])

# LOAD DNDZ
# =======================
# alternative dndz from Sam email

# res = pk.load(open('/home/manzotti/cosmosis/modules/limber/data_input/DES/des.pkl'))
# spline = res['spline']
# N = res['N']
dndz = np.loadtxt(
    '/home/manzotti/cosmosis/modules/limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt'
)
dndzfun = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1], ext=2)
norm = scipy.integrate.quad(
    dndzfun, dndz[0, 0], dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
# print('norm', norm)
# normalize
dndzfun = InterpolatedUnivariateSpline(
    dndz[:, 0], dndz[:, 1] / norm, ext='zeros')
des = gals_kernel.kern(dndz[:, 0], dndzfun, chispline, pars.omegac, h, b=1.)
sigmaz = 0.05 * (dndz[:, 0] + 1.)
# width = z_lsst[-1] / nbins
# print(dndz[:, 0].shape, dndz[:, 1].shape)
des_tomo_bins, galaxies_fraction_des = make_tomo_bins_equal_gals(
    dndz[:, 0],
    dndz[:, 1],
    sigmaz=sigmaz,
    nbins=3,
    hspline=hspline,
    omegac=pars.omegac,
    h=h,
    b=1.)
# print('frac',galaxies_fraction_des/norm)

# ======
# DESI
# =======

#     desi_dndz = np.loadtxt("/home/manzotti/cosmosis/modules/limber/data_input/DESI/DESI_dndz.txt")
#     desi_dndz[:, 1] = np.sum(desi_dndz[:, 1:], axis=1)

#     dndzfun_desi = interp1d(desi_dndz[:, 0], desi_dndz[:, 1])
# # Use sam desi

# make_spec_bins(desi_dndz[:, 0], dndzfun_desi, nbins=2,
#                hspline=hspline, omegac=pars.omegac, h=h, b=1.17)

# norm = scipy.integrate.quad(
#     dndzfun_desi, desi_dndz[0, 0], desi_dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
# # normalize
# dndzfun_desi = InterpolatedUnivariateSpline(
#     desi_dndz[:, 0], desi_dndz[:, 1] / norm, ext='zeros')
# desi1 = gals_kernel.kern(desi_dndz[:, 0], dndzfun_desi, hspline, pars.omegac, h, b=1.17)

# DESI SAM
# print('norm' ,scipy.integrate.quad(DESI.DESISpline_normalized, 0, 3, limit=600, epsabs=0. , epsrel=1.49e-03)[0])

desi = gals_kernel.kern(
    np.linspace(0, 2, 100),
    DESI.DESISpline_normalized,
    hspline,
    pars.omegac,
    h,
    b=1.17)
desi_spec_bins, galaxies_fraction_desi = make_spec_bins(
    np.linspace(0, 2, 100),
    DESI.DESISpline_normalized,
    4,
    hspline,
    pars.omegac,
    h,
    b=1.)
# print('DESI',galaxies_fraction_desi, np.sum(galaxies_fraction_desi))

# ======
# LSST
# =======
z_lsst = np.linspace(0.01, 4.5, 200)
dndzlsst = gals_kernel.dNdZ_parametric_LSST(z_lsst)
dndzfun = interp1d(z_lsst, dndzlsst)

norm = scipy.integrate.quad(
    dndzfun, 0.01, z_lsst[-1], limit=100, epsrel=1.49e-03)[0]
# used the same bias model of euclid. Find something better
dndzlsst = InterpolatedUnivariateSpline(
    z_lsst, dndzlsst * 1. * np.sqrt(1. + z_lsst), ext='zeros')
lsst = gals_kernel.kern(z_lsst, dndzlsst, hspline, pars.omegac, h, b=1.)
nbins = 5

dndzlsst = gals_kernel.dNdZ_parametric_LSST(z_lsst)
sigmaz = 0.05 * (z_lsst + 1.)
# print('lsst', z_lsst.shape, dndzlsst.shape)
# width = z_lsst[-1] / nbins
lsst_tomo_bins, galaxies_fraction_lsst = make_tomo_bins_equal_gals(
    z_lsst,
    dndzlsst,
    sigmaz=sigmaz,
    nbins=10,
    hspline=hspline,
    omegac=pars.omegac,
    h=h,
    b=1.)
# print('frac',galaxies_fraction_lsst/norm)

# ======
# Euclid
# =======
z_euclid = np.linspace(0.01, 5, 200)
z_mean = 0.9
dndzeuclid = gals_kernel.dNdZ_parametric_Euclid(z_euclid, z_mean)
# dndzeuclid_deriv = gals_kernel.dNdZ_deriv_Euclid_ana(z_euclid,0.9)
z_mean_array = np.linspace(0.9 - 0.4, 0.9 + 0.4, 200)
dndzeuclid_param = RectBivariateSpline(
    z_mean_array, z_euclid,
    gals_kernel.dNdZ_parametric_Euclid_fulld(z_euclid, z_mean_array))
dndzfun = interp1d(z_euclid, dndzeuclid)
# dndzeuclid_deriv_fun = interp1d(
#     z_euclid, dndzeuclid_param.__call__(z_mean, z_euclid, dx=1, dy=0))

norm = scipy.integrate.quad(dndzfun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
# norm_deriv = scipy.integrate.quad(dndzeuclid_deriv_fun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
# dndzeuclid_deriv_fun = InterpolatedUnivariateSpline(
#     z_euclid, dndzeuclid_deriv_fun / norm_deriv * 1. * np.sqrt(1. + z_euclid))h
dndzeuclid = InterpolatedUnivariateSpline(
    z_euclid, dndzeuclid / norm * 1. * np.sqrt(1. + z_euclid), ext='zeros')
# bias montanari et all for Euclid https://arxiv.org/pdf/1506.01369.pdf

euclid = gals_kernel.kern(z_euclid, dndzeuclid, hspline, pars.omegac, h, b=1.)
nbins = 10
dndzeuclid = gals_kernel.dNdZ_parametric_LSST(z_euclid)
sigmaz = 0.05 * (z_euclid + 1.)
euclid_tomo_bins = make_tomo_bins_equal_gals(
    z_euclid,
    dndzeuclid,
    sigmaz=sigmaz,
    nbins=nbins,
    hspline=hspline,
    omegac=pars.omegac,
    h=h,
    b=1.)

z = np.linspace(0, 15, 100)[::-1]
lkern = kappa_kernel.kern(z, hspline, chispline, pars.omegac, h, xlss)

# ============================================
# SIZE OF THE PICTURE
# ============================================
WIDTH = 246.0 * 1.9  # the number latex spits out
# the fraction of the width you'd like the figure to occupy, 0.5 if double
# column cause
FACTOR = 0.99
fig_width_pt = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
ratio = 0.7
fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * ratio  # figure height in inches
fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list
# ============================================
# ============================================

# ============================================
# SET LATEX
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# ============================================

font_size = 16.
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.labelsize'] = font_size * 1.3
plt.rcParams['axes.linewidth'] = font_size / 10.
plt.rcParams['axes.titlesize'] = font_size * 1.3
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size / 1.2
plt.rcParams['ytick.labelsize'] = font_size / 1.2
plt.rcParams['xtick.major.width'] = font_size / 10.
plt.rcParams['ytick.major.width'] = font_size / 10.
# plt.rcParams['xtick.labelsize'] = font_size / 1.2
# plt.rcParams['ytick.labelsize'] = font_size / 1.2

# plt.rcParams['axes.color_cycle'] = '#e41a1c,#377eb8,#4daf4a,#984ea3,#ff7f00,#ffff33,#a65628'
plt.rcParams['lines.linewidth'] = font_size / 8.
# ============================================

# Simplify paths by removing "invisible" points, useful for reducing
# file size when plotting a large number of points
plt.rcParams['path.simplify'] = False
# ============================================

# ============================================

# Have the legend only plot one point instead of two, turn off the
# frame, and reduce the space between the point and the label

# ============================================
# LEGEND
# ============================================
# OPTION
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.handletextpad'] = 0.3
plt.rcParams['legend.handlelength'] = font_size / 9.
# ============================================

# PLOTS
fg = plt.figure(figsize=fig_dims)
# fg = plt.figure(figsize=(8, 10))
z = np.linspace(0, 5, 1000)
ax1 = fg.add_subplot(111)

number_of_lines = len(lsst_tomo_bins)
cm_subsection = np.linspace(0.3, 1., number_of_lines)

colors = [cm.Blues(x) for x in cm_subsection]

for i, bin_tomo in enumerate(lsst_tomo_bins):
    plt.plot(
        z,
        bin_tomo.dndzfun(z) / np.max(lsst.dndzfun(z)) *
        galaxies_fraction_lsst[i],
        color=colors[i],
        alpha=0.8)

plt.plot(
    z,
    lsst.dndzfun(z) / np.max(lsst.dndzfun(z)),
    alpha=0.8,
    color='#1f77b4',
    label='LSST')

# plt.plot(weakkernel_l100[:, 0], weakkernel_l100[:, 1] /
#          np.max(weakkernel_l100[:, 1]), label=r'$Weak Lensing$', alpha=0.8)

ell = 100
zs = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(zs)
for i, z in enumerate(zs):

    x = chispline(z)
    # w_kappa[i] = 1. / x * lkern.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_kappa[i] = lkern.w_lxz(ell, x, z)

plt.plot(
    zs, w_kappa / np.max(w_kappa), alpha=0.8, color='#d62728', label='CMB')

plt.xlabel(r'z')
plt.ylabel(r' $W(z)$ ')
plt.legend(loc='best')
plt.xlim(0, 5)
plt.ylim(0, 1.2)
fg.tight_layout()

# ============================================
# FINALLY SAVE
plt.savefig(
    '../images/' + 'compare_kernel_tomo_thesis.pdf', papertype='Letter')
plt.savefig(
    '../images/' + 'compare_kernel_tomo_thesis.png', papertype='Letter')
plt.clf()
