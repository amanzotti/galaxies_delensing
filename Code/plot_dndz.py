
from __future__ import division
# from __future__ import print_function
print "\n Python start up script from /home/manzotti/.py_startup.py.\n"
import scipy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.integrate

import matplotlib.pyplot as plt

try:
    import pyfits
    import pdb

except:
    pass

try:
    import palettable

except Exception, e:
    pass
import os
import os
import sys
import time
import subprocess
import imp
from pprint import pprint
import __builtin__
import pyximport
from scipy.interpolate import RectBivariateSpline, interp1d
pyximport.install(reload_support=True)
try:
    cmap = palettable.colorbrewer.get_map('RdBu', 'diverging', 11, reverse=True).mpl_colormap
    plt.rcParams['image.cmap'] = cmap
except Exception, e:
    print 'probably palettable not found'


datadir = '/Users/alessandromanzotti/Work/Software/cosmosis_new/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/'

image_dir = '../images/'


# Import data
des_dndz = np.loadtxt(
    "/home/manzotti/cosmosis/modules/limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt")

gals_kernel = imp.load_source(
    'gals_kernel', '/home/manzotti/cosmosis/modules/limber/gals_kernel.py')
# kappa_kenel = imp.load_source('kappa_kenel', '/home/manzotti/cosmosis/modules/limber/kappa_kenel.py')
# gals_kern = imp.load_source('gals_kern', '/home/manzotti/cosmosis/modules/limber/gals_kernel.py')

# ============================================
# SIZE OF THE PICTURE
# ============================================
WIDTH = 462.0  # the number latex spits out
# the fraction of the width you'd like the figure to occupy, 0.5 if double
# column cause
FACTOR = 0.6
fig_width_pt = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
# golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
ratio = 0.8
fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * ratio   # figure height in inches
fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list
# ============================================
# ============================================


# ============================================
# SET LATEX
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# ============================================


font_size = 10.
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


# LOAD DNDZ
# =======================


# =======================
# DEFINE KERNELs
# CIB
# j2k = 1.e-6 / np.sqrt(83135.)  # for 353
# lkern = kappa_kernel.kern(zdist, hspline, omega_m, h0, xlss)
# cib = cib_hall.ssed_kern(
#     h0, zdist, chispline, hspline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})

desi_dndz = np.loadtxt("/home/manzotti/cosmosis/modules/limber/data_input/DESI/DESI_dndz.txt")
desi_dndz[:, 1] = np.sum(desi_dndz[:, 1:], axis=1)

dndzfun_desi = interp1d(desi_dndz[:, 0], desi_dndz[:, 1])
norm = scipy.integrate.quad(
    dndzfun_desi, desi_dndz[0, 0], desi_dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
# normalize
dndzfun_desi = interp1d(desi_dndz[:, 0], desi_dndz[:, 1] / norm)
# desi = gals_kernel.kern(desi_dndz[:, 0], dndzfun_desi, hspline, omega_m, h0, b=1.17)

# DES bias taken from Giannantonio et
# DES


# Weak lensing

# SKA
z_ska = np.linspace(0.01, 10, 600)
dndzska10 = gals_kernel.dNdZ_parametric_SKA_10mujk(z_ska)
dndzska1 = gals_kernel.dNdZ_parametric_SKA_1mujk(z_ska)
dndzska5 = gals_kernel.dNdZ_parametric_SKA_5mujk(z_ska)
dndzska01 = gals_kernel.dNdZ_parametric_SKA_01mujk(z_ska)

# ===
dndzfun = interp1d(z_ska, dndzska01)
norm = scipy.integrate.quad(dndzfun, z_ska[0], 10, limit=100, epsrel=1.49e-03)[0]
# print(norm)
# normalize
dndzska01 = interp1d(z_ska, dndzska01 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=0.1))
# ska01 = gals_kernel.kern(z_ska, dndzska01, hspline, omega_m, h0, b=1.)

# ===
dndzfun = interp1d(z_ska, dndzska1)
norm = scipy.integrate.quad(dndzfun, z_ska[0], 10, limit=100, epsrel=1.49e-03)[0]
# print(norm)

# normalize
dndzska1 = interp1d(z_ska, dndzska1 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=1))
# ska1 = gals_kernel.kern(z_ska, dndzska1, hspline, omega_m, h0, b=1.)

# ===
dndzfun = interp1d(z_ska, dndzska5)
norm = scipy.integrate.quad(dndzfun, z_ska[0], 10, limit=100, epsrel=1.49e-03)[0]
# print(norm)

# normalize
dndzska5 = interp1d(z_ska, dndzska5 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=5))
# ska5 = gals_kernel.kern(z_ska, dndzska5, hspline, omega_m, h0, b=1.)

# ===
dndzfun = interp1d(z_ska, dndzska10)
norm = scipy.integrate.quad(dndzfun, z_ska[0], 10, limit=100, epsrel=1.49e-03)[0]
# print(norm)

# normalize
dndzska10 = interp1d(z_ska, dndzska10 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=10))
# ska10 = gals_kernel.kern(z_ska, dndzska10, hspline, omega_m, h0, b=1.)

# LSST
z_lsst = np.linspace(0.01, 10, 200)
dndzlsst = gals_kernel.dNdZ_parametric_LSST(z_lsst)
dndzfun = interp1d(z_lsst, dndzlsst)

norm = scipy.integrate.quad(dndzfun, 0.01, 6, limit=100, epsrel=1.49e-03)[0]
# used the same bias model of euclid. Find something better
dndzlsst = interp1d(z_lsst, dndzlsst / norm * 1. * np.sqrt(1. + z_lsst))
# lsst = gals_kernel.kern(z_lsst, dndzlsst, hspline, omega_m, h0, b=1.)

# des_weak = kappa_gals_kernel.kern(z_lsst, dndzlsst, chispline, hspline, omega_m, h0)

# Euclid
z_euclid = np.linspace(0.01, 5, 200)
dndzeuclid = gals_kernel.dNdZ_parametric_Euclid(z_euclid)
dndzfun = interp1d(z_euclid, dndzeuclid)

norm = scipy.integrate.quad(dndzfun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
dndzeuclid = interp1d(z_euclid, dndzeuclid / norm * 1. * np.sqrt(1. + z_euclid))
# bias montanari et all for Euclid https://arxiv.org/pdf/1506.01369.pdf
# euclid = gals_kernel.kern(z_euclid, dndzeuclid, hspline, omega_m, h0, b=1.)


# PLOTS
fg = plt.figure(figsize=fig_dims)
# fg = plt.figure(figsize=(8, 10))

ax1 = fg.add_subplot(111)
z = np.arange(0, 6, 100)
plt.plot(des_dndz[:, 0], des_dndz[:, 1], label='D.E.S')
plt.plot(z_euclid, dndzeuclid(z_euclid), label='Euclid')
plt.plot(z_lsst, dndzlsst(z_lsst), label='LSST')
plt.plot(z_ska, dndzska10(z_ska), label='SKA 10')
plt.plot(z_ska, dndzska01(z_ska), label='SKA 0.1')
plt.plot(z_ska, dndzska5(z_ska), label='SKA 5')
plt.plot(z_ska, dndzska1(z_ska), label='SKA 1')
plt.plot(desi_dndz[:, 0], dndzfun_desi(desi_dndz[:, 0]), label='DESI')


plt.xlabel(r'z')
plt.ylabel(r' $dN/dz$ ')
plt.legend(loc='best')
plt.xlim(0, 6)
fg.tight_layout(pad=0.4)

# ============================================
# FINALLY SAVE
plt.savefig(image_dir + 'dndz.pdf', dpi=1200, papertype='Letter',
            format='pdf', transparent=True)

plt.clf()
