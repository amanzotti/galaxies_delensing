

# from __future__ import print_function
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
import os
import sys
import time
import subprocess

from pprint import pprint
import builtins
import pyximport
pyximport.install(reload_support=True)

# try:
#     cmap = palettable.colorbrewer.get_map('RdBu', 'diverging', 11, reverse=True).mpl_colormap
#     plt.rcParams['image.cmap'] = cmap
# except Exception as e:
#     print('probably palettable not found')


datadir = '../../Data/limber_spectra/'

image_dir = '../../images/'


# Import data
kappakernel_l30 = np.loadtxt(datadir + 'kappa_kernel_l30.txt')
kappakernel_l100 = np.loadtxt(datadir + 'kappa_kernel_l100.txt')
kappakernel_l500 = np.loadtxt(datadir + 'kappa_kernel_l500.txt')

lsstkernel_l100 = np.loadtxt(datadir + 'lsst_kernel_l100.txt')

skakernel_l100 = np.loadtxt(datadir + 'ska_kernel_l100.txt')
weakkernel_l100 = np.loadtxt(datadir + 'weak_kernel_l100.txt')


# cibkernel_l30 = np.loadtxt(datadir + 'cib_kernel_l30.txt')
cibkernel_l100 = np.loadtxt(datadir + 'cib_kernel_l100.txt')
# cibkernel_l500 = np.loadtxt(datadir + 'cib_kernel_l500.txt')


deskernel_l30 = np.loadtxt(datadir + 'des_kernel_l30.txt')
deskernel_l100 = np.loadtxt(datadir + 'des_kernel_l100.txt')
deskernel_l500 = np.loadtxt(datadir + 'des_kernel_l500.txt')


desikernel_l30 = np.loadtxt(datadir + 'desi_kernel_l30.txt')
desikernel_l100 = np.loadtxt(datadir + 'desi_kernel_l100.txt')
desikernel_l500 = np.loadtxt(datadir + 'desi_kernel_l500.txt')


# ============================================
# SIZE OF THE PICTURE
# ============================================
WIDTH = 246.0  # the number latex spits out
# the fraction of the width you'd like the figure to occupy, 0.5 if double
# column cause
FACTOR = 0.99
fig_width_pt = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
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


font_size = 8.
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
plt.rcParams['legend.handlelength'] = font_size /9.
# ============================================


# PLOTS
fg = plt.figure(figsize=fig_dims)
# fg = plt.figure(figsize=(8, 10))
print(fig_dims)
ax1 = fg.add_subplot(111)

plt.plot(kappakernel_l30[:, 0], kappakernel_l100[:, 1] /
         np.max(kappakernel_l100[:, 1]), label=r'$CMB$', alpha=0.8)
# plt.plot(kappakernel_l30[:, 0], kappakernel_l100[:, 1] /
#          np.max(kappakernel_l100[:, 1]), color='#e41a1c', linestyle=':')
# plt.plot(kappakernel_l30[:, 0], kappakernel_l500[:, 1] /
#          np.max(kappakernel_l500[:, 1]), color='#e41a1c', linestyle='--')

plt.plot(cibkernel_l100[:, 0], cibkernel_l100[:, 1] /
         np.max(cibkernel_l100[:, 1]), label=r'$CIB$')
# plt.plot(cibkernel_l30[:, 0], cibkernel_l100[:, 1] /
#          np.max(cibkernel_l100[:, 1]), color='#377eb8', linestyle=':')
# plt.plot(cibkernel_l30[:, 0], cibkernel_l500[:, 1] /
#          np.max(cibkernel_l500[:, 1]), color='#377eb8', linestyle='--')

plt.plot(deskernel_l30[:, 0], deskernel_l100[:, 1] /
         np.max(deskernel_l100[:, 1]), label=r'$DES$', alpha=0.8)
# plt.plot(deskernel_l30[:, 0], deskernel_l100[:, 1] /
#          np.max(deskernel_l100[:, 1]), color='#4daf4a', linestyle=':')
# plt.plot(deskernel_l30[:, 0], deskernel_l500[:, 1] /
#          np.max(deskernel_l500[:, 1]), color='#4daf4a', linestyle='--')


plt.plot(skakernel_l100[:, 0], skakernel_l100[:, 1] /
         np.max(skakernel_l100[:, 1]), label=r'$SKA$', alpha=0.8)


# plt.plot(lsstkernel_l100[:, 0], lsstkernel_l100[:, 1] /
#          np.max(lsstkernel_l100[:, 1]), label=r'$LSST$', alpha=0.8)

plt.plot(desikernel_l30[:, 0], desikernel_l100[:, 1] /
         np.max(desikernel_l100[:, 1]), label=r'$DESI$', alpha=0.8)

# plt.plot(weakkernel_l100[:, 0], weakkernel_l100[:, 1] /
#          np.max(weakkernel_l100[:, 1]), label=r'$Weak Lensing$', alpha=0.8)


# plt.plot(desikernel_l30[:,0],deskernel_l100[:,1]/np.max(desikernel_l100[:,1]),color='#984ea3',linestyle=':')
# plt.plot(desikernel_l30[:,0],desikernel_l500[:,1]/np.max(desikernel_l500[:,1]),color='#984ea3',linestyle='--')

# print(desikernel_l30[:, 1] /
#       np.max(desikernel_l30[:, 1]))
plt.xlabel(r'z')
plt.ylabel(r' $W(z)$ ')
plt.legend(loc='best', ncol =3 ,mode='expand' )
plt.xlim(0, 4.5)
plt.ylim(0, 1.4)
fg.tight_layout()

# ============================================
# FINALLY SAVE
plt.savefig(image_dir + 'compare_kernel.pdf', papertype='Letter')
plt.savefig(image_dir + 'compare_kernel.png', papertype='Letter')
plt.clf()
