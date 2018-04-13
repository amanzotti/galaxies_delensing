from __future__ import division
# from __future__ import print_function
import scipy as sp
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
from itertools import cycle
import matplotlib.pyplot as plt

# try:
#     import palettable

# except ValueError:
#     pass
import os
import os
import sys
import time
import subprocess
import imp
from pprint import pprint
import pyximport
pyximport.install(reload_support=True)

# try:
#     cmap = palettable.colorbrewer.get_map('RdBu', 'diverging', 11, reverse=True).mpl_colormap
#     plt.rcParams['image.cmap'] = cmap

# except ValueError:
#     print('probably palettable not found')

image_dir = '../../images/'

# ============================================
# SIZE OF THE PICTURE
# ============================================
WIDTH = 700.0  # the number latex spits out
# the fraction of the width you'd like the figure to occupy, 0.5 if double
# column cause
FACTOR = 0.6
fig_width_pt = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
# golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
ratio = 0.8
fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * ratio  # figure height in inches
fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list
# ============================================
# ============================================

# ============================================
# SET LATEX
plt.rc('text', usetex=False)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# ============================================

font_size = 10.
plt.rcParams['font.size'] = font_size

plt.rcParams['axes.labelsize'] = font_size * 1.3
plt.rcParams['axes.linewidth'] = font_size / 22.
plt.rcParams['axes.titlesize'] = font_size * 1.3
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size / 1.2
plt.rcParams['ytick.labelsize'] = font_size / 1.2
# plt.rcParams['axes.color_cycle'] = '#e41a1c,#377eb8,#4daf4a,#984ea3,#ff7f00,#ffff33,#a65628'

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
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)

datadir = '/home/manzotti/cosmosis/modules/limber/galaxies_delens/'


def load_res(labels):
    datadir = '/home/manzotti/cosmosis/modules/limber/galaxies_delens/'
    res_list = {}
    for label in labels:
        res_list[label] = (
            np.loadtxt(datadir + 'limber_spectra/cbb_res_' + label + '.txt'))
    return res_list


surveys = ['cib', 'des']
# , 'comb_des_cib', 'comb_des_cib_cmb',
#            'ska10', 'ska5', 'ska1', 'ska01', 'lsst', 'euclid', 'rho_comb', 'rho_cmbS4', 'rho_cmbS3']
res_lbins = np.loadtxt(datadir + 'limber_spectra/cbb_res_ls.txt')

B_res = load_res(surveys)
# load B_res after del

ells_cmb = np.loadtxt(datadir + 'cmb_cl/ell.txt')

clbb_th = np.loadtxt(datadir + 'cmb_cl/bb.txt')
clbb_th *= 2. * np.pi / (ells_cmb.astype(float) *
                         (ells_cmb.astype(float) + 1.))

# PLOTS
fg = plt.figure(figsize=fig_dims)
# fg = plt.figure(figsize=(8, 10))

ax1 = fg.add_subplot(111)

# ax1.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)

# plt.plot(ell, clkappacib / np.sqrt(clkappa * (clcib + 225.)), linestyle='--')
plt.plot(
    res_lbins,
    B_res['cib'] / np.interp(res_lbins, ells_cmb, clbb_th),
    label='CIB')
plt.plot(
    res_lbins,
    B_res['des'] / np.interp(res_lbins, ells_cmb, clbb_th),
    label='DES')

plt.xlabel(r'$\ell$')
plt.ylabel(
    r' $\rho_{\ell} = \frac{C_{\ell}^{I\kappa}}{\sqrt{C_{\ell}^{II} C_{\ell}^{\kappa \kappa}  }  }$ '
)
plt.legend(loc='best')
plt.xlim(40, 1000)
plt.ylim(0., 1.)
# fg.tight_layout(pad=0.4)

# ============================================
# FINALLY SAVE
plt.savefig(
    image_dir + 'delens_eff.pdf',
    dpi=600,
    papertype='Letter',
    format='pdf',
    transparent=True)
plt.savefig(
    image_dir + 'delens_eff.png',
    dpi=600,
    papertype='Letter',
    format='png',
    transparent=True)
plt.clf()
