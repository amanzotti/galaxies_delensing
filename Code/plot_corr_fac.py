
from __future__ import division
# from __future__ import print_function
print "\n Python start up script from /home/manzotti/.py_startup.py.\n"
import scipy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
from itertools import cycle

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

from pprint import pprint
import __builtin__
import pyximport
pyximport.install(reload_support=True)

try:
    cmap = palettable.colorbrewer.get_map('RdBu', 'diverging', 11, reverse=True).mpl_colormap
    plt.rcParams['image.cmap'] = cmap

except Exception, e:
    print 'probably palettable not found'


datadir = '../Data/limber_spectra/'

image_dir = '../images/'


# Import data
clkappades = np.loadtxt(datadir + 'cldeskdes.txt')
clkappacib = np.loadtxt(datadir + 'clcibkdes.txt')
cldes = np.loadtxt(datadir + 'cldesdes.txt')
clcib = np.loadtxt(datadir + 'clcibdes.txt')
clkappa = np.loadtxt(datadir + 'clkdes.txt')
ell = np.loadtxt(datadir + 'ells_des.txt')

desi_ell = np.loadtxt(datadir + 'ells.txt')
cldesi = np.loadtxt(datadir + 'cldesi.txt')
clkappadesi = np.loadtxt(datadir + 'cldesik.txt')
desi_clkappa = np.loadtxt(datadir + 'clk.txt')
# combined rho coming from multiple_survey_delens.py on midway
rho_comb_des_cib = np.loadtxt(datadir + 'rho_comb_des_cib.txt')
rho_comb_des_cib_cmb= np.loadtxt(datadir + 'rho_comb_des_cib_cmb.txt')
rho_des = np.loadtxt(datadir + 'rho_des.txt')
rho_cib = np.loadtxt(datadir + 'rho_cib.txt')
rho_cmb = np.loadtxt(datadir + 'rho_cmb.txt')




# ADD NOISE
# using table 6 of 1309.0382v1
# note: you need the power spectrum in the right units to add the noise this way
# clcib+=225.


# DES NOISE Evaluation

degree_sq = 500
rad_sq = 500 * (np.pi / 180) ** 2
fsky = rad_sq / 4. / np.pi
n_gal = 3207184
nlgg = 1. / (3207184. / rad_sq) * np.ones_like(cldes)  # they mention N=2.1 10^-8 in Tommaso's paper


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
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

# PLOTS
fg = plt.figure(figsize=fig_dims)
# fg = plt.figure(figsize=(8, 10))

ax1 = fg.add_subplot(111)

ax1.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)

# plt.plot(ell, clkappacib / np.sqrt(clkappa * (clcib + 225.)), linestyle='--')
plt.plot(rho_cib[:,0], rho_cib[:,1],  label='CIB Hall model', linestyle = next(linecycler))
plt.plot(rho_des[:,0], rho_des[:,1],  label='D.E.S', linestyle = next(linecycler))
plt.plot(rho_cmb[:,0], rho_cmb[:,1],  label='CMB', linestyle = next(linecycler))

# plt.plot(ell, clkappades / np.sqrt(clkappa * (cldes + nlgg)), color='#377eb8', linestyle='--')
plt.plot( desi_ell, clkappadesi / np.sqrt(desi_clkappa * cldesi),  label='DESI', linestyle = next(linecycler))
plt.plot( rho_comb_des_cib[:,0], rho_comb_des_cib[:,1],   label='CIB+D.E.S', linestyle = next(linecycler))
plt.plot( rho_comb_des_cib_cmb[:,0], rho_comb_des_cib_cmb[:,1],  label='CIB+D.E.S+CMB', linestyle = next(linecycler))


plt.xlabel(r'\ell')
plt.ylabel(r' $\rho_{\ell} = \frac{C_{\ell}^{I\kappa}}{\sqrt{C_{\ell}^{II} C_{\ell}^{\kappa \kappa}  }  }$ ')
plt.legend(loc='best')
plt.xlim(40, 1000)
plt.ylim(0.,1.)
fg.tight_layout(pad=0.4)

# ============================================
# FINALLY SAVE
plt.savefig(image_dir + 'spectra.pdf', dpi=1200, papertype='Letter',
            format='pdf', transparent=True)

plt.clf()
