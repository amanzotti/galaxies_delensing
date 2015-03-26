
from __future__ import division
# from __future__ import print_function
print "\n Python start up script from /home/manzotti/.py_startup.py.\n"
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


datadir = '/Users/alessandromanzotti/Work/Software/cosmosis_new/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/'

image_dir = '../images/'


# Import data
des_dndz=np.loadtxt("/Users/alessandromanzotti/Work/Software/cosmosis_new/cosmosis-standard-library/structure/PS_limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt")



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
plt.rcParams['axes.titlesize'] = font_size*1.3
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


# PLOTS
fg = plt.figure(figsize=fig_dims)
# fg = plt.figure(figsize=(8, 10))

ax1 = fg.add_subplot(111)

plt.plot(des_dndz[:,0], des_dndz[:,1],label='D.E.S')

plt.xlabel(r'z')
plt.ylabel(r' $dN/dz$ ')
plt.legend(loc='best')
plt.xlim(0, 2)
fg.tight_layout(pad=0.4)

# ============================================
# FINALLY SAVE
plt.savefig(image_dir + 'dndz.pdf', dpi=1200, papertype='Letter',
            format='pdf', transparent=True)

plt.clf()
