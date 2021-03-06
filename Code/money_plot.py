import scipy as sp
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import subprocess
import pyximport
pyximport.install(reload_support=True)
import numpy as np
import scipy.integrate

import pickle as pk

from matplotlib import cm

# ============================================
# SIZE OF THE PICTURE
# ============================================
WIDTH = 246.0 * 2.  # the number latex spits out
# the fraction of the width you'd like the figure to occupy, 0.5 if double
# column cause
FACTOR = 0.99
fig_width_pt = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
golden_ratio = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
ratio = 0.5
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

font_size = 12.
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.labelsize'] = font_size * 1.3
plt.rcParams['axes.linewidth'] = font_size / 10.
plt.rcParams['axes.titlesize'] = font_size * 1.3
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size / 1.
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

colormap = plt.cm.Set1
# colormap = plt.cm.Vega10
# dictionary of colors
colors = {}
colors['CIB'] = colormap(0)
colors['DES'] = colormap(1)
colors['WISE'] = colormap(2)
colors['DESI'] = colormap(3)
colors['LSST'] = colormap(4)
colors['SKA'] = colormap(5)

plt.close()
plt.close()

# PLOTS
fg = plt.figure(figsize=fig_dims)
N = 4
ind = np.arange(N)  # the x locations for the groups
ind = np.array([0, 2, 4, 6])
width = 0.55  # the width of the bars: can also be len(x) sequence

# ==========================================
# ==========================================
plt.gca().set_prop_cycle(None)
# WISE
p1 = plt.barh(ind[0], -8, width, alpha=0.8, color=colors['WISE'])
# Planck
p2 = plt.barh(ind[0], -7, width, left=-8, alpha=0.8, color=colormap(6))
# DES
p2 = plt.barh(ind[0], -17, width, left=-8 - 7, alpha=0.8, color=colors['DES'])
# CIB
p2 = plt.barh(
    ind[0], -30, width, left=-8 - 7 - 17, alpha=0.8, color=colors['CIB'])
# LSS
p1 = plt.barh(ind[0] + 1.5 * width, 42, width, color=colormap(7), alpha=0.8)
# total
p2 = plt.barh(
    ind[0] + 1.5 * width,
    45 - 42,
    width,
    color=colormap(8),
    left=42,
    alpha=0.8)

# ==========================================
# ==========================================

# ==========================================
plt.gca().set_prop_cycle(None)
p1 = plt.barh(
    ind[1], -8., width * 1.1, left=0., alpha=0.8, color=colors['WISE'])
p2 = plt.barh(
    ind[1], -31., width * 1.1, left=-8., alpha=0.8, color=colormap(6))
p2 = plt.barh(
    ind[1], -17, width * 1.1, left=-8 - 31, alpha=0.8, color=colors['DES'])
p2 = plt.barh(
    ind[1],
    -30,
    width * 1.1,
    left=-8 - 31 - 17,
    alpha=0.8,
    color=colors['CIB'])

p1 = plt.barh(
    ind[1] + 1.5 * width, 42, width * 0.9, color=colormap(7), alpha=0.8)
p2 = plt.barh(
    ind[1] + 1.5 * width,
    55 - 42,
    width * 0.9,
    color=colormap(8),
    left=42,
    alpha=0.8)
# ==========================================

# ==========================================
# ==========================================
plt.gca().set_prop_cycle(None)
p1 = plt.barh(ind[2], -11., width * 1.1, alpha=0.8, color=colors['DESI'])
p2 = plt.barh(
    ind[2], -45., width * 1.1, left=-11., alpha=0.8, color=colormap(7))
p2 = plt.barh(
    ind[2], -56., width * 1.1, left=-11. - 45, alpha=0.8, color=colormap(6))

p1 = plt.barh(
    ind[2] + 1.5 * width, 45, width * 0.9, color=colormap(7), alpha=0.8)
p2 = plt.barh(
    ind[2] + 1.5 * width,
    68 - 45,
    width * 0.9,
    color=colormap(8),
    left=45,
    alpha=0.8)

# ==========================================
# ==========================================

# ==========================================
# ==========================================
plt.gca().set_prop_cycle(None)
p1 = plt.barh(ind[3], -51, width * 1.1, alpha=0.8, color=colors['LSST'])
p2 = plt.barh(
    ind[3], -53, width * 1.1, left=-51, alpha=0.8, color=colors['SKA'])
p2 = plt.barh(
    ind[3], -81, width * 1.1, left=-51 - 53, alpha=0.8, color=colormap(6))

p1 = plt.barh(
    ind[3] + 1.5 * width,
    59,
    width * 0.9,
    color=colormap(7),
    alpha=0.8,
    label='LSS')
p2 = plt.barh(
    ind[3] + 1.5 * width,
    86 - 59,
    width * 0.9,
    color=colormap(8),
    left=59,
    alpha=0.8,
    label='LSS +CMB')

# ==========================================
# ==========================================

plt.legend()
# plt.ylim(-0.2, 3.7)
plt.xlabel(r'Power removed (\%)')
plt.yticks(ind + 0.2, ('Planck', 'SPTpol', 'Stage-3', 'Stage-4'))
plt.xticks([0, 20, 40, 60, 80, 100])
plt.xlim(-200, 110)
# plt.legend((p1[0], p2[0]), ('Gals', 'CMB'))
plt.grid(color='k', ls='--', lw=0.6, alpha=0.6)
plt.gca().yaxis.grid(False)

plt.gca().set_axisbelow(True)

# start annotation

plt.text(
    -49,
    -0.2,
    'CIB',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold')

plt.text(
    -29,
    -0.18,
    'DES',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.8,
    fontweight='bold')

plt.text(
    -12,
    0.51,
    'Planck',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold',
    rotation=90.)

plt.text(
    -6,
    0.55,
    'WISE',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold',
    rotation=90.)

plt.text(
    -72,
    2 - 0.2,
    'CIB',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold')

plt.text(
    -52.5,
    2 - 0.18,
    'DES',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.8,
    fontweight='bold')

plt.text(
    -32,
    2 - 0.2,
    'SPTPol',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold')

plt.text(
    -6,
    2 + 0.55,
    'WISE',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold',
    rotation=90.)

plt.text(
    -88,
    4 - 0.2,
    'SPT3G',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold')

plt.text(
    -44,
    4 - 0.2,
    'LSS-S3',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold')

plt.text(
    -6.9,
    4 + 0.56,
    'DESI',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold',
    rotation=90.)

plt.text(
    -160,
    6 - 0.2,
    'CMB-S4',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold',
)

plt.text(
    -98,
    6 - 0.2,
    r'SKA (10 $\mu$Jy)',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold')

plt.text(
    -35,
    6 - 0.2,
    r'LSST',
    horizontalalignment='left',
    verticalalignment='bottom',
    fontsize=font_size / 1.5,
    fontweight='bold')

fg.tight_layout()

# ============================================
# FINALLY SAVE
plt.savefig('../images/' + 'errors_summary.pdf', papertype='Letter', dpi=800)
plt.savefig('../images/' + 'errors_summary.png', papertype='Letter', dpi=800)
# plt.clf()
# plt.close()
