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
fig_height_in = fig_width_in * ratio   # figure height in inches
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

plt.close()
plt.close()

# PLOTS
fg = plt.figure(figsize=fig_dims)
N = 4
ind = np.arange(N)    # the x locations for the groups
width = 0.3       # the width of the bars: can also be len(x) sequence

# ==========================================
# ==========================================
plt.gca().set_prop_cycle(None)
p1 = plt.barh(ind[0], -6, width, alpha=0.8)
p2 = plt.barh(ind[0], -8, width, left=-6, alpha=0.8)
p2 = plt.barh(ind[0], -14, width, left=-6 - 8, alpha=0.8)
p2 = plt.barh(ind[0], -27, width, left=-6 - 8 - 14, alpha=0.8)

p1 = plt.barh(ind[0] + 1.5 * width, 36, width, color='#7f7f7f', alpha=0.8)
p2 = plt.barh(ind[0] + 1.5 * width, 41 - 36, width, color='#bcbd22', left=36, alpha=0.8)

# ==========================================
# ==========================================


# ==========================================
plt.gca().set_prop_cycle(None)
p1 = plt.barh(ind[1], -6., width * 1.1, left=0.)
p2 = plt.barh(ind[1], -35., width * 1.1, left=-6., alpha=0.8)
p2 = plt.barh(ind[1], -14, width * 1.1, left=-6 - 35, alpha=0.8)
p2 = plt.barh(ind[1], -27, width * 1.1, left=-6 - 35 - 14, alpha=0.8)

p1 = plt.barh(ind[1] + 1.5 * width, 36, width * 0.9, color='#7f7f7f', alpha=0.8)
p2 = plt.barh(ind[1] + 1.5 * width, 56 - 36, width * 0.9, color='#bcbd22', left=36, alpha=0.8)
# ==========================================


# ==========================================
# ==========================================
plt.gca().set_prop_cycle(None)
p1 = plt.barh(ind[2], -10., width * 1.1, alpha=0.8)
p2 = plt.barh(ind[2], -40., width * 1.1, left=-10., alpha=0.8)
p2 = plt.barh(ind[2], -56., width * 1.1, left=-10. - 40., alpha=0.8)

p1 = plt.barh(ind[2] + 1.5 * width, 40, width * 0.9, color='#7f7f7f', alpha=0.8)
p2 = plt.barh(ind[2] + 1.5 * width, 69 - 40, width * 0.9, color='#bcbd22', left=40, alpha=0.8)

# ==========================================
# ==========================================

# ==========================================
# ==========================================
plt.gca().set_prop_cycle(None)
p1 = plt.barh(ind[3], -48, width * 1.1, alpha=0.8)
p2 = plt.barh(ind[3], -56, width * 1.1, left=-48, alpha=0.8)
p2 = plt.barh(ind[3], -84, width * 1.1, left=-48 - 56, alpha=0.8)

p1 = plt.barh(ind[3] + 1.5 * width, 60, width * 0.9, color='#7f7f7f', alpha=0.8, label='LSS')
p2 = plt.barh(ind[3] + 1.5 * width, 86 - 60, width * 0.9,
              color='#bcbd22', left=60, alpha=0.8, label='LSS +CMB')

# ==========================================
# ==========================================

plt.legend()
# plt.ylim(-0.2, 3.7)
plt.xlabel(r'Power removed (\%)')
plt.yticks(ind+0.2, ('Planck', 'SPTpol', 'Stage-3', 'Stage-4'))
plt.xticks([0, 20,  40,  60, 80,  100])
plt.xlim(-200, 110)
# plt.legend((p1[0], p2[0]), ('Gals', 'CMB'))
plt.grid(alpha=0.8, ls='--', lw=0.3)
plt.gca().yaxis.grid(False)

plt.gca().set_axisbelow(True)
fg.tight_layout()
# ============================================
# FINALLY SAVE
plt.savefig('../images/' + 'errors_summary.pdf', papertype='Letter')
plt.savefig('../images/' + 'errors_summary.png', papertype='Letter')
plt.clf()
plt.close()
