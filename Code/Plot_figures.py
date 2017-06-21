

import numpy as np
import matplotlib.pyplot as plt
import multiple_survey_delens


# In[14]:

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

plt.rcParams['axes.color_cycle'] = '#e41a1c,#377eb8,#4daf4a,#984ea3,#ff7f00,#ffff33,#a65628'
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


BB_contr = np.genfromtxt('../Data/BB_contribution.csv', delimiter=',')


plt.rcParams["figure.figsize"] = fig_dims

labels = ['wise', 'cib', 'des']
cmb = 'Planck'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)


# In[16]:

plt.plot(lbins, rho['cib'])
plt.plot(lbins, rho['des'])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.xlim(10, 1400)


# In[20]:

cib = np.loadtxt('/home/manzotti/galaxies_delensing/Data/limber_spectra/cl_cibcib_delens.txt')
kk = np.loadtxt('/home/manzotti/galaxies_delensing/Data/limber_spectra/cl_kk_delens.txt')
kcib = np.loadtxt('/home/manzotti/galaxies_delensing/Data/limber_spectra/cl_kcib_delens.txt')
# plt.plot(lbins, 2.4 * 1e3 * (lbins/2000.)** 0.53)
# alpha = 0.387
# lc = 162.9
# gamma = 0.168
# A_dust = 16.44 * 1e3
# D_dust = A_dust * ((100. / lbins)**alpha) / (1 + (lbins / lc)**2)**(gamma / 2.)
# plt.semilogy(lbins, D_dust, '.-')
# plt.loglog(lbins,cib)
plt.loglog(lbins, 100 * cib[100] * (lbins / lbins[100])**-4.6)
plt.loglog(lbins, np.array([3500. * (1. * l / 3000.)**(-1.25) for l in lbins]))
plt.xlim(10, 2000)


# In[21]:

fg = plt.figure(figsize=[10, 8])
ax1 = fg.add_subplot(111)

plt.plot(lbins, rho['cib']**2, label='CIB')
plt.plot(lbins, rho['wise']**2, label='WISE')
plt.plot(lbins, rho['des']**2, label='DES')
plt.plot(lbins, rho_cmb**2, label='Planck')
plt.plot(lbins, rho_gals**2, label='DES +CIB+WISE')
plt.plot(lbins, rho_comb**2, label='gals + Planck')
plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
         '--', alpha=0.6, linewidth=font_size / 14., color='k')
plt.plot(lbins, np.loadtxt('/home/manzotti/galaxies_delensing/Data/limber_spectra/rho_comb.txt')**2)
plt.legend(loc=0, ncol=2)
plt.title('2016 Scenario with Planck')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho^{2}$')
plt.xlim(10, 1400)
plt.ylim(0, 1.4)
plt.grid()
fg.tight_layout()


# In[22]:

fg = plt.figure(figsize=fig_dims)
ax1 = fg.add_subplot(111)

plt.plot(lbins, rho['cib'], label='CIB')
plt.plot(lbins, rho['wise'], label='WISE')
plt.plot(lbins, rho['des'], label='DES')
plt.plot(lbins, rho_cmb, label='Planck')
plt.plot(lbins, rho_gals, label='DES +CIB+WISE')
plt.plot(lbins, rho_comb, label='gals + Planck')
plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
         '--', alpha=0.6, linewidth=font_size / 14., color='k')

plt.plot()
plt.legend(loc=0, ncol=2)
plt.title('2016 Scenario with Planck')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.xlim(10, 1400)
plt.ylim(0, 1.15)
fg.tight_layout()

plt.savefig('../images/actual_scenario_planck.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/actual_scenario_planck.png')


# ## CMB current scenario

# In[23]:

labels = ['wise', 'cib', 'des']
cmb = 'now'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
np.savetxt('rho_cib.txt', rho['cib'])
np.savetxt('rho_des.txt', rho['des'])
np.savetxt('rho_cmb_current.txt', rho_cmb)
np.savetxt('rho_gals_current.txt', rho_gals)
np.savetxt('rho_comb_current.txt', rho_comb)
np.savetxt('lbins.txt', lbins)


# In[24]:

fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho['cib'], label='CIB')
plt.plot(lbins, rho['des'], label='DES')
plt.plot(lbins, rho_cmb, label='SPT Pol')
plt.plot(lbins, rho_gals, label='DES +CIB')
plt.plot(lbins, rho_comb, label='DES + CIB + SPTPol')
plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]), '--', alpha=0.6, color='k')
plt.legend(loc=0, ncol=2)
plt.ylim(0, 1.2)
plt.title('2016 Scenario')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.xlim(10, 1400)
fg.tight_layout()


plt.savefig('../images/actual_scenario.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/actual_scenario.png')


# ## CMB S3 scenario

# In[25]:

labels = ['wise', 'cib', 'desi', 'des']
cmb = 'S3'


# In[26]:

lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)


# In[27]:


np.savetxt('rho_cmb_S3.txt', rho_cmb)
np.savetxt('rho_gals_S3.txt', rho_gals)
np.savetxt('rho_comb_S3.txt', rho_comb)


# In[28]:

fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho['desi'], label='DESI')
# plt.plot(lbins,rho['cib'],label = 'CIB')
# plt.plot(lbins,rho['des'],label = 'DES')
plt.plot(lbins, rho_cmb, label='SPT 3G')
plt.plot(lbins, rho_gals, label='DES + CIB + DES')
plt.plot(lbins, rho_comb, label='DES + CIB + DESI + 3G')
plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]), '--', alpha=0.6, color='k')

plt.legend(loc=0, ncol=2)
plt.ylim(0, 1.3)
plt.xlim(10, 1400)

plt.title('2020 Scenario')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
fg.tight_layout()

plt.savefig('../images/S3_scenario.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/S3_scenario.png')


# ## CMB S4 scenario

# In[29]:

# labels = ['cib', 'desi', 'des']
labels = ['wise', 'euclid', 'lsst', 'ska10', 'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
np.savetxt('rho_cmb_S4.txt', rho_cmb)
np.savetxt('rho_gals_S4.txt', rho_gals)
np.savetxt('rho_comb_S4.txt', rho_comb)


# In[31]:

# plt.plot(lbins,rho['desi'],label = 'DESI')
# plt.plot(lbins,rho['cib'],label = 'CIB')
# plt.plot(lbins,rho['des'],label = 'DES')
fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho_cmb, label='CMB S4')
# plt.plot(lbins, rho_gals, label='Galaxies')
plt.plot(lbins, rho_comb, label='Galaxies + CMB S4')
plt.plot(lbins, rho['euclid'], label='Euclid')
plt.plot(lbins, rho['lsst'], label='LSST')
plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
         '--', alpha=0.6, linewidth=font_size / 14., color='k')

labels = ['ska10', 'ska01', 'ska5', 'ska1']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
plt.plot(lbins, rho_gals, label='SKA')


labels = ['wise', 'euclid', 'lsst', 'cib', 'desi', 'des']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
plt.plot(lbins, rho_gals, label='Gals no SKA')


# labels = ['ska10']
# cmb = 'S4'
# %run -i multiple_survey_delens.py
# plt.plot(lbins,rho_gals,label = 'SKA')
plt.legend(loc=0, ncol=2)
plt.ylim(0, 1.2)
plt.xlim(10, 1400)

plt.title('2024 Scenario')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
fg.tight_layout()

plt.savefig('../images/S4_scenario.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/S4_scenario.png')


# ## CMB Internal

# In[32]:

fg = plt.figure(figsize=fig_dims)
labels = ['cib']
cmb = 'Planck'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
plt.plot(lbins, rho_cmb, label='Planck')
plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
         '--', alpha=0.6, linewidth=font_size / 14., color='k')

cmb = 'now'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
plt.plot(lbins, rho_cmb, label='Current stage')

cmb = 'S3'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
plt.plot(lbins, rho_cmb, label='CMB S3')

cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)


plt.plot(lbins, rho_cmb, label='CMB S4')

plt.title('CMB internal reconstruction correlation')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.legend(loc=0, ncol=2)
plt.ylim(0, 1.2)
plt.xlim(10, 1400)

fg.tight_layout()

plt.savefig('../images/cmb_internal.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/cmb_internal.png')

sys.exit('stopping before B_res plots')

# ## Galaxies Alone

# In[ ]:


# # From Correlation coeff to B$_{res}$

# Make plots of B_Res for actual cmb cib des and combined and future galaxies and CMB

# In[20]:

get_ipython().magic('run -i rho_to_Bres.py')


# In[21]:

ell = np.loadtxt(datadir + 'limber_spectra/cbb_res_ls.txt')


# In[22]:

rho_names = ['rho_cib.txt', 'rho_des.txt', 'rho_cmb_current.txt', 'rho_gals_current.txt', 'rho_comb_current.txt', 'rho_cib.txt',
             'rho_cmb_S3.txt', 'rho_gals_S3.txt', 'rho_comb_S3.txt', 'rho_cmb_S4.txt', 'rho_gals_S4.txt', 'rho_comb_S4.txt']


# In[23]:

[plt.plot(ell, np.array(B_res3[i]) * 1e6, label=r'$' + rho_names[i].split('.txt')
          [0].split('rho_')[1] + '$') for i in np.arange(0, len(B_res3))]
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{res}~\times~10^6$')


# In[24]:

[plt.loglog(ell, B_res3[i], label=r'$' + rho_names[i].split('.txt')[0].split('rho_')[1] + '$')
 for i in np.arange(0, len(B_res3))]
plt.loglog(cl_len.ls, cl_len.clbb)
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{res}$')


# In[25]:

[plt.plot(ell, (np.interp(ell, cl_len.ls, cl_len.clbb) - B_res3[i]) / np.interp(ell, cl_len.ls, cl_len.clbb),
          label=r'$' + rho_names[i].split('.txt')[0].split('rho_')[1] + '$') for i in np.arange(0, len(B_res3))]
plt.legend()
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{res}$')
plt.title('Percentage of Residual power after delensing')


# ## Quantify R constrain

# In[26]:

nl = sl.spec.nl(4.5, 4, 2000)


# In[27]:

plt.loglog(cl_len.ls, cl_len.clbb)
plt.loglog(nl)
plt.ylim(1e-8, 1e-4)


# In[28]:

alpha = {}
for i in np.arange(0, len(B_res3)):
    alpha[rho_names[i]] = np.mean(cl_len.clbb[:150] + nl[:150]) / \
        np.mean(np.interp(np.arange(0, 150), ell, B_res3[i]) + nl[:150])
