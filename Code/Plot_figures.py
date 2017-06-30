

import numpy as np
import matplotlib.pyplot as plt
import multiple_survey_delens

import rho_to_Bres
import configparser as ConfigParser
import camb
from camb import model, initialpower
try:
    import functools32
except ImportError:
    import functools as functools32
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

# plt.rcParams['axes.prop_cycles'] = '#e41a1c,#377eb8,#4daf4a,#984ea3,#ff7f00,#ffff33,#a65628'
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


# # labels = ['desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3', 'desi_bin4',
# #           'desi_bin5', 'desi_bin6', 'desi_bin7', 'desi_bin8', 'desi_bin9']
# # cmb = 'Planck'
# # lbins, rho, _, rho_gals_desi, _ = multiple_survey_delens.main(labels, cmb)

# labels = ['desi2_bin0', 'desi2_bin1']
# cmb = 'Planck'
# lbins, rho, _, rho_gals_desi2, _ = multiple_survey_delens.main(labels, cmb)

# labels = ['desi4_bin0', 'desi4_bin1','desi4_bin2','desi4_bin3']
# cmb = 'Planck'
# lbins, rho, _, rho_gals_desi4, _ = multiple_survey_delens.main(labels, cmb)

# labels = ['desi6_bin0', 'desi6_bin1','desi6_bin2','desi6_bin3']
# cmb = 'Planck'
# lbins, rho, _, rho_gals_desi6, _ = multiple_survey_delens.main(labels, cmb)


# labels = ['desi']
# cmb = 'Planck'
# lbins, rho_desi, _, _, _ = multiple_survey_delens.main(labels, cmb)


# plt.close()
# fg = plt.figure(figsize=fig_dims)
# plt.plot(lbins, rho_desi['desi'], label='DESI', color='#2ca02c')
# plt.plot(lbins, rho_gals_desi2, linestyle='-.', color='#2ca02c')
# plt.plot(lbins, rho_gals_desi4, linestyle='-.')
# plt.plot(lbins, rho_gals_desi6, linestyle='-.')

# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$\rho$')
# plt.legend(loc=0, ncol=1)
# plt.ylim(0.0, 1.)
# plt.xlim(10, 1400)
# fg.tight_layout()

# plt.savefig('../images/B_test.pdf', dpi=600, papertype='Letter')
# plt.savefig('../images/B_test.png')

# sys.exit()


# =============================================

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
plt.close()

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
plt.title('2017 Scenario with Planck')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho^{2}$')
plt.xlim(10, 1400)
plt.ylim(0, 1.4)
plt.grid()
fg.tight_layout()


# In[22]:
plt.close()

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
plt.title('2017 Scenario with Planck')
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
plt.close()

fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho['cib'], label='CIB')
plt.plot(lbins, rho['des'], label='DES')
plt.plot(lbins, rho_cmb, label='SPT Pol')
plt.plot(lbins, rho_gals, label='DES +CIB')
plt.plot(lbins, rho_comb, label='DES + CIB + SPTPol')
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]), '--', alpha=0.6, color='k')
plt.legend(loc=0, ncol=2)
plt.ylim(0, 1.2)
plt.title('2017 Scenario')
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
plt.close()

fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho['desi'], label='DESI')
# plt.plot(lbins,rho['cib'],label = 'CIB')
# plt.plot(lbins,rho['des'],label = 'DES')
plt.plot(lbins, rho_cmb, label='SPT 3G')
plt.plot(lbins, rho_gals, label='DES + CIB + DES')
plt.plot(lbins, rho_comb, label='DES + CIB + DESI + 3G')
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]), '--', alpha=0.6, color='k')

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
labels = ['wise', 'lsst', 'ska10', 'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
np.savetxt('rho_cmb_S4.txt', rho_cmb)
np.savetxt('rho_gals_S4.txt', rho_gals)
np.savetxt('rho_comb_S4.txt', rho_comb)


# In[31]:

# plt.plot(lbins,rho['desi'],label = 'DESI')
# plt.plot(lbins,rho['cib'],label = 'CIB')
# plt.plot(lbins,rho['des'],label = 'DES')
plt.close()

fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho_cmb, label='CMB S4')
# plt.plot(lbins, rho_gals, label='Galaxies')
plt.plot(lbins, rho_comb, label='Galaxies + CMB S4')
# plt.plot(lbins, rho['euclid'], label='Euclid')
plt.plot(lbins, rho['lsst'], label='LSST')
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
#          '--', alpha=0.6, linewidth=font_size / 14., color='k')

labels = ['ska10', 'ska01', 'ska5', 'ska1']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
plt.plot(lbins, rho_gals, label='SKA')


labels = ['wise', 'lsst', 'cib', 'desi', 'des']
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


labels = ['lsst_bin0', 'lsst_bin1', 'lsst_bin2', 'lsst_bin3', 'lsst_bin4',
          'lsst_bin5', 'lsst_bin6', 'lsst_bin7', 'lsst_bin8', 'lsst_bin9']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals_lsst, rho_cmb = multiple_survey_delens.main(labels, cmb)

labels = ['lsst']
cmb = 'S4'
lbins, rho_lsst, rho_comb_lsst, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)

labels = ['des_bin0', 'des_bin1', 'des_bin2']
cmb = 'Planck'
lbins, rho, rho_comb, rho_gals_des, rho_cmb = multiple_survey_delens.main(labels, cmb)

labels = ['des']
cmb = 'Planck'
lbins, rho_des, rho_comb_des, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)

labels = ['desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']
cmb = 'Planck'
lbins, rho, _, rho_gals_desi, _ = multiple_survey_delens.main(labels, cmb)

labels = ['desi']
cmb = 'Planck'
lbins, rho_desi, _, _, _ = multiple_survey_delens.main(labels, cmb)


plt.close()
fg = plt.figure(figsize=fig_dims)
plt.plot(lbins, rho_lsst['lsst'], label='LSST', color='#1f77b4')
plt.plot(lbins, rho_gals_lsst, linestyle='-.', color='#1f77b4')
plt.plot(lbins, rho_des['des'], label='DES', color='#ff7f0e')
plt.plot(lbins, rho_gals_des, linestyle='-.', color='#ff7f0e')
plt.plot(lbins, rho_desi['desi'], label='DESI', color='#2ca02c')
plt.plot(lbins, rho_gals_desi, linestyle='-.', color='#2ca02c')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.legend(loc=0, ncol=1)
plt.ylim(0.0, 1.)
plt.xlim(10, 1400)
fg.tight_layout()

plt.savefig('../images/B_res_bin.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/B_res_bin.png')


# ## CMB Internal


# In[32]:
plt.close()

fg = plt.figure(figsize=fig_dims)
labels = ['cib']
cmb = 'Planck'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(labels, cmb)
plt.plot(lbins, rho_cmb, label='Planck')
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
#          '--', alpha=0.6, linewidth=font_size / 14., color='k')

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

# sys.exit('stopping before B_res plots')

# ## Galaxies Alone

# In[ ]:


# # From Correlation coeff to B$_{res}$

# Make plots of B_Res for actual cmb cib des and combined and future galaxies and CMB

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

# In[21]:


pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars.set_cosmology(H0=70, ombh2=0.0226, omch2=0.112,
                   mnu=0.029, omk=0, tau=0.079)
pars.InitPower.set_params(ns=0.96, r=0., nt=0)
pars.set_for_lmax(5000, lens_potential_accuracy=3)
# pars.set_for_lmax?

pars.AccurateBB = True
pars.OutputNormalization = False
pars.WantTensors = True
pars.DoLensing = True
pars.max_l_tensor = 3000
pars.max_eta_k_tensor = 3000.

# print(pars) # if you want to test parasm
results = camb.get_results(pars)
tens_cl = results.get_tensor_cls(3000)


# ### Remember there is a 7.4e12 missing and CAMB always give you $ \ell (\ell +1)/2\pi  $

@functools32.lru_cache(maxsize=64)
def clbb(r=0.1, nt=None, lmax=3000):
    inflation_params = initialpower.InitialPowerParams()
    if nt is None:
        nt = -r / 8.
    inflation_params.set_params(As=2.1e-9, r=r, nt=nt)
    results.power_spectra_from_transfer(inflation_params)
    return results.get_total_cls(lmax)[:, 2] * 7.42835025e12


@functools32.lru_cache(maxsize=64)
def clbb_tens(r=0.1, nt=None, lmax=3000):
    inflation_params = initialpower.InitialPowerParams()
    if nt is None:
        nt = -r / 8.
    inflation_params.set_params(As=2.1e-9, r=r, nt=nt)
    results.power_spectra_from_transfer(inflation_params)
    return results.get_tensor_cls(lmax)[:, 2] * 7.42835025e12


clbb_tens.cache_clear()
clbb.cache_clear()


inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
Config_ini = ConfigParser.ConfigParser()
Config_ini.read(inifile)
output_dir = Config_ini.get('test', 'save_dir')

datadir = output_dir

clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
ells_cmb = np.loadtxt(datadir + 'cmb_cl/ell.txt')
ells_cmb = np.loadtxt(output_dir + 'cmb_cl/ell.txt')


# In[22]:

rho_names = ['test', 'rho_cib.txt', 'rho_des.txt', 'rho_gals.txt',
             'rho_wise.txt', 'rho_comb.txt', 'rho_cmb_' + cmb + '.txt']
# deep survey to delens or what is giving you E-mode
nle = nl(9, 1, lmax=ells_cmb[-1])[2:]
B_res3 = rho_to_Bres.main(rho_names, nle)
ell = np.loadtxt(datadir + 'limber_spectra/cbb_res_ls.txt')
# In[23]:
rho_names = ['rho_cib.txt', 'rho_des.txt', 'rho_cmb_current.txt', 'rho_gals_current.txt', 'rho_comb_current.txt', 'rho_cib.txt',
             'rho_cmb_S3.txt', 'rho_gals_S3.txt', 'rho_comb_S3.txt', 'rho_cmb_S4.txt', 'rho_gals_S4.txt', 'rho_comb_S4.txt']

plt.clf()
plt.close()
fg = plt.figure(figsize=fig_dims)
plt_func = plt.semilogx
# compare with noise at 5 muk arcm
# ell * (ell + 1.) / 2. / np.pi
# plt_func(ell, ell * np.array(B_res3[0]), label=r'$C^{BB}_{\ell}^{\rm{lens}}$')
plt_func(ell, ell * np.array(B_res3[1]) * 1e3,
         label=r'$C^{BB^{\rm{res}}}_{\ell}}$', linewidth=font_size / 12.5, alpha=0.8)
plt_func(ell, ell * np.array(B_res3[-1]) * 1e3,
         label=r'$C^{BB^{\rm{res}}}_{\ell}}$', linewidth=font_size / 12.5, alpha=0.8)

plt_func(clbb_tens(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1) *
         np.pi * 2. * 1e3, label=r'$C^{BB^{\rm{tens}}}_{\ell}, ~ r=0.01$', linestyle='--', linewidth=font_size / 7.5, alpha=0.8)
plt_func(clbb(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1) *
         np.pi * 2. * 1e3, label=r'$C^{BB^{\rm{tot}}}_{\ell}$', linewidth=font_size / 12.5)
fact = np.arange(0, 4001)
plt.fill_between(np.arange(0, 4001), fact * nl(1, 1, 4000) * 1e3,
                 fact * nl(9, 1, 4000) * 1e3, alpha=0.2, label='noise')
plt.ylim(0., 0.00075 * 1e3)
plt.xlim(10, 2000)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell C^{BB}_{\ell}  [ 10^{-3} \mu K^{2} ]$')
plt.legend(loc=0)
fg.tight_layout()
plt.text(30, 0.25, 'SPTPol', rotation=50, va='bottom', ha='left', fontsize=font_size / 2.)
plt.text(800, 0.03, 'CMB S4', rotation=15, va='bottom', ha='left', fontsize=font_size / 2.)

plt.savefig('../images/BB_res.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/BB_res.png')


plt.clf()
plt.close()
fg = plt.figure(figsize=fig_dims)
plt_func = plt.loglog
# compare with noise at 5 muk arcm
# ell * (ell + 1.) / 2. / np.pi
plt_func(ell, ell * (ell + 1) * np.array(B_res3[1]) /
         2. / np.pi, label=r'$C^{BB}_{\ell}^{\rm{lens}}$')
plt_func(ell, ell * (ell + 1) * np.array(B_res3[-1]) /
         2. / np.pi, label=r'$C^{BB^{\rm{res}}}_{\ell}}$')
plt_func(clbb_tens(r=0.01, lmax=3000), label=r'$C^{BB^{\rm{tens}}}_{\ell}, ~ r=0.01$')

plt_func(clbb(r=0.01, lmax=3000), label=r'$C^{BB^{\rm{tot}}}_{\ell}$')
# plt_func(clbb(r=0.01, lmax=3000), label=r'$C^{BB^{\rm{tot}}}_{\ell}$')
fact = np.arange(0, 4001) * (np.arange(0, 4001) + 1.) / 2. / np.pi
plt.fill_between(np.arange(0, 4001), fact * nl(1, 1, 4000),
                 fact * nl(9, 1, 4000), alpha=0.2, label='noise')
plt.ylim(1e-5, 4e-1)
plt.xlim(10, 2000)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C^{BB}_{\ell}/ 2 \pi [ 10^{-3} \mu K^{2} ]$')
plt.legend(loc=0)
fg.tight_layout()
# plt.text('SPTPol', rotation=45)
# plt.text('CMB S4', rotation=45)

plt.savefig('../images/BB_res_ell2.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/BB_res_ell2.png')


plt.clf()
plt.close()
fg = plt.figure(figsize=fig_dims)
plt_func = plt.loglog
# compare with noise at 5 muk arcm
# ell * (ell + 1.) / 2. / np.pi
plt_func(ell, np.array(B_res3[1]), label=r'$C^{BB}_{\ell}^{\rm{lens}}$')
plt_func(ell, np.array(B_res3[-1]), label=r'$C^{BB^{\rm{res}}}_{\ell}}$')
plt_func(clbb_tens(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1)**2 *
         np.pi * 2., label=r'$C^{BB^{\rm{tens}}}_{\ell}, ~ r=0.01$')

plt_func(clbb(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1) **
         2 * np.pi * 2., label=r'$C^{BB^{\rm{tot}}}_{\ell}$')


# plt_func(clbb(r=0.01, lmax=3000), label=r'$C^{BB^{\rm{tot}}}_{\ell}$')
fact = 1.
plt.fill_between(np.arange(0, 4001), fact * nl(1, 1, 4000),
                 fact * nl(9, 1, 4000), alpha=0.2, label='noise')
# plt.ylim(1e-5, 4e-1)
plt.xlim(10, 2000)
plt.xlabel(r'$\ell$')
# plt.ylabel(r'$C^{BB}_{\ell} [ 10^{-3} \mu K^{2} ]$')
plt.legend(loc=0, ncol=2)
plt.ylim(1e-8, 1e-5)
fg.tight_layout()
# plt.text('SPTPol', rotation=45)
# plt.text('CMB S4', rotation=45)

plt.savefig('../images/BB_res_ell0.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/BB_res_ell0.png')
