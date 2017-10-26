'''
Here we plot all the correlations plot (rho) for different lensing tracers.

'''


import numpy as np
import matplotlib.pyplot as plt
import multiple_survey_delens
import sys
import rho_to_Bres
import configparser as ConfigParser
import camb
from camb import initialpower
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
plt.rcParams['legend.fontsize'] = font_size / 1.2
plt.rcParams['xtick.labelsize'] = font_size / 1.2
plt.rcParams['ytick.labelsize'] = font_size / 1.2
plt.rcParams['xtick.major.width'] = font_size / 10.
plt.rcParams['ytick.major.width'] = font_size / 10.
# plt.rcParams['xtick.labelsize'] = font_size / 1.2
# plt.rcParams['ytick.labelsize'] = font_size / 1.2

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


# =============================================

BB_contr = np.genfromtxt('../Data/BB_contribution.csv', delimiter=',')
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


plt.rcParams["figure.figsize"] = fig_dims

labels = ['wise', 'cib', 'des_bin0', 'des_bin1', 'des_bin2', 'des_bin3']
cmb = 'Planck'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

# Because we want to plot the rho for the tomographic binned survey.
# you will see this for all the survey here. Is a bit pedantic and should be made
# more beautiful
labels = ['des_bin0', 'des_bin1', 'des_bin2', 'des_bin3']
cmb = 'Planck'
lbins, _, _, rho_gals_des, _ = multiple_survey_delens.main(labels, cmb)


fg = plt.figure(figsize=fig_dims)
ax1 = fg.add_subplot(111)

plt.plot(lbins, rho['cib'], color=colors['CIB'], label='CIB')
plt.plot(lbins, rho['wise'], color=colors['WISE'], label='WISE')
plt.plot(lbins, rho_gals_des, color=colors['DES'], label='DES')
plt.plot(lbins, rho_cmb, label='Planck', color=colormap(6))
plt.plot(lbins, rho_gals, label='DES+CIB+WISE (LSS S2)', color=colormap(7))
plt.plot(lbins, rho_comb, label='LSS S2 + Planck', color=colormap(8))
plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
         '--', alpha=0.6, linewidth=font_size / 14., color='k')

plt.plot()
plt.legend(loc=0, ncol=2)
plt.title('Current Scenario with Planck')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.xlim(10, 1400)
plt.ylim(0, 1.15)
fg.tight_layout()

plt.savefig('../images/actual_scenario_planck.pdf',
            dpi=600, papertype='Letter')
plt.savefig('../images/actual_scenario_planck.png')


# ## CMB current scenario

# In[23]:

labels = ['wise', 'cib', 'des_bin0', 'des_bin1', 'des_bin2', 'des_bin3']
cmb = 'now'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)


labels = ['des_bin0', 'des_bin1', 'des_bin2', 'des_bin3']
cmb = 'now'
lbins, _, _, rho_gals_des, _ = multiple_survey_delens.main(labels, cmb)


# In[24]:
plt.close()

fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho['cib'], color=colors['CIB'], label='CIB')
plt.plot(lbins, rho_gals_des, color=colors['DES'], label='DES')
plt.plot(lbins, rho['wise'], color=colors['WISE'], label='WISE')
plt.plot(lbins, rho_cmb, label='SPT Pol', color=colormap(6))
plt.plot(lbins, rho_gals, label='DES+CIB', color=colormap(7))
plt.plot(lbins, rho_comb, label='LSS S2 +  SPTPol', color=colormap(8))
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]), '--', alpha=0.6, color='k')
plt.legend(loc=0, ncol=2)
plt.ylim(0, 1.2)
plt.title('Current Scenario with SPTPol')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.xlim(10, 1400)
fg.tight_layout()


plt.savefig('../images/actual_scenario.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/actual_scenario.png')

# ## CMB S3 scenario

labels = ['wise', 'cib', 'des_bin0', 'des_bin1', 'des_bin2',
          'des_bin3', 'desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']
cmb = 'S3'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

labels = ['desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']
cmb = 'S3'
lbins, _, _, rho_gals_desi, _ = multiple_survey_delens.main(labels, cmb)

plt.close()

fg = plt.figure(figsize=fig_dims)

plt.plot(lbins, rho_gals_desi, color=colors['DESI'], label='DESI')
plt.plot(lbins, rho_cmb, label='SPT 3G', color=colormap(6))
plt.plot(lbins, rho_gals, label='LSS-S2+DESI (LSS-S3)', color=colormap(7))
plt.plot(lbins, rho_comb, label='LSS-S3+SPT 3G', color=colormap(8))
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]), '--', alpha=0.6, color='k')

plt.legend(loc=0, ncol=2)
plt.ylim(0, 1.3)
plt.xlim(10, 1400)

plt.title('Stage-3 Scenario')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
fg.tight_layout()

plt.savefig('../images/S3_scenario.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/S3_scenario.png')


# ## CMB S4 scenario

# In[29]: 'ska10', 'ska01', 'ska5', 'ska1'

labels = ['wise', 'ska10', 'cib', 'des_bin0', 'des_bin1', 'des_bin2', 'des_bin3', 'lsst_bin0', 'lsst_bin1', 'lsst_bin2',
          'lsst_bin3', 'lsst_bin4', 'lsst_bin5', 'lsst_bin6', 'lsst_bin7', 'lsst_bin8', 'lsst_bin9', 'desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

plt.close()
fg = plt.figure(figsize=fig_dims)
plt.plot(lbins, rho_cmb, label='CMB S4', color=colormap(6))
# plt.plot(lbins, rho_gals, label='Galaxies')
plt.plot(lbins, rho_comb, label=r'LSS-S4 + SKA' +
         '\n' + '+ CMB S4', color=colormap(8))
# plt.plot(lbins, rho['euclid'], label='Euclid')
labels = ['lsst_bin0', 'lsst_bin1', 'lsst_bin2',
          'lsst_bin3', 'lsst_bin4', 'lsst_bin5', 'lsst_bin6', 'lsst_bin7', 'lsst_bin8', 'lsst_bin9']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)
plt.plot(lbins, rho_gals, color=colors['LSST'], label='LSST')
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
#          '--', alpha=0.6, linewidth=font_size / 14., color='k')

labels = ['ska10']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)
plt.plot(lbins, rho_gals, color=colors['SKA'], label=r'SKA $10\mu$Jy')


labels = ['wise', 'cib', 'des_bin0', 'des_bin1', 'des_bin2', 'des_bin3', 'lsst_bin0', 'lsst_bin1', 'lsst_bin2',
          'lsst_bin3', 'lsst_bin4', 'lsst_bin5', 'lsst_bin6', 'lsst_bin7', 'lsst_bin8', 'lsst_bin9', 'desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)
plt.plot(lbins, rho['ska10'], label='LSS-S3+LSST(LSS-S4)', color=colormap(7))

plt.legend(loc=0, ncol=2)
plt.ylim(0.45, 1.25)
plt.xlim(10, 1400)

plt.title('Stage-4 Scenario')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
fg.tight_layout()

plt.savefig('../images/S4_scenario_ska10.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/S4_scenario_ska10.png')

sys.exit()

# In[29]: 'ska10', 'ska01', 'ska5', 'ska1'

labels = ['wise', 'ska01', 'cib', 'des_bin0', 'des_bin1', 'des_bin2', 'des_bin3', 'lsst_bin0', 'lsst_bin1', 'lsst_bin2',
          'lsst_bin3', 'lsst_bin4', 'lsst_bin5', 'lsst_bin6', 'lsst_bin7', 'lsst_bin8', 'lsst_bin9', 'desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

plt.close()
fg = plt.figure(figsize=fig_dims)
# plt.plot(lbins, rho_gals, label='Galaxies')
plt.plot(lbins, rho_comb, label='LSS-S4 + CMB S4', color=colormap(8))

# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
#          '--', alpha=0.6, linewidth=font_size / 14., color='k')

labels = ['ska01']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)
plt.plot(lbins, rho_gals, color=colors['SKA'], label=r'SKA $0.1\mu$Jy')
plt.plot(lbins, rho_cmb, label='CMB S4', color=colormap(6))

plt.legend(loc=0, ncol=2)
plt.ylim(0.5, 1.2)
plt.xlim(10, 1400)

plt.title('Stage-4 Scenario')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
fg.tight_layout()

plt.savefig('../images/S4_scenario_ska01.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/S4_scenario_ska01.png')


# sys.exit()


labels = ['lsst_bin0', 'lsst_bin1', 'lsst_bin2', 'lsst_bin3', 'lsst_bin4',
          'lsst_bin5', 'lsst_bin6', 'lsst_bin7', 'lsst_bin8', 'lsst_bin9']
cmb = 'S4'
lbins, rho, rho_comb, rho_gals_lsst, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

labels = ['lsst']
cmb = 'S4'
lbins, rho_lsst, rho_comb_lsst, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

labels = ['des_bin0', 'des_bin1', 'des_bin2', 'des_bin3']
cmb = 'Planck'
lbins, rho, rho_comb, rho_gals_des, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

labels = ['des']
cmb = 'Planck'
lbins, rho_des, rho_comb_des, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)

labels = ['desi_bin0', 'desi_bin1', 'desi_bin2', 'desi_bin3']
cmb = 'Planck'
lbins, rho, _, rho_gals_desi, _ = multiple_survey_delens.main(labels, cmb)

labels = ['desi']
cmb = 'Planck'
lbins, rho_desi, _, _, _ = multiple_survey_delens.main(labels, cmb)


plt.close()
fg = plt.figure(figsize=fig_dims)
plt.plot(lbins, rho_lsst['lsst'], label='LSST', color=colors['LSST'])
plt.plot(lbins, rho_gals_lsst, linestyle='-.', color=colors['LSST'])
plt.plot(lbins, rho_des['des'], label='DES', color=colors['DES'],)
plt.plot(lbins, rho_gals_des, linestyle='-.', color=colors['DES'])
plt.plot(lbins, rho_desi['desi'], label='DESI', color=colors['DESI'])
plt.plot(lbins, rho_gals_desi, linestyle='-.', color=colors['DESI'])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\rho$')
plt.legend(loc=0, ncol=1)
plt.ylim(0.2, 1.)
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
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)
plt.plot(lbins, rho_cmb, label='Planck')
# plt.plot(BB_contr[:, 0], BB_contr[:, 1] - np.min(BB_contr[:, 1]),
#          '--', alpha=0.6, linewidth=font_size / 14., color='k')

cmb = 'now'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)
plt.plot(lbins, rho_cmb, label='Current stage')

cmb = 'S3'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)
plt.plot(lbins, rho_cmb, label='CMB S3')

cmb = 'S4'
lbins, rho, rho_comb, rho_gals, rho_cmb = multiple_survey_delens.main(
    labels, cmb)


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
sys.exit()
############################################
############################################
#  THIS WILL PLOT RESIDUALS FIGURES!!!!!
############################################
############################################

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

# plt.clf()
# plt.close()
# fg = plt.figure(figsize=fig_dims)
# plt_func = plt.semilogx
# # compare with noise at 5 muk arcm
# # ell * (ell + 1.) / 2. / np.pi
# # plt_func(ell, ell * np.array(B_res3[0]), label=r'$C^{BB}_{\ell}^{\rm{lens}}$')
# plt_func(ell, ell * np.array(B_res3[1]) * 1e3,
#          label=r'$C^{BB^{\rm{res}}}_{\ell}}$', linewidth=font_size / 12.5, alpha=0.8)
# plt_func(ell, ell * np.array(B_res3[-1]) * 1e3,
#          label=r'$C^{BB^{\rm{res}}}_{\ell}}$', linewidth=font_size / 12.5, alpha=0.8)

# plt_func(clbb_tens(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1) *
#          np.pi * 2. * 1e3, label=r'$C^{BB^{\rm{tens}}}_{\ell}, ~ r=0.01$', linestyle='--', linewidth=font_size / 7.5, alpha=0.8)
# plt_func(clbb(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1) *
#          np.pi * 2. * 1e3, label=r'$C^{BB^{\rm{tot}}}_{\ell}$', linewidth=font_size / 12.5)
# fact = np.arange(0, 4001)
# plt.fill_between(np.arange(0, 4001), fact * nl(1, 1, 4000) * 1e3,
#                  fact * nl(9, 1, 4000) * 1e3, alpha=0.2, label='noise')
# plt.ylim(0., 0.00075 * 1e3)
# plt.xlim(10, 2000)
# plt.xlabel(r'$\ell$')
# plt.ylabel(r'$\ell C^{BB}_{\ell}  [ 10^{-3} \mu K^{2} ]$')
# plt.legend(loc=0)
# fg.tight_layout()
# plt.text(30, 0.25, 'SPTPol', rotation=50, va='bottom', ha='left', fontsize=font_size / 2.)
# plt.text(800, 0.03, 'CMB S4', rotation=15, va='bottom', ha='left', fontsize=font_size / 2.)

# plt.savefig('../images/BB_res.pdf', dpi=600, papertype='Letter')
# plt.savefig('../images/BB_res.png')


plt.clf()
plt.close()
fg = plt.figure(figsize=fig_dims)
plt_func = plt.loglog
# compare with noise at 5 muk arcm
# ell * (ell + 1.) / 2. / np.pi

plt_func(clbb_tens(r=0.01, lmax=3000),
         label=r'$C^{BB^{\rm{tens}}}_{\ell} ~ _{r=0.01}$', linestyle='--')

plt_func(clbb(r=0.0, lmax=3000),
         label=r'$C^{BB^{\rm{lens}}}_{\ell}$', linewidth=font_size / 12.5, alpha=0.8)

plt_func(ell, ell * (ell + 1) * np.array(B_res3[1]) /
         2. / np.pi, label=r'$C^{BB^{\rm{res}}}_{\ell} (\rm{CIB})$', linewidth=font_size / 12.5, alpha=0.8)
plt_func(ell, ell * (ell + 1) * np.array(B_res3[-1]) /
         2. / np.pi, label=r'$C^{BB^{\rm{res}}}_{\ell} (\rm{S4})$', linewidth=font_size / 12.5, alpha=0.8)

# plt_func(clbb(r=0.01, lmax=3000), label=r'$C^{BB^{\rm{tot}}}_{\ell}$')
fact = np.arange(0, 4001) * (np.arange(0, 4001) + 1.) / 2. / np.pi
plt.plot(np.arange(0, 4001), fact * nl(9, 1, 4000), color='black', linestyle='--',
         linewidth=font_size / 14.5, alpha=0.7)
plt.plot(np.arange(0, 4001), fact * nl(1, 1, 4000), color='black', linestyle='--',
         linewidth=font_size / 14.5, alpha=0.7)
plt.ylim(1e-5, 1.5e-1)
plt.xlim(10, 1700)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C^{BB}_{\ell}/ 2 \pi [ \mu K^{2} ]$')
plt.legend(loc=0)
plt.text(100, 2.e-2, 'SPTPol Noise', rotation=40,
         va='bottom', ha='left', fontsize=font_size / 1.7)
plt.text(600, 2.8e-3, 'CMB S4 Noise', rotation=40,
         va='bottom', ha='left', fontsize=font_size / 1.7)

fg.tight_layout()

plt.savefig('../images/BB_res_ell2.pdf', dpi=600, papertype='Letter')
plt.savefig('../images/BB_res_ell2.png')


# plt.clf()
# plt.close()
# fg = plt.figure(figsize=fig_dims)
# plt_func = plt.loglog
# # compare with noise at 5 muk arcm
# # ell * (ell + 1.) / 2. / np.pi
# plt_func(ell, np.array(
#     B_res3[1]), label=r'$C^{BB}_{\ell}^{\rm{lens}}$', linewidth=font_size / 12.5, alpha=0.8)
# plt_func(ell, np.array(
#     B_res3[-1]), label=r'$C^{BB^{\rm{res}}}_{\ell}}$', linewidth=font_size / 12.5, alpha=0.8)
# plt_func(clbb_tens(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1)**2 *
#          np.pi * 2., label=r'$C^{BB^{\rm{tens}}}_{\ell}, ~ r=0.01$', linestyle='--',)

# plt_func(clbb(r=0.01, lmax=3000) / (np.arange(0, 3001) + 1) **
# 2 * np.pi * 2., label=r'$C^{BB^{\rm{tot}}}_{\ell}$', linewidth=font_size
# / 12.5, alpha=0.8)

# plt.ylabel(r'$C^{BB}_{\ell} [ \mu K^{2} ]$')

# # plt_func(clbb(r=0.01, lmax=3000), label=r'$C^{BB^{\rm{tot}}}_{\ell}$')
# fact = 1.
# plt.fill_between(np.arange(0, 4001), fact * nl(1, 1, 4000),
#                  fact * nl(9, 1, 4000), alpha=0.2, label='noise')
# # plt.ylim(1e-5, 4e-1)
# plt.xlim(10, 2000)
# plt.xlabel(r'$\ell$')
# # plt.ylabel(r'$C^{BB}_{\ell} [ 10^{-3} \mu K^{2} ]$')
# plt.legend(loc=0)
# plt.ylim(3e-8, 1e-5)

# # plt.text(30, 0.25, 'SPTPol', rotation=0, va='bottom', ha='left', fontsize=font_size / 2.)
# # plt.text(800, 0.03, 'CMB S4', rotation=0, va='bottom', ha='left', fontsize=font_size / 2.)
# fg.tight_layout()


# plt.savefig('../images/BB_res_ell0.pdf', dpi=600, papertype='Letter')
# plt.savefig('../images/BB_res_ell0.png')
