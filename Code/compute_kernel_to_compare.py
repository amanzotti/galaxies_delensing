
'''
'''
import numpy as np
# import kappa_cmb_kernel as kappa_kernel
# import gals_kernel
# import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import imp
import pyximport
pyximport.install(reload_support=True)
import sys
sys.path.append('/home/manzotti/cosmosis/modules/limber/')
import gals_kernel
import kappa_gals_kernel
import kappa_cmb_kernel as kappa_kernel
import camb
from camb import model
import DESI

cib_hall = imp.load_source('cib_hall', '/home/manzotti/cosmosis/modules/limber/hall_CIB_kernel.py')

spectra_dir = '/home/manzotti/galaxies_delensing/Data/'


# SET UP CAMB
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars = camb.CAMBparams()
# This function sets up CosmoMC-like settings, with one massive neutrino
# and helium set using BBN consistency
pars.set_cosmology(H0=67.26, ombh2=0.02222, omch2=0.1199,
                   mnu=0.06, omk=0, tau=0.079)
pars.InitPower.set_params(ns=0.96, r=0., nt=0, pivot_tensor=0.01, As=2.1e-9)
pars.set_for_lmax(5000, lens_potential_accuracy=3)
pars.NonLinear = model.NonLinear_both

pars.AccurateBB = True
pars.OutputNormalization = False
pars.WantTensors = True
pars.DoLensing = True
pars.max_l_tensor = 3000
pars.max_eta_k_tensor = 3000.

pars.set_matter_power(redshifts=np.linspace(0., 13, 50), kmax=5.0)
results = camb.get_results(pars)

# P(z,k)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(
    minkh=1e-6, maxkh=5, have_power_spectra=False, npoints=250)
rbs = RectBivariateSpline(kh_nonlin, z_nonlin, pk_nonlin.T)
h = pars.H0 / 100.

# Distances
# =======================
# distance to last scattering surface
xlss = (results.conformal_time(0) - model.tau_maxvis.value) * h
# spline the redshift and the comoving distance
z = np.linspace(0, 15, 100)[::-1]
chispline = InterpolatedUnivariateSpline(
    np.linspace(0, 15, 100), results.comoving_radial_distance(np.linspace(0, 15, 100)) * h, ext=0)
hspline = InterpolatedUnivariateSpline(
    np.linspace(0, 15, 100), [results.hubble_parameter(z_vector) / pars.H0 / 3000. for z_vector in np.linspace(0, 15, 100)], ext=0)

# GROWTH

growth = InterpolatedUnivariateSpline(np.linspace(0, 15, 100), np.sqrt(
    (rbs(0.01, np.linspace(0, 15, 100)) / rbs(0.01, 0)))[0])

omega_m = pars.omegac
h0 = pars.H0 / 100.


dndz_filename = '/home/manzotti/cosmosis/modules/limber/' + \
    'data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt'
dndz_des = np.loadtxt(dndz_filename)
dndzfun = InterpolatedUnivariateSpline(dndz_des[:, 0], dndz_des[:, 1])
norm = dndzfun.integral(dndz_des[0, 0], dndz_des[-1, 0])
# normalize
dndzfun_des = InterpolatedUnivariateSpline(dndz_des[:, 0], dndz_des[:, 1] / norm)

dndz_filename = '/home/manzotti/cosmosis/modules/limber/' + 'data_input/DESI/DESI_dndz.txt'
dndz = np.loadtxt(dndz_filename)
dndz[:, 1] = np.sum(dndz[:, 1:], axis=1)
dndzfun = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1])
norm = dndzfun.integral(dndz[0, 0], dndz[-2, 0])
# normalize
dndzfun_desi = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1] / norm)


wise_dn_dz = np.loadtxt('/home/manzotti/galaxies_delensing/Code/wise_dn_dz.txt')
dndzwise = InterpolatedUnivariateSpline(wise_dn_dz[:, 0], wise_dn_dz[:, 1], k=3, ext='zeros')
norm = dndzwise.integral(0, 2)
dndzwise = InterpolatedUnivariateSpline(
    wise_dn_dz[:, 0], wise_dn_dz[:, 1] / norm, ext='zeros')
# Biased was measured equal to 1 in Feerraro et al. WISE ISW measureament
# by cross correlating with planck lensing
wise = gals_kernel.kern(wise_dn_dz[:, 0], dndzwise, hspline, omega_m, h0, b=1.)
zdist = np.linspace(0, 15, 100)[::-1]

nu = 353e9
zs = 2.
b = 1.
j2k = 1.e-6 / np.sqrt(83135.)  # for 353
lkern = kappa_kernel.kern(zdist, hspline, chispline, omega_m, h0, xlss)
cib = cib_hall.ssed_kern(
    h0, zdist, chispline, hspline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})
des = gals_kernel.kern(dndz_des[:, 0], dndzfun_des, hspline, omega_m, h0)
desi = gals_kernel.kern(dndz[:, 0], dndzfun_desi, hspline, omega_m, h0)

des_weak = kappa_gals_kernel.kern(dndz_des[:, 0], dndzfun_des, chispline, hspline, omega_m, h0)

l = 100


z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    # w_kappa[i] = 1. / x * lkern.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_kappa[i] = lkern.w_lxz(l, x, z)

# LSST
z_lsst = np.linspace(0.01, 10, 200)
dndzlsst = gals_kernel.dNdZ_parametric_LSST(z_lsst)
dndzfun = interp1d(z_lsst, dndzlsst)
norm = scipy.integrate.quad(dndzfun, 0.01, z_lsst[-1], limit=100, epsrel=1.49e-03)[0]
# used the same bias model of euclid. Find something better
dndzlsst = InterpolatedUnivariateSpline(z_lsst, dndzlsst / norm * 1. * np.sqrt(1. + z_lsst))
lsst = gals_kernel.kern(z_lsst, dndzlsst, hspline, omega_m, h0, b=1.)
w_lsst = np.zeros_like(z_lsst)
for i, z in enumerate(z_lsst):

    x = chispline(z)
    # w_kappa[i] = 1. / x * lkern.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_lsst[i] = lsst.w_lxz(l, x, z)

z_ska = np.linspace(0, 13, 500)
w_ska = np.zeros_like(z_ska)
dndzska10 = gals_kernel.dNdZ_parametric_SKA_10mujk(z_ska)
dndzska1 = gals_kernel.dNdZ_parametric_SKA_1mujk(z_ska)
dndzska5 = gals_kernel.dNdZ_parametric_SKA_5mujk(z_ska)
dndzska01 = gals_kernel.dNdZ_parametric_SKA_01mujk(z_ska)

# ===
dndzfun = interp1d(z_ska, dndzska10)
norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]
# print(norm)
# normalize
# dndzska01 = InterpolatedUnivariateSpline(
#     z_ska, dndzska01 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=0.1))

dndzska10 = InterpolatedUnivariateSpline(
    z_ska, dndzska10 / norm)


ska01 = gals_kernel.kern(z_ska, dndzska10, hspline, omega_m, h0, b=1.)
# ska10 = gals_kernel.kern(z_ska, dndzska10, hspline, omega_m, h0, b=1.)


for i, z in enumerate(z_ska):

    x = chispline(z)
    # w_kappa[i] = 1. / x * lkern.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_ska[i] = ska01.w_lxz(l, x, z)


z_kappa_gal = np.linspace(0, des_weak.zmax, 500)
w_kappa_gal = np.zeros_like(z_kappa_gal)
w_structure = np.zeros_like(z_kappa_gal)

for i, z in enumerate(z_kappa_gal):

    x = chispline(z)
    # w_kappa_gal[i] = 1. / x * des_weak.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_structure[i] = np.sqrt(rbs.ev((l + 0.5) / x, z))  # des_weak.w_lxz(l, x, z)

z_cib = np.linspace(0, 13., 500)
w_cib = np.zeros_like(z_cib)
for i, z in enumerate(z_cib):

    x = chispline(z)
    # w_cib[i] = 1. / x * cib.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_cib[i] = cib.w_lxz(l, x, z)

# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    # w_des[i] = 1. / x * des.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_des[i] = des.w_lxz(l, x, z)


z_wise = np.linspace(0, 1.5, 500)
w_wise = np.zeros_like(z_wise)
for i, z in enumerate(z_wise):

    x = chispline(z)
    # w_des[i] = 1. / x * des.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_wise[i] = wise.w_lxz(l, x, z)


z_desi = np.linspace(0.0, 2, 500)
w_desi = np.zeros_like(z_desi)

desi = gals_kernel.kern(np.linspace(0, 2, 100), DESI.DESISpline_normalized,
                        hspline, pars.omegac, h, b=1.17)

for i, z in enumerate(z_desi):

    x = chispline(z)
    # w_desi[i] = 1. / x * desi.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))
    w_desi[i] = desi.w_lxz(l, x, z)

datadir = '../Data/limber_spectra/'

np.savetxt(datadir + 'desi_kernel_l{}.txt'.format(l), np.vstack((z_desi, w_desi)).T)
np.savetxt(datadir + 'des_kernel_l{}.txt'.format(l), np.vstack((z_des, w_des)).T)
np.savetxt(datadir + 'cib_kernel_l{}.txt'.format(l), np.vstack((z_cib, w_cib)).T)
np.savetxt(datadir + 'kappa_kernel_l{}.txt'.format(l),
           np.vstack((z_kappa, w_kappa)).T)

np.savetxt(datadir + 'ska_kernel_l{}.txt'.format(l),
           np.vstack((z_ska, w_ska)).T)

np.savetxt(datadir + 'lsst_kernel_l{}.txt'.format(l),
           np.vstack((z_lsst, w_lsst)).T)

np.savetxt(datadir + 'weak_kernel_l{}.txt'.format(l),
           np.vstack((z_kappa_gal, w_kappa_gal)).T)
