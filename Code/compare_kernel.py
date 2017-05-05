
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


cib_hall = imp.load_source('cib_hall', '/home/manzotti/cosmosis/modules/limber/hall_CIB_kernel.py')

spectra_dir = '/home/manzotti/galaxies_delensing/Data/'

zpower = np.loadtxt(spectra_dir + 'matter_power_nl/z.txt')
kpower = np.loadtxt(spectra_dir + 'matter_power_nl/k_h.txt')
powerarray = np.loadtxt(spectra_dir + 'matter_power_nl/p_k.txt')
powerarray = np.loadtxt(
    spectra_dir + 'matter_power_nl/p_k.txt').reshape([np.size(zpower), np.size(kpower)]).T
rbs = RectBivariateSpline(kpower, zpower, powerarray)

omega_m = 0.3

h0 = 0.7


h = np.loadtxt(spectra_dir + '/distances/h.txt')
tmp = h[::-1]
h = tmp

xlss = 13615.317054155654
zdist = np.loadtxt(spectra_dir + '/distances/z.txt')
tmp = zdist[::-1]
zdist = tmp

d_m = np.loadtxt(spectra_dir + '/distances/d_m.txt')
tmp = d_m[::-1]
d_m = tmp

d_m *= h0
h /= h0
xlss *= h0

chispline = InterpolatedUnivariateSpline(zdist[::-1], d_m[::-1])
z_chi_spline = InterpolatedUnivariateSpline(d_m[::-1], zdist[::-1])
hspline = InterpolatedUnivariateSpline(zdist[::-1], h[::-1])

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

l = 500
z_kappa = np.linspace(0, 13, 500)
w_kappa = np.zeros_like(z_kappa)
for i, z in enumerate(z_kappa):

    x = chispline(z)
    w_kappa[i] = 1. / x * lkern.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))



z_kappa_gal = np.linspace(0, des_weak.zmax, 500)
w_kappa_gal = np.zeros_like(z_kappa_gal)
for i, z in enumerate(z_kappa_gal):

    x = chispline(z)
    w_kappa_gal[i] = 1. / x * des_weak.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))


# plt.plot(z_kappa,w_kappa,label='cmb kappa')

# z_cib = np.linspace(0, 13., 500)
# w_cib = np.zeros_like(z_cib)
# for i, z in enumerate(z_cib):

#     x = chispline(z)
#     w_cib[i] = 1. / x * cib.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))

# plt.plot(z_cib,w_cib,label = 'cib')

z_des = np.linspace(0, 1.5, 500)
w_des = np.zeros_like(z_des)
for i, z in enumerate(z_des):

    x = chispline(z)
    w_des[i] = 1. / x * des.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))

z_desi = np.linspace(0.7, 1.8, 500)
w_desi = np.zeros_like(z_desi)
for i, z in enumerate(z_desi):

    x = chispline(z)
    w_desi[i] = 1. / x * desi.w_lxz(l, x, z) * np.sqrt(rbs.ev((l + 0.5) / x, z))

datadir = '../Data/limber_spectra/'

np.savetxt(datadir + 'desi_kernel_l{}.txt'.format(l), np.vstack((z_desi, w_desi)).T)
np.savetxt(datadir + 'des_kernel_l{}.txt'.format(l), np.vstack((z_des, w_des)).T)
# np.savetxt(spectra_dir + 'output/kernel/cib_kernel_l30.txt', np.vstack((z_cib, w_cib)).T)
np.savetxt(datadir + 'kappa_kernel_l{}.txt'.format(l),
           np.vstack((z_kappa, w_kappa)).T)
