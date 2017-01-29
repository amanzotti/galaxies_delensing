import numpy as np
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import ConfigParser


# cosmosis_dir = '/home/manzotti/my_version_cosmosis/'
# inifile = 'des_cib.ini'
# n_slices = 13


# Config_ini = ConfigParser.ConfigParser()
# values = ConfigParser.ConfigParser()
# Config_ini.read(inifile)
# values_file = Config_ini.get('pipeline', 'values')
# output_dir = Config_ini.get('test', 'save_dir')

# values.read(cosmosis_dir + values_file)


# lbins = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/ells_des.txt')


# ============================================================
# FIRST EXERCISE

# JUST GET RHO FOR DIFFERENT Z

# ============================================================

# cgk = np.zeros((n_slices, np.size(lbins)))
# cgg = np.zeros((n_slices, np.size(lbins)))
# rho = np.zeros((n_slices, np.size(lbins)))
# ckk = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clk.txt')

# for i, z_bin in enumerate(np.arange(1, n_slices + 1)):

#     cgk[i, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clgk_' + str(z_bin) + '.txt')
#     cgg[i, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clg_' + str(z_bin) + '.txt')

# rho = cgk/np.sqrt(cgg*ckk)

# fig, ax = plt.subplots(figsize=(6,6))
# cax = ax.matshow(rho)
# cbar = fig.colorbar(cax)
# ax.set_aspect('auto')
# plt.savefig('try2.pdf')


# ============================================================
# THIRD EXERCISE
# ============================================================


# cgk = np.zeros((n_slices, np.size(lbins)))
# cgg = np.zeros((n_slices, n_slices, np.size(lbins)))
# ckk = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clk.txt')

# for i in np.arange(0, n_slices ):
#     cgk[i,:] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clgk_' + str(i+1) + '.txt')

#     for j in np.arange(i, n_slices):
#         cgg[i, j, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clg_' + str(i+1) + '.txt')
#         cgg[j,i, :] =cgg[i,j ,:]

# cgk[i,j :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clgk_' + str(z_bin1)+str(z_bin2) + '.txt')
# cgg[i, j :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clg_' + str(z_bin) +str(z_bin2)  + '.txt')


# rho = cgk/np.sqrt(cgg*ckk)

# fig, ax = plt.subplots(figsize=(6,6))
# cax = ax.matshow(rho)
# cbar = fig.colorbar(cax)
# ax.set_aspect('auto')
# plt.savefig('try2.pdf')

'''
To add the cmb contribution just add a ckg with value ckk and a cgg with value cgg+noise
'''


# ============================================================
# DES CIB EXERCISE
# ============================================================


cosmosis_dir = '/home/manzotti/cosmosis/'
inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'
n_slices = 2


Config_ini = ConfigParser.ConfigParser()
values = ConfigParser.ConfigParser()
Config_ini.read(inifile)
values_file = Config_ini.get('pipeline', 'values')
output_dir = Config_ini.get('test', 'save_dir')

values.read(cosmosis_dir + values_file)


lbins = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/ells__delens.txt')


cgk = np.zeros((3, np.size(lbins)))
cgg = np.zeros((3, 3, np.size(lbins)))

ckk = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/clk_delens.txt')

noise_cl = np.loadtxt(
    './compute_chi2_lensingiswpowerspectrum/multipole_noisebias.txt')
# check the first ell is the same
# assert (ells_cmb[0] == noise_cl[0, 0])

noise_cl[:, 1] = noise_cl[:, 1] * noise_cl[:, 0] ** 4 / 4.  # because the power C_kk is l^4/4 C_phi
noise_fun = interp1d(noise_cl[:, 0], noise_cl[:, 1])
ckk_noise = np.zeros_like(ckk)
ckk_noise = noise_fun(lbins)

# initialize
rho_comb = np.zeros((np.size(lbins)))
rho_cib_des = np.zeros((np.size(lbins)))
rho_des = np.zeros((np.size(lbins)))
rho_cib = np.zeros((np.size(lbins)))

labels = ['des', 'cib']

for i in np.arange(0, n_slices):
    cgk[i, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + labels[i] + 'k_delens.txt')

    for j in np.arange(i, n_slices):
        cgg[i, j, :] = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + labels[i] + labels[j] + '_delens.txt')
        cgg[j, i, :] = cgg[i, j, :]


# add cmb lensing

cgk[2, :] = ckk
cgg[2, :, :] = cgk[:, :]
cgg[:, 2, :] = cgg[2, :, :]
# add noise
cgg[2, 2, :] = ckk + ckk_noise

# Single survey
rho_des = cgk[0, :] / np.sqrt(ckk[:] * cgg[0, 0, :])
rho_cib = cgk[1, :] / np.sqrt(ckk[:] * cgg[1, 1, :])
rho_cmb = cgk[2, :] / np.sqrt(ckk[:] * cgg[2, 2, :])
cl_ska10_k = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska10' + 'k_delens.txt')
cl_ska5_k = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska5' + 'k_delens.txt')
cl_ska1_k = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska1' + 'k_delens.txt')
cl_ska01_k = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska01' + 'k_delens.txt')

cl_ska10 = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska10ska10' + '_delens.txt')
cl_ska5 = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska5ska5' + '_delens.txt')
cl_ska1 = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska1ska1' + '_delens.txt')
cl_ska01 = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'ska01ska01' + '_delens.txt')

rho_ska10 = cl_ska10_k / np.sqrt(ckk[:] * cl_ska10)
rho_ska5 = cl_ska5_k / np.sqrt(ckk[:] * cl_ska5)
rho_ska1 = cl_ska1_k / np.sqrt(ckk[:] * cl_ska1)
rho_ska01 = cl_ska01_k / np.sqrt(ckk[:] * cl_ska01)

cl_lsst_k = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'lsst' + 'k_delens.txt')
cl_euclid_k = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'euclid' + 'k_delens.txt')

cl_euclid = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'euclideuclid' + '_delens.txt')
cl_lsst = np.loadtxt(cosmosis_dir + output_dir + '/limber_spectra/cl' + 'lsstlsst' + '_delens.txt')

rho_lsst = cl_lsst_k / np.sqrt(ckk[:] * cl_lsst)
rho_euclid = cl_euclid_k / np.sqrt(ckk[:] * cl_euclid)


for i, ell in enumerate(lbins):
    rho_comb[i] = np.dot(cgk[:, i], np.dot(np.linalg.inv(cgg[:, :, i]), cgk[:, i])) / ckk[i]
    rho_cib_des[i] = np.dot(cgk[:2, i], np.dot(np.linalg.inv(cgg[:2, :2, i]), cgk[:2, i])) / ckk[i]


np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_cib.txt', np.vstack((lbins, rho_cib)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_des.txt', np.vstack((lbins, rho_des)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_cmb.txt', np.vstack((lbins, rho_cmb)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_ska10.txt', np.vstack((lbins, rho_ska10)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_ska5.txt', np.vstack((lbins, rho_ska5)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_ska1.txt', np.vstack((lbins, rho_ska1)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_ska01.txt', np.vstack((lbins, rho_ska01)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_lsst.txt', np.vstack((lbins, rho_lsst)).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_euclid.txt', np.vstack((lbins, rho_euclid)).T)

np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_comb_des_cib_cmb.txt', np.vstack((lbins, np.sqrt(rho_comb))).T)
np.savetxt(cosmosis_dir + output_dir + '/limber_spectra/rho_comb_des_cib.txt', np.vstack((lbins, np.sqrt(rho_cib_des))).T)

# rho = cgk/np.sqrt(cgg*ckk)

# fig, ax = plt.subplots(figsize=(6,6))
# cax = ax.matshow(rho)
# cbar = fig.colorbar(cax)
# ax.set_aspect('auto')
# plt.savefig('try2.pdf')
