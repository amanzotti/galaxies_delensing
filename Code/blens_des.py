# coding: utf-8
from scipy.interpolate import RectBivariateSpline, interp1d

lmin = 60

clcibk = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/clcibkdes.txt')
clcib = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/clcibdes.txt')

clk_des = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/clkdes.txt')

cldesk = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cldeskdes.txt')
cldes = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cldesdes.txt')



cldesik = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cldesikdesi.txt')
cldesi = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cldesidesi.txt')
clk = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/clkdesi.txt')





clpp = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/cmb_cl/pp.txt')
clee = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/cmb_cl/ee.txt')
ells_cmb = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/cmb_cl/ell.txt')

ells_desi = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/ellsdesi.txt')
ells_des = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/ells_des.txt')

clee *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
clpp = clpp / (ells_cmb.astype(float)) ** 4

# for i, l in enumerate(ells_cmb):
#     if l < lmin:
#         clpp[i] = 0.0

lbins = np.logspace(1, 3.2, 90)
import lensing

print

clbb = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp, lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_th = np.loadtxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/cmb_cl/bb.txt')
clbb_th *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))




rho_cib = clcibk / (np.sqrt(clk_des * (clcib + 525.)))
rho_cib2 = clcibk / (np.sqrt(clk_des * (clcib + 225.)))
rho_des = cldesk / (np.sqrt(clk_des * (cldes)))
rho_desi = cldesik / (np.sqrt(clk * (cldesi)))

print rho_desi,rho_des,rho_cib2


for i, l in enumerate(ells_des):
    if l < lmin:
        rho_cib[i] = 0.0
        rho_cib2[i] = 0.0

rho_cib_fun = interp1d(ells_des, rho_cib)
rho_cib_fun2 = interp1d(ells_des, rho_cib2)
rho_des_fun = interp1d(ells_des, rho_des)
rho_desi_fun = interp1d(ells_desi, rho_desi)

# print ells_cmb,ells_desi,ells_des

clbb_cib = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_cib_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_cib2 = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_cib_fun2(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_des = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_des_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
clbb_desi = lensing.utils.calc_lensed_clbb_first_order(
    lbins, clee, clpp * (1. - rho_desi_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)

np.savetxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cbb_res_cib_525.txt',
           np.array(clbb_cib2.specs['cl'], dtype=float))

np.savetxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cbb_res_des.txt',
           np.array(clbb_des.specs['cl'], dtype=float))

np.savetxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cbb_res_cib.txt',
           np.array(clbb_cib.specs['cl'], dtype=float))

np.savetxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cbb_res_desi.txt',
           np.array(clbb_desi.specs['cl'], dtype=float))

np.savetxt('/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cbb.txt',
           np.array(clbb.specs['cl'], dtype=float))

np.savetxt(
    '/home/manzotti/my_version_cosmosis/cosmosis-standard-library/structure/PS_limber/output/limber_spectra/cbb_res_cib.txt', clbb.ls)
