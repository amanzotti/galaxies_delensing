# coding: utf-8
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import sys
import configparser as ConfigParser
import numpy as np
from joblib import Parallel, delayed
import scipy.integrate as integrate


def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      - beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             - maximum multipole.
    """
    ls = np.arange(0, lmax + 1)
    return np.exp(-(fwhm_arcmin * np.pi / 180. / 60.)**2 / (16. * np.log(2.)) * ls * (ls + 1.))


def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
          * noise_uK_arcmin - map noise level in uK.arcmin
          * fwhm_arcmin     - beam full-width-at-half-maximum (fwhm) in arcmin.
          * lmax            - maximum multipole.
    """
    return (noise_uK_arcmin * np.pi / 180. / 60.)**2 / bl(fwhm_arcmin, lmax)**2


def f_TT(l1, l2, theta):
    return cltt_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) + cltt_fun(l2) * (l2**2 + l1 * l2 * np.cos(theta))


def f_TE(l1, l2, theta):
    return clte_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) * np.cos(2. * theta) + clte_fun(l2) * (l2**2 + l1 * l2 * np.cos(theta))


def f_TB(l1, l2, theta):
    return clte_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) * np.sin(2. * theta)


def f_EE(l1, l2, theta):
    return (clee_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) + clee_fun(l2) * (l2**2 + l1 * l2 * np.cos(theta))) * np.cos(2. * theta)


def f_EB(l1, l2, theta):
    global clee_fun, clbb_fun
    return (clee_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) - clbb_fun(l2) * (l2**2 + l1 * l2 * np.cos(theta))) * np.sin(2. * theta)


def f_BB(l1, l2, theta):
    return (clbb_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) + clbb_fun(l2) * (l2**2 + l1 * l2 * np.cos(theta))) * np.cos(2. * theta)


def fl1l2(l1, l2, theta, x1='E', x2='E'):
    fields = ''.join(sorted(x2 + x1)).upper()
    if fields == 'TT':
        return f_TT(l1, l2, theta)
    elif fields == 'ET':
        return f_TE(l1, l2, theta)

    elif fields == 'EE':
        return f_EE(l1, l2, theta)

    elif fields == 'BE':
        return f_EB(l1, l2, theta)
    else:
        sys.exit('fields not recognized')
    return 1


def F_alpha(l1, l2, theta, x1='E', x2='E'):
    fields = ''.join(sorted(x2 + x1)).upper()
    if fields == 'TT':
        cltt1 = cltt_fun(l1)
        cltt2 = cltt_fun(l2)
        ftt = cltt1 * (l1 * l1 + l1 * l2 * np.cos(theta)) + \
            cltt2 * (l2 * l2 + l1 * l2 * np.cos(theta))
        return (ftt) / (2. * cltt1 * cltt2)

    elif fields == 'ET':
        clee1 = clee_fun(l1)
        clee2 = clee_fun(l2)
        cltt1 = clee_fun(l1)
        cltt2 = clee_fun(l2)

        return (cltt1 * clee2 * fl1l2(l1, l2, theta, x1, x2) - clte_fun(l1) * clte_fun(l1) * fl1l2(l1, l2, theta, x1, x2)) / (cltt1 * clee2 * clee1 * cltt2 - (clte_fun(l1) * clte_fun(l2))**2)

    elif fields == 'EE':

        return (fl1l2(l1, l2, theta, x1, x2)) / (2. * clee_fun(l1) * clee_fun(l2))

    elif fields == 'BE':
        return (fl1l2(l1, l2, theta, x1, x2)) / (clbb_fun(l1) * clee_fun(l2))
    else:
        sys.exit('fields not recognized')

    return 1


def reconstruction_noise_integrand(theta, ell, L, x1, x2):
    l2 = np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))
    if l2 < 4.:
        return 1e30
    else:
        return (ell / (2. * np.pi)**2 * fl1l2(ell, l2, theta, x1, x2) * F_alpha(ell, l2, theta, x1, x2))


def reconstruction_noise(x1='E', x2='E'):
    reconstruction_noise_ell = [integrate.dblquad(
        reconstruction_noise_integrand, 4, 2500, lambda x: 0, lambda x: 2. * np.pi, args=(L, x1, x2), epsabs=1.49e-08, epsrel=1.49e-07)[0] for L in np.arange(100, 110, 500)]
#     np.savetxt(datadir + 'limber_spectra/' +
#                'Cphi_noise{}.txt'.format(x1 + x2), reconstruction_noise_ell)
#     np.savetxt(datadir + 'limber_spectra/cphi_noise_ls.txt', np.arange(4, 1500, 10))
    return L**2 / reconstruction_noise_ell


def compute_res_parallel(rho_filename):

    print('start integration')

    if rho_filename == 'test':
        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2)
    else:
        rho = np.loadtxt(rho_filename)
        lbins = np.loadtxt('lbins.txt')
        rho_fun = InterpolatedUnivariateSpline(
            lbins, np.nan_to_num(rho), ext='raise')

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell) ** 2)

    clbb_res_ell = [integrate.dblquad(
        integrand, 4, 2500, lambda x: 0, lambda x: 2. * np.pi, args=(L,), epsabs=1.49e-08, epsrel=1.49e-07)[0] for L in np.arange(10, 1500, 10)]

    np.savetxt(rho_filename.split('.txt')[0] + 'Cbb_res.txt', clbb_res_ell)
    np.savetxt(datadir + 'limber_spectra/cbb_res_ls.txt', np.arange(4, 1500, 10))

    return clbb_res_ell


def compute_res(rho_filename, noise_pol=2., fwhm_beam=30.):

    cosmosis_dir = '/home/manzotti/cosmosis/'
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    Config_ini = ConfigParser.ConfigParser()
    # values = ConfigParser.ConfigParser()
    Config_ini.read(inifile)
    # values_file = Config_ini.get('pipeline', 'values')
    output_dir = Config_ini.get('test', 'save_dir')

    datadir = output_dir

    clpp = np.loadtxt(datadir + '/cmb_cl/pp.txt')
    clee = np.loadtxt(datadir + '/cmb_cl/ee.txt')
    clte = np.loadtxt(datadir + '/cmb_cl/te.txt')
    cltt = np.loadtxt(datadir + '/cmb_cl/tt.txt')
    # cltb = np.loadtxt(datadir + 'cmb_cl/ee.txt')

    ells_cmb = np.loadtxt(datadir + '/cmb_cl/ell.txt')

    clee *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
    clte *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
    cltt *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

    clpp = clpp * 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

    lbins = np.logspace(1, 3.5, 190)

    clbb_th = np.loadtxt(
        output_dir + '/cmb_cl/bb.txt')
    clbb_th *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
    nle = nl(noise_pol, fwhm_beam, lmax=ells_cmb[-1])[2:]

    clee_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clee[:5000], ext=2)
    clte_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clte[:5000], ext=2)
    cltt_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], cltt[:5000], ext=2)
    clpp_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clpp[:5000], ext='zeros')
    nle_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], nle[:5000], ext=2)

    print('start integration')

    if rho_filename == 'test':
        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2)
    else:
        rho = np.loadtxt(rho_filename)[:, 1]
        lbins = np.loadtxt(rho_filename)[:, 0]
        rho_fun = InterpolatedUnivariateSpline(lbins, np.nan_to_num(rho), ext='raise')

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell) ** 2)

    clbb_res_ell = [integrate.dblquad(
        integrand, 4, 2500, lambda x: 0, lambda x: 2. * np.pi, args=(L,), epsabs=1.49e-06, epsrel=1.49e-05)[0] for L in np.arange(4, 1500, 10)]

    np.savetxt(rho_filename.split('.txt')[0] + 'Cbb_res.txt', clbb_res_ell)
    np.savetxt(datadir + 'limber_spectra/cbb_res_ls.txt', np.arange(4, 1500, 10))

    return clbb_res_ell


def load_res(labels):
    datadir = '/home/manzotti/cosmosis/modules/limber/galaxies_delens/'

    res_list = []
    for label in labels:
        res_list.append(np.loadtxt(datadir + 'limber_spectra/cbb_res_' + label + 'test3.txt'))
    return res_list

if __name__ == "__main__":

    cosmosis_dir = '/home/manzotti/cosmosis/'
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    noise_pol = 2.
    fwhm_beam = 30.

    Config_ini = ConfigParser.ConfigParser()
    values = ConfigParser.ConfigParser()
    Config_ini.read(inifile)
    values_file = Config_ini.get('pipeline', 'values')
    output_dir = Config_ini.get('test', 'save_dir')

    datadir = output_dir

    clpp = np.loadtxt(datadir + '/cmb_cl/pp.txt')
    clee = np.loadtxt(datadir + '/cmb_cl/ee.txt')
    clte = np.loadtxt(datadir + '/cmb_cl/te.txt')
    cltt = np.loadtxt(datadir + '/cmb_cl/tt.txt')
    # cltb = np.loadtxt(datadir + 'cmb_cl/ee.txt')

    ells_cmb = np.loadtxt(datadir + '/cmb_cl/ell.txt')

    clee *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
    clte *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
    cltt *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

    clpp = clpp * 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

    lbins = np.logspace(1, 3.5, 190)

    clbb_th = np.loadtxt(
        output_dir + '/cmb_cl/bb.txt')
    clbb_th *= 2. * np.pi / (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))
    nle = nl(noise_pol, fwhm_beam, lmax=ells_cmb[-1])[2:]

    clee_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clee[:5000], ext=2)
    clte_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clte[:5000], ext=2)
    cltt_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], cltt[:5000], ext=2)
    clpp_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clpp[:5000], ext='zeros')
    nle_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], nle[:5000], ext=2)

    surveys = ['test', 'cib', 'des', 'comb_des_cib', 'comb_des_cib_cmb',
               'ska10', 'ska5', 'ska1', 'ska01', 'lsst', 'euclid', 'rho_comb', 'rho_cmbS4', 'rho_cmbS3']

    surveys = ['rho_cmbS3', 'rho_cmbS4']
    # generating noise in E-modes
    nle = nl(1, 1, lmax=ells_cmb[-1])[2:]

    # B_res = Parallel(n_jobs=len(surveys), verbose=50)(delayed(
    #     compute_res)(i) for i in surveys)

    clee_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clee[:5000], ext=2)
    clpp_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clpp[:5000], ext='zeros')
    nle_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], nle[:5000], ext=2)

    rho_names = ['rho_cib.txt', 'rho_des.txt', 'rho_cmb_current.txt', 'rho_gals_current.txt', 'rho_comb_current.txt', 'rho_cib.txt',
                 'rho_cmb_S3.txt', 'rho_gals_S3.txt', 'rho_comb_S3.txt', 'rho_cmb_S4.txt', 'rho_gals_S4.txt', 'rho_comb_S4.txt']

    # for label in surveys:
    #     B_res2 = compute_res_2(label, clee_fun, clpp_fun, nle_fun)

    # for label in surveys:
    #     compute_res_3(label, clee_fun, clpp_fun, nle_fun)
    # compute_res_parallel('rho_cmbS4')
    print('doing integral')
    noise = reconstruction_noise('T', 'T')

    # B_res3 = Parallel(n_jobs=6, verbose=500)(delayed(
    #     compute_res_parallel)(i) for i in rho_names)

    # def compute_res(label_survey):
    #     lbins = np.logspace(1, 3.5, 190)

    #     if label_survey == 'test':
    #         clbb_res = lensing.utils.calc_lensed_clbb_first_order(
    #             lbins, clee, clpp, lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
    #     else:
    #         rho = np.loadtxt(
    #             '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
    #         rho_fun = interp1d(rho[:, 0], rho[:, 1], bounds_error=False, fill_value=0.)
    #         clbb_res = lensing.utils.calc_lensed_clbb_first_order(
    #             lbins, clee, clpp * (1. - (clee / (clee + nle)) * rho_fun(ells_cmb) ** 2), lmax=ells_cmb[-1], nx=2048, dx=4. / 60. / 180. * np.pi)
    #     np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + '.txt',
    #                np.array(clbb_res.specs['cl'], dtype=float))
    #     return np.array(clbb_res.specs['cl'], dtype=float)

    # def compute_res_2(label_survey, clee, clpp, nle):

    #     if label_survey == 'test':
    #         def integrand(ell):
    #             return ell**5 / 4. / np.pi * clpp(ell) * clee(ell)
    #     else:
    #         rho = np.loadtxt(
    #             '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
    #         rho_fun = interp1d(rho[:, 0], rho[:, 1], bounds_error=False, fill_value=0.)

    #         def integrand(ell):
    # return ell**5 / 4. / np.pi * clpp(ell) * clee(ell) * (1. - (clee(ell) /
    # (clee(ell) + nle(ell))) * rho_fun(ell) ** 2)

    #     clbb_res = integrate.quad(integrand, 4, 2500, limit=100, epsrel=1.49e-09)
    #     np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey +
    #                'test2.txt', clbb_res[0] * np.ones(2000))
    #     return clbb_res[0] * np.ones(2000)

    # def compute_res_3(label_survey, clee_fun, clpp_fun, nle_fun):
    #     rho = np.loadtxt(
    #         '/home/manzotti/cosmosis/modules/limber/galaxies_delens/limber_spectra/rho_' + label_survey + '.txt')
    #     rho_fun = interp1d(rho[:1000, 0], rho[:1000, 1], bounds_error=False, fill_value=0.)
    #     print('start integration')

    #     @jit
    #     def integrand(theta, ell, L):
    #         #     print(ell,theta)
    #         clee = clee_fun(ell)
    # return (ell / (2. * np.pi)**2 * (L * ell * np.cos(theta) - ell**2)**2 *
    # clpp_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee *
    # (np.sin(2. * theta))**2) * (1. - (clee / (clee + nle_fun(ell))) *
    # rho_fun(ell) ** 2)

    #     clbb_res_ell = [integrate.dblquad(
    #         integrand, 4, 3000, lambda x: 0, lambda x: 2. * np.pi, args=(L,))[0] for L in np.arange(4, 500, 10)]
    #     np.savetxt(datadir + 'limber_spectra/cbb_res_' + label_survey + 'test3.txt', clbb_res_ell)
    #     return clbb_res_ell
