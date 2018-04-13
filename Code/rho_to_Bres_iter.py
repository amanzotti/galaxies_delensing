# coding: utf-8
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import sys
import configparser as ConfigParser
import numpy as np
from joblib import Parallel, delayed
import scipy.integrate as integrate

# import pyximport
# pyximport.install()
# import rho_to_Bres_iter_c
import quicklens as ql


def calc_nlqq(qest, clXX, clXY, clYY, flX, flY):

    print("calculating full noise level for estimator of type")
    clqq_fullsky = qest.fill_clqq(
        np.zeros(lmax + 1, dtype=np.complex), clXX * flX * flX,
        clXY * flX * flY, clYY * flY * flY)
    resp_fullsky = qest.fill_resp(qest, np.zeros(lmax + 1, dtype=np.complex),
                                  flX, flY)
    nlqq_fullsky = clqq_fullsky / resp_fullsky**2

    return nlqq_fullsky


def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      - beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             - maximum multipole.
    """
    ls = np.arange(0, lmax + 1)
    return np.exp(-(fwhm_arcmin * np.pi / 180. / 60.)**2 /
                  (16. * np.log(2.)) * ls * (ls + 1.))


def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
          * noise_uK_arcmin - map noise level in uK.arcmin
          * fwhm_arcmin     - beam full-width-at-half-maximum (fwhm) in arcmin.
          * lmax            - maximum multipole.
    """
    return (noise_uK_arcmin * np.pi / 180. / 60.)**2 / bl(fwhm_arcmin, lmax)**2


def f_TT(l1, l2, theta):
    return cltt_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) + cltt_fun(l2) * (
        l2**2 + l1 * l2 * np.cos(theta))


def f_TE(l1, l2, theta):
    return clte_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) * np.cos(
        2. * theta) + clte_fun(l2) * (l2**2 + l1 * l2 * np.cos(theta))


def f_TB(l1, l2, theta):
    return clte_fun(l1) * (
        l1**2 + l1 * l2 * np.cos(theta)) * np.sin(2. * theta)


def f_EE(l1, l2, theta):
    return (clee_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) + clee_fun(l2) *
            (l2**2 + l1 * l2 * np.cos(theta))) * np.cos(2. * theta)


def f_EB(l1, l2, theta):
    global clee_fun, clbb_fun
    return (clee_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) - clbb_fun(l2) *
            (l2**2 + l1 * l2 * np.cos(theta))) * np.sin(2. * theta)


def f_BB(l1, l2, theta):
    return (clbb_fun(l1) * (l1**2 + l1 * l2 * np.cos(theta)) + clbb_fun(l2) *
            (l2**2 + l1 * l2 * np.cos(theta))) * np.cos(2. * theta)


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

        return (cltt1 * clee2 * fl1l2(l1, l2, theta, x1, x2) -
                clte_fun(l1) * clte_fun(l1) * fl1l2(l1, l2, theta, x1, x2)) / (
                    cltt1 * clee2 * clee1 * cltt2 -
                    (clte_fun(l1) * clte_fun(l2))**2)

    elif fields == 'EE':

        return (fl1l2(l1, l2, theta, x1, x2)) / (
            2. * clee_fun(l1) * clee_fun(l2))

    elif fields == 'BE':
        return (fl1l2(l1, l2, theta, x1, x2)) / (clbb_fun(l1) * clee_fun(l2))
    else:
        sys.exit('fields not recognized')

    return 1


def reconstruction_noise_integrand(theta, ell, L, x1, x2):
    # print(theta, ell, L, x1,x2)
    l2 = np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))
    # print(np.log(ell / (2. * np.pi)**2 * fl1l2(ell, l2, theta, x1, x2) * F_alpha(ell, l2, theta, x1, x2)))
    if l2 < 4.:
        return 1e20
    else:
        return 1e-15 * (ell / (
            2. * np.pi
        )**2 * fl1l2(ell, l2, theta, x1, x2) * F_alpha(ell, l2, theta, x1, x2))


# (ell / (2. * np.pi)**2 * fl1l2(ell, np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta)), theta, x1, x2) * F_alpha(ell, np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta)), theta, x1, x2))


def reconstruction_noise(x1, x2):
    options1 = {'limit': 900, 'epsabs': 0., 'epsrel': 1.49e-05}
    options2 = {'limit': 900, 'epsabs': 0., 'epsrel': 1.49e-4}

    reconstruction_noise_ell = [
        integrate.nquad(
            reconstruction_noise_integrand, [[0., 2. * np.pi], [6, 2500]],
            args=(L, x1, x2),
            opts=[options1, options2])[0] for L in np.arange(10, 1000, 50)
    ]
    print('integral', reconstruction_noise_ell)
    #     np.savetxt(datadir + 'limber_spectra/' +
    #                'Cphi_noise{}.txt'.format(x1 + x2), reconstruction_noise_ell)
    #     np.savetxt(datadir + 'limber_spectra/cphi_noise_ls.txt', np.arange(4, 1500, 10))
    return L**2 / (1e15 * np.array(reconstruction_noise_ell))


def compute_res_from_rho(rho):

    lbins = np.logspace(0, 4, 190)
    rho = np.interp(lbins, np.arange(0, len(rho)), rho)

    rho_fun = InterpolatedUnivariateSpline(
        lbins, np.nan_to_num(rho), ext='raise')

    def integrand(theta, ell, L):
        # print(ell,np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta)))
        clee = clee_fun(ell)
        return (ell / (2. * np.pi)**2 * (
            L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(
                np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee *
                (np.sin(2. * theta))**2) * (
                    1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell)**2)

    options1 = {'limit': 900, 'epsabs': 0., 'epsrel': 1.49e-05}
    options2 = {'limit': 900, 'epsabs': 0., 'epsrel': 1.49e-4}
    l_res = np.arange(10, 2500, 50)
    clbb_res_ell = [
        integrate.nquad(
            integrand, [[0., 2. * np.pi], [4, 3000]],
            args=(L, ),
            opts=[options1, options2])[0] for L in l_res
    ]

    return l_res, clbb_res_ell


def compute_res_parallel(rho_filename):

    print('start integration')

    if rho_filename == 'test':

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 *
                    (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(
                        np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta)))
                    * clee * (np.sin(2. * theta))**2)
    else:
        rho = np.loadtxt(rho_filename)
        lbins = np.loadtxt('lbins.txt')
        rho_fun = InterpolatedUnivariateSpline(
            lbins, np.nan_to_num(rho), ext='raise')

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            return (ell / (2. * np.pi)**2 *
                    (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(
                        np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))
                    ) * clee * (np.sin(2. * theta))**2) * (
                        1. - (clee / (clee + nle_fun(ell))) * rho_fun(ell)**2)

    clbb_res_ell = [
        integrate.dblquad(
            integrand,
            4,
            2500,
            lambda x: 0,
            lambda x: 2. * np.pi,
            args=(L, ),
            epsabs=1.49e-08,
            epsrel=1.49e-07)[0] for L in np.arange(10, 1500, 10)
    ]

    np.savetxt(rho_filename.split('.txt')[0] + 'Cbb_res.txt', clbb_res_ell)
    np.savetxt(datadir + 'limber_spectra/cbb_res_ls.txt', np.arange(
        4, 1500, 10))

    return clbb_res_ell


if __name__ == "__main__":

    cosmosis_dir = '/home/manzotti/cosmosis/'
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    noise_pol = 0.
    fwhm_beam = 1.

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

    ells_cmb = np.loadtxt(datadir + '/cmb_cl/ell.txt')

    clee *= 2. * np.pi / (ells_cmb.astype(float) *
                          (ells_cmb.astype(float) + 1.))
    clte *= 2. * np.pi / (ells_cmb.astype(float) *
                          (ells_cmb.astype(float) + 1.))
    cltt *= 2. * np.pi / (ells_cmb.astype(float) *
                          (ells_cmb.astype(float) + 1.))

    clpp = clpp * 2. * np.pi / (ells_cmb.astype(float) *
                                (ells_cmb.astype(float) + 1.))

    lbins = np.logspace(0, 3.5, 190)

    clbb_th = np.loadtxt(output_dir + '/cmb_cl/bb.txt')
    clbb_th *= 2. * np.pi / (ells_cmb.astype(float) *
                             (ells_cmb.astype(float) + 1.))
    nle = nl(noise_pol, fwhm_beam, lmax=ells_cmb[-1])[2:]

    clee_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clee[:5000], ext=2)
    clte_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clte[:5000], ext=2)
    cltt_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], cltt[:5000] + nle, ext=2)
    clpp_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clpp[:5000], ext='zeros')
    nle_fun = InterpolatedUnivariateSpline(ells_cmb[:5000], nle[:5000], ext=2)

    surveys = [
        'test', 'cib', 'des', 'comb_des_cib', 'comb_des_cib_cmb', 'ska10',
        'ska5', 'ska1', 'ska01', 'lsst', 'euclid', 'rho_comb', 'rho_cmbS4',
        'rho_cmbS3'
    ]
    # generating noise in E-modes
    # B_res = Parallel(n_jobs=len(surveys), verbose=50)(delayed(
    #     compute_res)(i) for i in surveys)

    rho_names = [
        'rho_cib.txt', 'rho_des.txt', 'rho_cmb_current.txt',
        'rho_gals_current.txt', 'rho_comb_current.txt', 'rho_cib.txt',
        'rho_cmb_S3.txt', 'rho_gals_S3.txt', 'rho_comb_S3.txt',
        'rho_cmb_S4.txt', 'rho_gals_S4.txt', 'rho_comb_S4.txt'
    ]

    # calculation parameters.
    lmax = 3000  # maximum multipole for T, E, B and \phi.
    nx = 512  # number of pixels for flat-sky calc.
    dx = 1. / 60. / 180. * np.pi  # pixel width in radians.

    nlev_t = 0.0  # temperature noise level, in uK.arcmin.
    nlev_p = 0.0  # polarization noise level, in uK.arcmin.
    beam_fwhm = 1.  # Gaussian beam full-width-at-half-maximum.

    cl_unl = ql.spec.get_camb_scalcl(lmax=lmax)  # unlensed theory spectra.
    cl_len = ql.spec.get_camb_lensedcl(lmax=lmax)  # lensed theory spectra.

    bl = bl(beam_fwhm, lmax)  # transfer function.
    pix = ql.maps.pix(nx, dx)

    # noise spectra
    nltt = 0.0 * (np.pi / 180. / 60. * nlev_t)**2 / bl**2
    nlee = nlbb = 0.0 * (np.pi / 180. / 60. * nlev_p)**2 / bl**2
    # signal spectra
    slee = cl_len.clee
    slpp = cl_unl.clpp

    slbb = cl_len.clbb
    zero = np.zeros(lmax + 1)

    # signal+noise spectra
    clee = slee + nlee
    clbb = slbb + nlbb
    clpp = slpp
    # filter functions

    fle = np.zeros(lmax + 1)
    fle[2:] = 1. / clee[2:]
    flb = np.zeros(lmax + 1)
    flb[2:] = 1. / clbb[2:]

    # intialize quadratic estimators
    qest_EB = ql.qest.lens.phi_EB(slee)

    # first step get phi

    nlpp_EB_fullsky_dict = []
    rho_dict = []
    clbb_res_dict = []

    nlpp_EB_fullsky = calc_nlqq(qest_EB, clee, zero, clbb, fle, flb)
    nlpp_EB_fullsky_dict.append(nlpp_EB_fullsky)

    rho = np.sqrt(clpp / (clpp + nlpp_EB_fullsky))
    rho_dict.append(rho)

    l_res, clbb_res = compute_res_from_rho(rho)
    clbb_res_dict.append(clbb_res)

    for iter_n in np.arange(0, 3):
        print(iter_n)

        slbb = np.interp(cl_len.ls, l_res, clbb_res)
        zero = np.zeros(lmax + 1)

        # signal+noise spectra
        clee = slee + nlee
        clbb = slbb + nlbb
        clpp = slpp
        # filter functions

        fle = np.zeros(lmax + 1)
        fle[2:] = 1. / clee[2:]
        flb = np.zeros(lmax + 1)
        flb[2:] = 1. / clbb[2:]

        # intialize quadratic estimators
        qest_EB = ql.qest.lens.phi_EB(slee)

        # first step get phi
        nlpp_EB_fullsky = calc_nlqq(qest_EB, clee, zero, clbb, fle, flb)
        nlpp_EB_fullsky_dict.append(nlpp_EB_fullsky)
        rho = np.sqrt(clpp / (clpp + nlpp_EB_fullsky))
        rho_dict.append(rho)
        l_res, clbb_res = compute_res_from_rho(rho)
        clbb_res_dict.append(clbb_res)


def run_one_iter():

    nlpp_EB_fullsky = calc_nlqq(qest_EB, clee, zero, clbb, fle, flb)
    nlpp_EB_fullsky_dict.append(nlpp_EB_fullsky)
    rho = np.sqrt(clpp / (clpp + nlpp_EB_fullsky))
    rho_dict.append(rho)
    clbb_res_dict.append(clbb_res)
    l_res, clbb_res = compute_res_from_rho(rho)

    slbb = np.interp(cl_len.ls, l_res, clbb_res)
    zero = np.zeros(lmax + 1)

    # signal+noise spectra
    clee = slee + nlee
    clbb = slbb + nlbb
    clpp = slpp
    # filter functions

    fle = np.zeros(lmax + 1)
    fle[2:] = 1. / clee[2:]
    flb = np.zeros(lmax + 1)
    flb[2:] = 1. / clbb[2:]

    # intialize quadratic estimators
    qest_EB = ql.qest.lens.phi_EB(slee)

    # first step get phi
