# coding: utf-8
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
import configparser as ConfigParser
import numpy as np
from joblib import Parallel, delayed, cpu_count
import scipy.integrate as integrate
# import profiling
# from profiling.sampling import SamplingProfiler
import time


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


def compute_BB(clee_fun, clpp_fun, ell_b_bin, ell_phi_bin):
    def integrand(theta, ell, L):
        clee = clee_fun(ell)
        return (ell / (2. * np.pi)**2 * (
            L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(
                np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee *
            (np.sin(2. * theta))**2)

    options1 = {'limit': 4900, 'epsabs': 0., 'epsrel': 1.e-7}
    options2 = {'limit': 4900, 'epsabs': 0., 'epsrel': 1.e-3}

    lbins_int = np.arange(ell_b_bin[0], ell_b_bin[1], 15)

    clbb_ell = [
        integrate.nquad(
            integrand, [[0., 2. * np.pi], [4, 3000]],
            args=(L, ),
            opts=[options1, options2])[0] for L in lbins_int
    ]
    # print(clbb_ell, 'errors on b', [integrate.nquad(
    # integrand, [[0., 2. * np.pi], [4, 3000]], args=(L,), opts=[options1,
    # options2])[1] for L in lbins_int])

    # reconstruction_noise_ell = [integrate.nquad(reconstruction_noise_integrand, [[0., 2. * np.pi], [6, 2500]], args=(
    #     L, x1, x2), opts=[options1, options2])[0] for L in np.arange(10, 1000, 50)]

    return np.mean(clbb_ell)


def compute_res_parallel(rho_filename, output_dir, clee_fun, clpp_fun,
                         nle_fun):
    """
    This function compute both a lensing B-mode from cle and clpp ("test mode") as well as the residual power after delensing if a rho_file that represent the degree of correlation of the phi map with the true phi is passed.
    """
    # print(output_dir)
    # print('start integration')

    if rho_filename == 'test':
        lmax = 5000

        def integrand(theta, ell, L):
            clee = clee_fun(ell)
            # typical integrand to get BB from EE and phi. no correlation rho here
            return (ell / (2. * np.pi)**2 *
                    (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(
                        np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))) * clee * (np.sin(2. * theta))**2)
    else:
        rho_filename = output_dir + 'correlation_values_3G_MSIP/' + rho_filename
        print(rho_filename)
        rho = np.loadtxt(rho_filename)[:, 1]
        lbins = np.loadtxt(rho_filename)[:, 0]
        # print(lbins)
        rho_fun = InterpolatedUnivariateSpline(
            lbins, np.nan_to_num(rho), ext='zeros')
        lmax = 5000

        def integrand(theta, ell, L):
            """
            integrand to get the residual B-mode
            """
            clee = clee_fun(ell)
            # print(" ")
            # print(ell, rho_fun(ell))

            return (ell / (2. * np.pi)**2 *
                    (L * ell * np.cos(theta) - ell**2)**2 * clpp_fun(
                        np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta))
            ) * clee * (np.sin(2. * theta))**2) * (
                1. - (clee / (clee + nle_fun(ell))) * rho_fun(np.sqrt(L**2 + ell**2 - 2. * ell * L * np.cos(theta)))**2)

    lbins_int = np.linspace(29, 2000, 200)
    options1 = {'limit': 1500, 'epsabs': 0., 'epsrel': 5.e-3}
    options2 = {'limit': 1500, 'epsabs': 0., 'epsrel': 5.e-3}

    tic = time.time()
    clbb_res_ell = [
        integrate.nquad(
            integrand, [[0., 2. * np.pi], [30, lmax]],
            args=(L, ),
            opts=[options1, options2])[0] for L in lbins_int
    ]
    print('time for the integral total', time.time() - tic)
    np.savetxt(rho_filename.split('.txt')[0] + 'Cbb_res.txt', clbb_res_ell)
    # print(output_dir)
    np.savetxt(output_dir + 'limber_spectra/cbb_res_ls.txt', lbins_int)

    return clbb_res_ell


def load_res(rho_filenames):
    output_dir = '/home/manzotti/galaxies_delensing/Data/'
    print(output_dir)
    clbb_res_ell = []
    lbins_int = np.loadtxt(output_dir + 'limber_spectra/cbb_res_ls.txt')

    for rho_filename in rho_filenames:
        # print('start integration')
        if rho_filename is not 'test':
            rho_filename = output_dir + 'limber_spectra/' + rho_filename
            print(rho_filename)

        clbb_res_ell.append(
            np.loadtxt(
                rho_filename.split('.txt')[0] + 'limber_spectra/' +
                'Cbb_res.txt'))

    return clbb_res_ell, lbins_int


def compute_deriv_2(ells, clee_fun, clpp, l_phi, l_bb):

    clpp_mask = np.where(
        np.logical_and(ells >= l_phi[0], ells <= l_phi[1]), clpp,
        np.zeros_like(clpp))
    clpp_fun_plus = InterpolatedUnivariateSpline(
        ells[:5000], clpp + clpp_mask * 0.15, ext=1)
    clpp_fun_minus = InterpolatedUnivariateSpline(
        ells[:5000], clpp - clpp_mask * 0.15, ext=1)
    clbb_1 = compute_BB(clee_fun, clpp_fun_minus, l_bb, l_phi)
    clbb_2 = compute_BB(clee_fun, clpp_fun_plus, l_bb, l_phi)
    # print(l_bb,l_phi,(clbb_1 - clbb_2) / clbb_1)
    clbb_der = (np.array(clbb_2) - clbb_1) / (0.15 * 2.)
    return clbb_der


def compute_deriv(ells, clee_fun, clpp, l_phi, l_bb):

    clpp_mask = np.where(
        np.logical_and(ells >= l_phi[0], ells <= l_phi[1]), clpp,
        np.zeros_like(clpp))
    # dx = 2.75
    clpp_fun_test = InterpolatedUnivariateSpline(
        ells[:5000], clpp_mask, ext='zeros')
    clbb_der = np.array(compute_BB(clee_fun, clpp_fun_test, l_bb,
                                   l_phi))  # / dx
    return clbb_der


def compute_deriv_CEE2(ells, clee, clpp_fun, l_phi, l_bb):
    dx = 0.35
    clee_mask = np.where(
        np.logical_and(ells >= l_phi[0], ells <= l_phi[1]), clee,
        np.zeros_like(clee))
    clee_fun_plus = InterpolatedUnivariateSpline(
        ells[:5000], clee + clee_mask * dx, ext=1)
    clee_fun_minus = InterpolatedUnivariateSpline(
        ells[:5000], clee - clee_mask * dx, ext=1)
    clee_fun_plus2 = InterpolatedUnivariateSpline(
        ells[:5000], clee + 2. * clee_mask * dx, ext=1)
    clee_fun_minus2 = InterpolatedUnivariateSpline(
        ells[:5000], clee - 2. * clee_mask * dx, ext=1)

    # clee_fun_test = InterpolatedUnivariateSpline(
    #     ells[:5000], clee_mask * 2. * dx, ext=1)

    # clbb_1 = compute_BB(clee_fun_plus2, clpp_fun, l_bb, l_phi)
    clbb_2 = compute_BB(clee_fun_plus, clpp_fun, l_bb, l_phi)
    clbb_3 = compute_BB(clee_fun_minus, clpp_fun, l_bb, l_phi)
    # clbb_4 = compute_BB(clee_fun_minus2, clpp_fun, l_bb, l_phi)

    # print(l_phi,l_bb,(clbb_2 - clbb_3), compute_BB(clee_fun_test, clpp_fun, l_bb, l_phi))
    clbb_der = (np.array(clbb_2) - clbb_3) / (dx * 2.)
    return clbb_der


def compute_deriv_CEE(ells, clee, clpp_fun, l_phi, l_bb):
    dx = 0.35
    clee_mask = np.where(
        np.logical_and(ells >= l_phi[0], ells <= l_phi[1]), clee,
        np.zeros_like(clee))
    # clee_fun_plus = InterpolatedUnivariateSpline(
    #     ells[:5000], clee + clee_mask * dx, ext=1)
    # clee_fun_minus = InterpolatedUnivariateSpline(
    #     ells[:5000], clee - clee_mask * dx, ext=1)
    # clee_fun_plus2 = InterpolatedUnivariateSpline(
    #     ells[:5000], clee + 2. * clee_mask * dx, ext=1)
    # clee_fun_minus2 = InterpolatedUnivariateSpline(
    #     ells[:5000], clee - 2. * clee_mask * dx, ext=1)

    clee_fun_test = InterpolatedUnivariateSpline(
        ells[:5000], clee_mask * 2. * dx, ext=1)

    # # clbb_1 = compute_BB(clee_fun_plus2, clpp_fun, l_bb, l_phi)
    # clbb_2 = compute_BB(clee_fun_plus, clpp_fun, l_bb, l_phi)
    # clbb_3 = compute_BB(clee_fun_minus, clpp_fun, l_bb, l_phi)
    # # clbb_4 = compute_BB(clee_fun_minus2, clpp_fun, l_bb, l_phi)
    clbb_der = np.array(compute_BB(clee_fun_test, clpp_fun, l_bb, l_phi)) / dx
    return clbb_der


def compute_deriv_grid_CEE(delta_e, delta_b, n_jobs=15):
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    Config_ini = ConfigParser.ConfigParser()
    Config_ini.read(inifile)
    output_dir = Config_ini.get('test', 'save_dir')

    datadir = output_dir

    clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
    clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
    ells = np.loadtxt(datadir + 'cmb_cl/ell.txt')

    # prof = SamplingProfiler()s
    # prof.start()
    clpp_fun = InterpolatedUnivariateSpline(
        ells[:5000], clpp[:5000], ext='zeros')
    lb = np.arange(4, 1500, delta_b)
    lee = np.arange(4, 2000, delta_e)
    clbb_der = np.zeros((len(lb), len(lee)))
    for i, ell_b in enumerate(lb):
        print(ell_b + delta_b / 2.)
        clbb_der[i, :] = np.array(
            Parallel(n_jobs=n_jobs, verbose=50)(delayed(compute_deriv_CEE)(
                ells, clee, clpp_fun, [ell_e, ell_e + delta_e],
                [ell_b, ell_b + delta_b]) for ell_e in lee))

        np.save('./grid_deriv_delta_EE_{}_delta_B_{}_der1'.format(
            delta_e, delta_b), clbb_der)

        # prof.stop()
        # prof.run_viewer()

    np.save('./grid_deriv_delta_EE_{}_delta_B_{}_der1'.format(
        delta_e, delta_b), clbb_der)
    return clbb_der


def compute_deriv_grid(delta_phi, delta_b, n_jobs=15):
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    Config_ini = ConfigParser.ConfigParser()
    Config_ini.read(inifile)
    output_dir = Config_ini.get('test', 'save_dir')

    datadir = output_dir

    clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
    clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
    ells = np.loadtxt(datadir + 'cmb_cl/ell.txt')

    # prof = SamplingProfiler()s
    # prof.start()
    clee_fun = InterpolatedUnivariateSpline(ells[:5000], clee[:5000], ext=2)
    lb = np.arange(4, 1500, delta_b)
    lphi = np.arange(4, 2000, delta_phi)
    clbb_der = np.zeros((len(lb), len(lphi)))
    for i, ell_b in enumerate(lb):
        print(ell_b + delta_b / 2.)
        clbb_der[i, :] = (ell_b + delta_b / 2.) * np.array(
            Parallel(n_jobs=n_jobs, verbose=50)(delayed(compute_deriv)(
                ells, clee_fun, clpp, [ell_phi, ell_phi + delta_phi],
                [ell_b, ell_b + delta_b]) for ell_phi in lphi))

        np.save('./grid_deriv_delta_p_{}_delta_B_{}_post'.format(
            delta_phi, delta_b), clbb_der)

        # prof.stop()
        # prof.run_viewer()

    np.save('./grid_deriv_delta_p_{}_delta_B_{}'.format(delta_phi, delta_b),
            clbb_der)
    return clbb_der


def main(rho_names, nle_fun, clpp_fun, clee_fun):
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    Config_ini = ConfigParser.ConfigParser()
    Config_ini.read(inifile)
    # output_dir = Config_ini.get('test', 'save_dir')
    output_dir = '/home/manzotti/galaxies_delensing/Data/'
    # datadir = output_dir
    print('in rho_to res', cpu_count(), rho_names)
    cpus = np.min([len(rho_names), cpu_count() - 2])
    return Parallel(
        n_jobs=1, verbose=0)(delayed(compute_res_parallel)(
            i, output_dir, clee_fun, clpp_fun, nle_fun) for i in rho_names)


def compute_derivates():
    '''
    Here we compute the value we put in the figures of ckk_ell * dCbb/dckk
    '''
    inifile = '/home/manzotti/cosmosis/modules/limber/galaxies_delens.ini'

    Config_ini = ConfigParser.ConfigParser()
    Config_ini.read(inifile)
    output_dir = Config_ini.get('test', 'save_dir')

    datadir = output_dir
    print("output_dir", output_dir)

    clpp = np.loadtxt(datadir + 'cmb_cl/pp.txt')
    clee = np.loadtxt(datadir + 'cmb_cl/ee.txt')
    ells_cmb = np.loadtxt(datadir + 'cmb_cl/ell.txt')

    clee *= 2. * np.pi / (ells_cmb.astype(float) *
                          (ells_cmb.astype(float) + 1.))
    clpp = clpp * 2. * np.pi / \
        (ells_cmb.astype(float) * (ells_cmb.astype(float) + 1.))

    lbins = np.logspace(1, 3.5, 190)

    clbb_th = np.loadtxt(output_dir + 'cmb_cl/bb.txt')
    clbb_th *= 2. * np.pi / (ells_cmb.astype(float) *
                             (ells_cmb.astype(float) + 1.))

    # clpp_zeroth
    clee_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clee[:5000], ext=2)
    clpp_fun = InterpolatedUnivariateSpline(
        ells_cmb[:5000], clpp[:5000], ext='zeros')
    nle_fun = InterpolatedUnivariateSpline(ells_cmb[:5000], nle[:5000], ext=2)

    compute_res_parallel('test', output_dir, clee_fun, clpp_fun, nle_fun)


if __name__ == "__main__":
    main(rho_names, nle_fun, clpp_fun, clee_fun)
