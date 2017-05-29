'''
This script compute all the spectra needed for the delensing part.
'''
import pyximport
import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
import kappa_gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import limber_integrals
# import pickle as pk
import camb
from camb import model
import configparser
pyximport.install(reload_support=True)
# from joblib import Parallel, delayed
# from profiling.sampling import SamplingProfiler
# profiler = SamplingProfiler()


def cl_limber_z_local(chi_z, hspline, rbs, l, kernel_1, kernel_2=None, zmin=0.0, zmax=1100.):
    """ calculate the cross-spectrum at multipole l between kernels k1 and k2 in the limber approximation. redshift  version. See  cl_limber_x for the comoving distance version
   Notes: Here everything is assumed in h units. Maybe not the best choice but that is it.

    Args:
      z_chi: z(chi) redshift as a function of comoving distance.
      hspline: H(z). not used here kept to uniform to cl_limber_z
      rbs: Power spectrum spline P(k,z) k and P in h units
      l: angular multipole
      k1: First kernel
      k2: Optional Second kernel otherwise k2=k1
      zmin: Min range of integration, redshift
      zmax: Max range of integration, redshift


    Returns:

      cl_limber : C_l = \int_0^z_s dz {d\chi\over dz} {1/\chi^2} K_A(\chi(z)) K_B(\chi(z)\times P_\delta(k=l/\chi(z);z)

    """

    #  TODO check the H factor.
    if kernel_2 is None:
        kernel_2 = kernel_1

        def integrand(z):
            x = chi_z(z)
            h = hspline(z)
            k1 = kernel_1.w_lxz(l, x, z)**2
            pk = rbs.ev((l + 0.5) / x, z)
            return 1. / x / x * h * k1 * pk

    else:

        def integrand(z):
            x = chi_z(z)
            h = hspline(z)
            k1 = kernel_1.w_lxz(l, x, z)
            k2 = kernel_2.w_lxz(l, x, z)

            pk = rbs.ev((l + 0.5) / x, z)
            return 1. / x / x * h * k1 * k2 * pk

    # sys.exit()
    # func = InterpolatedUnivariateSpline(np.linspace(zmin,zmax,100), np.vectorize(integrand)(np.linspace(zmin,zmax,100)), ext='zeros')
    # print('')
    # print(func(zmax-(zmax-zmin)/2.))
    # print(scipy.integrate.quad(integrand, zmin, zmax, limit=300, epsrel=1.49e-06)[0])
    return scipy.integrate.quad(integrand, zmin, zmax, limit=300, epsrel=1.49e-06)[0]


def compute_spectra_parallel(label1, label2, kernels, chispline, hspline, rbs):
    limber_integrals.cl_limber_z_local(chispline, hspline, rbs, ini_pars['lbins'], kernel_1=kernels[
        i], kernel_2=kernels[j], zmin=max(kernels[i].zmin, kernels[j].zmin),
        zmax=min(kernels[i].zmax, kernels[j].zmax))


def setup(ini_file='./gal_delens_values.ini'):
    config = configparser.ConfigParser()

    # use configparser for this
    config.read('gal_delens_values.ini')
    # # L BINS
    llmin = config.getfloat('spectra', 'llmin', fallback=0.)
    llmax = config.getfloat('spectra', 'llmax', fallback=3.)
    dlnl = config.getfloat('spectra', "dlnl", fallback=.1)
    noisy = options.get_bool('spectra', "noisy", fallback=True)

    lbins = np.arange(llmin, llmax, dlnl)
    lbins = 10. ** lbins
    ini_pars = {}
    ini_pars['lbins'] = lbins
    return ini_pars


def parallel_limber(kernels):
    kernel1 = kernels[0]
    kernel2 = kernels[1]
    return cl_limber_z_local(chispline, hspline, rbs, l, kernel_1=kernel1, kernel_2=kernel2, zmin=max(kernels[i].zmin, kernels[j].zmin), zmax=min(kernels[i].zmax, kernels[j].zmax))


def main(ini_par):
    # LOAD POWER in h units
    # =======================
    # noisy = ini_par['noisy']
    cl_limber_z = limber_integrals.cl_limber_z
    noisy = True
    pars = camb.CAMBparams()
    # This function sets up CosmoMC-like settings, with one massive neutrino
    # and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(3500, lens_potential_accuracy=0)
    pars.NonLinear = model.NonLinear_both
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

    # LOAD DNDZ
    # =======================
    # alternative dndz from Sam email

    # res = pk.load(open('/home/manzotti/cosmosis/modules/limber/data_input/DES/des.pkl'))
    # spline = res['spline']
    # N = res['N']
    dndz = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt')
    dndzfun = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1], ext=2)
    norm = scipy.integrate.quad(dndzfun, dndz[0, 0], dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1] / norm, ext='zeros')
    des = gals_kernel.kern(dndz[:, 0], dndzfun, chispline, pars.omegac, h, b=1.)

    # DEFINE KERNELs
    # ======
    # CIB
    # =======
    # j2k = 1.e-6 / np.sqrt(83135.)  # for 353
    lkern = kappa_kernel.kern(z, hspline,
                              chispline, pars.omegac, h, xlss)
    cib = cib_hall.ssed_kern(
        h, z, chispline, hspline, 600e9, b=0.5, jbar_kwargs={'zc': 2.0, 'sigmaz': 2.})

    # ======
    # DESI
    # =======
    desi_dndz = np.loadtxt("/home/manzotti/cosmosis/modules/limber/data_input/DESI/DESI_dndz.txt")
    desi_dndz[:, 1] = np.sum(desi_dndz[:, 1:], axis=1)

    dndzfun_desi = interp1d(desi_dndz[:, 0], desi_dndz[:, 1])
    norm = scipy.integrate.quad(
        dndzfun_desi, desi_dndz[0, 0], desi_dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # normalize
    dndzfun_desi = InterpolatedUnivariateSpline(
        desi_dndz[:, 0], desi_dndz[:, 1] / norm, ext='zeros')
    desi = gals_kernel.kern(desi_dndz[:, 0], dndzfun_desi, hspline, pars.omegac, h, b=1.17)

    # DES bias taken from Giannantonio et
    # DES

    # ======
    # WISE
    # =======
    wise_dn_dz = np.loadtxt('/home/manzotti/galaxies_delensing/wise_dn_dz.txt')
    dndzwise = InterpolatedUnivariateSpline(wise_dn_dz[:, 0], wise_dn_dz[:, 1], k=3, ext='zeros')
    norm = dndzwise.integral(0, 2)
    dndzwise = InterpolatedUnivariateSpline(
        wise_dn_dz[:, 0], wise_dn_dz[:, 1] / norm, ext='zeros')
    # Biased was measured equal to 1 in Feerraro et al. WISE ISW measureament
    # by cross correlating with planck lensing
    wise = gals_kernel.kern(wise_dn_dz[:, 0], dndzwise, hspline, pars.omegac, h, b=1.)

    # Weak lensing

    # ======
    # SKA
    # =======
    z_ska = np.linspace(0.01, 6, 600)
    dndzska10 = gals_kernel.dNdZ_parametric_SKA_10mujk(z_ska)
    dndzska1 = gals_kernel.dNdZ_parametric_SKA_1mujk(z_ska)
    dndzska5 = gals_kernel.dNdZ_parametric_SKA_5mujk(z_ska)
    dndzska01 = gals_kernel.dNdZ_parametric_SKA_01mujk(z_ska)

    # ===
    dndzfun = interp1d(z_ska, dndzska01)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # print(norm)
    # normalize
    dndzska01 = InterpolatedUnivariateSpline(
        z_ska, dndzska01 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=0.1), ext='zeros')
    ska01 = gals_kernel.kern(z_ska, dndzska01, hspline, pars.omegac, h, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska1)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # print(norm)

    # normalize
    dndzska1 = InterpolatedUnivariateSpline(
        z_ska, dndzska1 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=1), ext='zeros')
    ska1 = gals_kernel.kern(z_ska, dndzska1, hspline, pars.omegac, h, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska5)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]

    dndzska5 = InterpolatedUnivariateSpline(
        z_ska, dndzska5 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=5), ext='zeros')
    ska5 = gals_kernel.kern(z_ska, dndzska5, hspline, pars.omegac, h, b=1.)

    # ===
    dndzfun = interp1d(z_ska, dndzska10)
    norm = scipy.integrate.quad(dndzfun, z_ska[0], z_ska[-1], limit=100, epsrel=1.49e-03)[0]

    dndzska10 = InterpolatedUnivariateSpline(
        z_ska, dndzska10 / norm * gals_kernel.dNdZ_SKA_bias(z_ska, mujk=10), ext='zeros')
    ska10 = gals_kernel.kern(z_ska, dndzska10, hspline, pars.omegac, h, b=1.)

    # ======
    # LSST
    # =======
    z_lsst = np.linspace(0.01, 10, 200)
    dndzlsst = gals_kernel.dNdZ_parametric_LSST(z_lsst)
    dndzfun = interp1d(z_lsst, dndzlsst)

    norm = scipy.integrate.quad(dndzfun, 0.01, z_ska[-1], limit=100, epsrel=1.49e-03)[0]
    # used the same bias model of euclid. Find something better
    dndzlsst = InterpolatedUnivariateSpline(
        z_lsst, dndzlsst / norm * 1. * np.sqrt(1. + z_lsst), ext='zeros')
    lsst = gals_kernel.kern(z_lsst, dndzlsst, hspline, pars.omegac, h, b=1.)

    des_weak = kappa_gals_kernel.kern(z_lsst, dndzlsst, chispline, hspline, pars.omegac, h)

    # ======
    # Euclid
    # =======
    z_euclid = np.linspace(0.01, 5, 200)
    z_mean = 0.9
    dndzeuclid = gals_kernel.dNdZ_parametric_Euclid(z_euclid, z_mean)
    # dndzeuclid_deriv = gals_kernel.dNdZ_deriv_Euclid_ana(z_euclid,0.9)
    z_mean_array = np.linspace(0.9 - 0.4, 0.9 + 0.4, 200)
    dndzeuclid_param = RectBivariateSpline(
        z_mean_array, z_euclid, gals_kernel.dNdZ_parametric_Euclid_fulld(z_euclid, z_mean_array))
    dndzfun = interp1d(z_euclid, dndzeuclid)
    # dndzeuclid_deriv_fun = interp1d(
    #     z_euclid, dndzeuclid_param.__call__(z_mean, z_euclid, dx=1, dy=0))

    norm = scipy.integrate.quad(dndzfun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
    # norm_deriv = scipy.integrate.quad(dndzeuclid_deriv_fun, 0.01, 4, limit=100, epsrel=1.49e-03)[0]
    # dndzeuclid_deriv_fun = InterpolatedUnivariateSpline(
    #     z_euclid, dndzeuclid_deriv_fun / norm_deriv * 1. * np.sqrt(1. + z_euclid))h
    dndzeuclid = InterpolatedUnivariateSpline(
        z_euclid, dndzeuclid / norm * 1. * np.sqrt(1. + z_euclid), ext='zeros')
    # bias montanari et all for Euclid https://arxiv.org/pdf/1506.01369.pdf

    euclid = gals_kernel.kern(z_euclid, dndzeuclid, hspline, pars.omegac, h, b=1.)

    # =======
    # Compute Cl implicit loops on ell
    # =======================

    kernels = [lkern, wise, euclid, des_weak, lsst, ska10, ska01, ska5, ska1, cib, desi, des]
    names = ['k', 'wise', 'euclid', 'des_weak', 'lsst', 'ska10',
             'ska01', 'ska5', 'ska1', 'cib', 'desi', 'des']
    # profiler.start()
    # run your program.
    names = names[:2]
    kernels = kernels[:2]
    # labels = []
    # for i in np.arange(0, len(kernels)):
    #     for j in np.arange(i, len(kernels)):
    #         labels.append([names[i], names[j]])

    # out = Parallel(n_jobs=2, verbose=100)(delayed(parallel_limber)(i) for i in labels)

    cls = {}
    for i in np.arange(0, len(kernels)):
        cls[names[i] + names[i]] = [
            limber_integrals.cl_limber_z_ell_parallel(chispline, hspline, rbs, l, kernel_1=kernels[i], zmin=kernels[i].zmin, zmax=kernels[i].zmax) for l in ini_pars['lbins']]

        for j in np.arange(i + 1, len(kernels)):
            print(names[i], names[j])
            print(max(kernels[i].zmin, kernels[j].zmin), min(kernels[i].zmax, kernels[j].zmax))

            cls[names[i] + names[j]] = limber_integrals.cl_limber_z_ell_parallel(chispline, hspline, rbs, ini_pars['lbins'], kernel_1=kernels[
                i], kernel_2=kernels[j], zmin=max(kernels[i].zmin, kernels[j].zmin),
                zmax=min(kernels[i].zmax, kernels[j].zmax))

    # profiler.stop()
    # profiler.run_viewer()

    # cls['cib_fitcib_fit'] = [3500. * (1. * l / 3000.)**(-1.25) for l in lbins]
    # cls['kcib_fit'] = cls['kcib']

    if noisy:
        print('Adding noise')
        # From Planck model, chenged a little bit to match Blake levels
        # print clcib
        cls['cibcib'] = np.array(cls['cibcib']) + 525.
        # cls['cib_fitcib_fit'] = np.array(cls['cib_fitcib_fit']) + 525.

        # from
        cls['ska01ska01'] = np.array(cls['ska01ska01']) + 1. / (183868. * 3282.80635)
        cls['ska1ska1'] = np.array(cls['ska1ska1']) + 1. / (65128. * 3282.80635)
        cls['ska5ska5'] = np.array(cls['ska5ska5']) + 1. / (21235. * 3282.80635)
        cls['ska10ska10'] = np.array(cls['ska10ska10']) + 1. / (11849. * 3282.80635)

        #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
        # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere
        # 4pi rad or 4pi*(180/pi)**2 deg**2

        # from Giannantonio Fosalba
        # galaxy number density of 10 arc min^-2

        arcmin_to_rad = np.pi / 180. / 60
        gal_arcmnin2 = 3.51
        gal_rad_sq = gal_arcmnin2 / arcmin_to_rad**2
        degree_sq = 500
        rad_sq = degree_sq * (np.pi / 180)**2
        # fsky = rad_sq / 4. / np.pi
        n_gal = 3207184.
        print(1. / (n_gal / rad_sq))
        # they mention N=2.1 10^-8 in Fosalba Giann
        nlgg = 1 / gal_rad_sq * np.ones_like(cls['desdes'])

        cls['desdes'] = np.array(cls['desdes']) + nlgg
        # ===============================================
        cls['euclideuclid'] = np.array(cls['euclideuclid']) + (30 / (0.000290888)**2)**(-1)
        cls['lsstlsst'] = np.array(cls['lsstlsst']) + (26 / (0.000290888)**2)**(-1)

    # SAVE
    obj = '_delens'
    section = "limber_spectra"
    np.save(section + obj, cls)
    np.save('ells', ini_pars['lbins'])
    # for i in np.arange(0, len(kernels)):
    #     for j in np.arange(i, len(kernels)):
    #         print(names[i], names[j])
    #         block[section, "cl_" + names[i] + names[j] + obj] = cls[names[i] + names[j]]

    # block[section, "ells_" + obj] = lbins
    # profiler.stop()
    # profiler.run_viewer()
    return ini_pars['lbins'], cls

    # =======================
    # SAVE IN DATABLOCK


if __name__ == "__main__":
    ini_pars = setup(ini_file='./gal_delens_values.ini')
    ells, cls = main(ini_pars)
