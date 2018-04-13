'''
This script compute all the spectra needed for the delensing part.
'''
import pyximport
pyximport.install(reload_support=True)

import numpy as np
import kappa_cmb_kernel as kappa_kernel
import gals_kernel
# import kappa_gals_kernel
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d, InterpolatedUnivariateSpline
import limber_integrals
import pickle as pk
import camb
from camb import model
import configparser
from joblib import Parallel, delayed
# from profiling.sampling import SamplingProfiler
# profiler = SamplingProfiler()
import DESI


def setup(ini_file='./gal_delens_values.ini'):
    #  read possible setup options from ini file
    config = configparser.ConfigParser()
    # use configparser for this
    config.read('gal_delens_values.ini')
    # # L BINS
    llmin = config.getfloat('spectra', 'llmin', fallback=0.)
    llmax = config.getfloat('spectra', 'llmax', fallback=3.)
    dlnl = config.getfloat('spectra', "dlnl", fallback=.1)
    noisy = config.getboolean('spectra', "noisy", fallback=True)

    lbins = np.arange(llmin, llmax, dlnl)
    lbins = 10.**lbins
    ini_pars = {}
    ini_pars['lbins'] = lbins
    ini_pars['noisy'] = noisy

    return ini_pars


def parallel_limber(kernels):
    kernel1 = kernels[0]
    kernel2 = kernels[1]
    return cl_limber_z_local(
        chispline,
        hspline,
        rbs,
        l,
        kernel_1=kernel1,
        kernel_2=kernel2,
        zmin=max(kernels[i].zmin, kernels[j].zmin),
        zmax=min(kernels[i].zmax, kernels[j].zmax))


def find_bins(z, dndz, nbins):
    '''
    Function that, given a redshift distribution returns binning with equal number of bins.

    It returns a list of the redshift of the bins.
    '''
    cum = np.cumsum(dndz)
    args = np.hstack(
        (np.array(0.),
         np.searchsorted(
             cum, [cum[-1] / nbins * n
                   for n in np.arange(1, nbins + 1)]))).astype(np.int)
    args[-1] = len(cum) - 1
    return [
        z[args[np.int(i)]:args[np.int(i) + 1]]
        for i in np.arange(0,
                           len(args) - 1.)
    ], args


def make_tomo_bins(z, dndz, sigmaz, width, nbins, hspline, omegac, h, b=1.):
    '''
    this function takes a full dndz distribution and given the number of bins and their width.sigma z returns a list of different tomographic bins
    '''
    lsst_tomo_bins = []
    for n in np.arange(1, nbins + 1):
        dndz_win = dndz * (scipy.special.erfc(
            (width *
             (n - 1) - z) / (sigmaz * np.sqrt(2))) - scipy.special.erfc(
                 (width * n - z) / (sigmaz * np.sqrt(2))))
        dndzlsst_temp = InterpolatedUnivariateSpline(z, dndz_win, ext='raise')
        norm = dndzlsst_temp.integral(z[0], z[-1])
        # print(norm, dndz_win, z)
        dndzlsst_temp_fun = InterpolatedUnivariateSpline(
            z, dndz_win / norm * np.sqrt(1. + z), ext='raise')
        lsst_tomo_bins.append(
            gals_kernel.kern(z, dndzlsst_temp_fun, hspline, omegac, h, b=1.))
    return lsst_tomo_bins


def make_tomo_bins_equal_gals(z, dndz, sigmaz, nbins, hspline, omegac, h,
                              b=1.):
    '''
    this function takes a full dndz distribution and given the number of bins and their width.sigma z returns a list of different tomographic bins
    '''

    z_bins = find_bins(z, dndz, nbins)

    # print('z_bins', z_bins)

    def p_z_ph_z(z_ph, z, sigma_z):
        return np.exp(-(z_ph - z)**2 /
                      (2. * sigma_z**2)) / np.sqrt(2 * np.pi * sigma_z**2)

    # print('print', z, dndz, sigmaz, z_bins)
    lsst_tomo_bins = []
    galaxies_fraction = []
    for n in range(0, len(z_bins[0])):
        # print('n',n,len(z_bins[0]),z[int(z_bins[1][n])], z[int(z_bins[1][n + 1])])
        # print([z_val for i, z_val in enumerate(z)])
        # print(int(z_bins[1][n]),int(z_bins[1][
        #                                          n + 1]))
        photoz_confusion = [
            scipy.integrate.quad(
                p_z_ph_z,
                z[int(z_bins[1][n])],
                z[int(z_bins[1][n + 1])],
                args=(z_val, sigmaz[i]),
                limit=600,
                epsabs=0,
                epsrel=1.49e-03)[0] for i, z_val in enumerate(z)
        ]

        dndz_win = dndz * photoz_confusion
        dndzlsst_temp = InterpolatedUnivariateSpline(z, dndz_win, ext='raise')
        norm = dndzlsst_temp.integral(z[0], z[-1])
        # print(norm, dndz_win, z)
        galaxies_fraction.append(norm)
        dndzlsst_temp_fun = InterpolatedUnivariateSpline(
            z, dndz_win / norm, ext='raise')
        lsst_tomo_bins.append(
            gals_kernel.kern(z, dndzlsst_temp_fun, hspline, omegac, h, b=b))
    return lsst_tomo_bins, np.array(galaxies_fraction)


def make_tomo_bins_equal_gals_lsst(z,
                                   dndz,
                                   sigmaz,
                                   nbins,
                                   hspline,
                                   omegac,
                                   h,
                                   b=1.):
    '''
    this function takes a full dndz distribution and given the number of bins and their width.sigma z returns a list of different tomographic bins
    '''

    z_bins = find_bins(z, dndz, nbins)

    # print('z_bins', z_bins)

    def p_z_ph_z(z_ph, z, sigma_z):
        return np.exp(-(z_ph - z)**2 /
                      (2. * sigma_z**2)) / np.sqrt(2 * np.pi * sigma_z**2)

    # print('print', z, dndz, sigmaz, z_bins)
    lsst_tomo_bins = []
    galaxies_fraction = []
    for n in range(0, len(z_bins[0])):
        # print('n',n,len(z_bins[0]),z[int(z_bins[1][n])], z[int(z_bins[1][n + 1])])
        # print([z_val for i, z_val in enumerate(z)])
        # print(int(z_bins[1][n]),int(z_bins[1][
        #                                          n + 1]))
        photoz_confusion = [
            scipy.integrate.quad(
                p_z_ph_z,
                z[int(z_bins[1][n])],
                z[int(z_bins[1][n + 1])],
                args=(z_val, sigmaz[i]),
                limit=600,
                epsabs=0,
                epsrel=1.49e-03)[0] for i, z_val in enumerate(z)
        ]

        dndz_win = dndz * photoz_confusion
        dndzlsst_temp = InterpolatedUnivariateSpline(z, dndz_win, ext='zeros')
        norm = dndzlsst_temp.integral(z[0], z[-1])
        # print(norm, dndz_win, z)
        galaxies_fraction.append(norm)
        dndzlsst_temp_fun = InterpolatedUnivariateSpline(
            z, dndz_win / norm * np.sqrt(1. + z), ext='zeros')
        lsst_tomo_bins.append(
            gals_kernel.kern(z, dndzlsst_temp_fun, hspline, omegac, h, b=b))
    return lsst_tomo_bins, np.array(galaxies_fraction)


def make_spec_bins(z, dndz_fun, nbins, hspline, omegac, h, b=1.):
    '''
    this takes a dndz distribution and splits it in nbins of equal lenght in z. this is quite rudimentary
    '''
    spec_bins = []
    galaxies_fraction = []
    z_bins = [
        z[i:i + int(len(z) / nbins) + 1]
        for i in range(0, len(z), int(len(z) / nbins))
    ]
    # +1 is inserted not to have gaps
    # print(z_bins)
    for z in z_bins:
        dndz = dndz_fun(z)
        # print('z',z)
        dndz_temp = InterpolatedUnivariateSpline(z, dndz, ext='zeros')
        norm = dndz_temp.integral(z[0], z[-1])
        galaxies_fraction.append(norm)
        # print('norm', norm)
        dndz_bin = InterpolatedUnivariateSpline(z, dndz / norm, ext='zeros')
        # print('here', dndz_bin.integral(z[0], z[-1]))
        spec_bins.append(
            gals_kernel.kern(z, dndz_bin, hspline, omegac, h, b=b))
    # sys.exit()
    return spec_bins, np.array(galaxies_fraction)


def main(ini_par):
    # LOAD POWER in h units
    # =======================
    # noisy = ini_par['noisy']
    # what integral routine to use
    cl_limber_z = limber_integrals.cl_limber_z
    noisy = True

    # SET UP CAMB
    pars = camb.CAMBparams()
    # This function sets up CosmoMC-like settings, with one massive neutrino
    # and helium set using BBN consistency

    pars = camb.CAMBparams()
    # This function sets up CosmoMC-like settings, with one massive neutrino
    # and helium set using BBN consistency
    pars.set_cosmology(
        H0=67.26, ombh2=0.02222, omch2=0.1199, mnu=0.06, omk=0, tau=0.079)
    pars.InitPower.set_params(
        ns=0.96, r=0., nt=0, pivot_tensor=0.01, As=2.1e-9)
    pars.set_for_lmax(5000, lens_potential_accuracy=3)
    # pars.set_for_lmax?

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
        np.linspace(0, 15, 100),
        results.comoving_radial_distance(np.linspace(0, 15, 100)) * h,
        ext=0)
    hspline = InterpolatedUnivariateSpline(
        np.linspace(0, 15, 100), [
            results.hubble_parameter(z_vector) / pars.H0 / 3000.
            for z_vector in np.linspace(0, 15, 100)
        ],
        ext=0)

    # GROWTH

    # growth = InterpolatedUnivariateSpline(np.linspace(0, 15, 100), np.sqrt(
    #     (rbs(0.01, np.linspace(0, 15, 100)) / rbs(0.01, 0)))[0])

    # LOAD DNDZ
    # =======================
    #
    dndz = np.loadtxt(
        '/home/manzotti/cosmosis/modules/limber/data_input/DES/N_z_wavg_spread_model_0.2_1.2_tpz.txt'
    )
    dndzfun = InterpolatedUnivariateSpline(dndz[:, 0], dndz[:, 1], ext=2)
    norm = scipy.integrate.quad(
        dndzfun, dndz[0, 0], dndz[-2, 0], limit=100, epsrel=1.49e-03)[0]
    # print('norm', norm)
    # normalize
    dndzfun = InterpolatedUnivariateSpline(
        dndz[:, 0], dndz[:, 1] / norm, ext='zeros')
    des = gals_kernel.kern(
        dndz[:, 0], dndzfun, chispline, pars.omegac, h, b=1.)
    sigmaz = 0.05 * (dndz[:, 0] + 1.)
    # width = z_lsst[-1] / nbins
    # print(dndz[:, 0].shape, dndz[:, 1].shape)
    des_tomo_bins, galaxies_fraction_des = make_tomo_bins_equal_gals(
        dndz[:, 0],
        dndz[:, 1],
        sigmaz=sigmaz,
        nbins=4,
        hspline=hspline,
        omegac=pars.omegac,
        h=h,
        b=1.)
    # print('frac',galaxies_fraction_des/norm)

    # DEFINE KERNELs

    # ======
    # CIB
    # =======
    # j2k = 1.e-6 / np.sqrt(83135.)  # for 353
    lkern = kappa_kernel.kern(z, hspline, chispline, pars.omegac, h, xlss)
    cib = cib_hall.ssed_kern(
        h,
        z,
        chispline,
        hspline,
        545e9,
        b=0.62,
        jbar_kwargs={
            'zc': 2.0,
            'sigmaz': 2.
        })

    # ======
    # Spitzer
    # =======
    spitzer_dn_dz = np.loadtxt('../Data/spitzer_dndz.txt')
    dndzspitzer = InterpolatedUnivariateSpline(
        spitzer_dn_dz[:, 0], spitzer_dn_dz[:, 1], k=3, ext='zeros')
    norm = dndzspitzer.integral(spitzer_dn_dz[:, 0], spitzer_dn_dz[:, 1])
    dndzspitzer = InterpolatedUnivariateSpline(
        spitzer_dn_dz[:, 0], spitzer_dn_dz[:, 1] / norm, ext='zeros')
    # Biased was measured equal to 1.41 in Ferraro et al. WISE ISW measureament
    # by cross correlating with planck lensing
    spitzer = gals_kernel.kern(
        spitzer_dn_dz[:, 0], dndzspitzer, hspline, pars.omegac, h, b=1.)

    # ======
    # WISE
    # ======

    wise_dn_dz = np.loadtxt('./wise_dn_dz.txt')
    dndzwise = InterpolatedUnivariateSpline(
        wise_dn_dz[:, 0], wise_dn_dz[:, 1], k=3, ext='zeros')
    norm = dndzwise.integral(wise_dn_dz[:, 0], wise_dn_dz[:, 1])
    dndzwise = InterpolatedUnivariateSpline(
        wise_dn_dz[:, 0], wise_dn_dz[:, 1] / norm, ext='zeros')
    # Biased was measured equal to 1.41 in Ferraro et al. WISE ISW measureament
    # by cross correlating with planck lensing
    wise = gals_kernel.kern(
        wise_dn_dz[:, 0], dndzwise, hspline, pars.omegac, h, b=1.41)

    # Weak lensing

    # =======
    # Compute Cl implicit loops on ell
    # =======

    kernels = [lkern, spitzer, wise, cib, des]
    names = ['k', 'spitzer', 'wise', 'cib', 'des']

    # kernels = [lkern, desi]
    # names = ['k', 'desi']
    assert (len(kernels) == len(names))
    # add binned surveys.
    for n, bin_gal in enumerate(des_tomo_bins):
        names.extend(['des_bin{}'.format(int(n))])
        kernels.extend([bin_gal])

    # print(kernels)


# kernels = [lkern, desi1, desi2]
# names = ['k', 'desi1', 'desi2']
    labels = []
    kernel_list = []

    for i in np.arange(0, len(kernels)):
        labels.append(names[i] + names[i])
        kernel_list.append([kernels[i], kernels[i]])
        for j in np.arange(i, len(kernels)):
            # print(i,j,names[i], names[j])
            labels.append(names[i] + names[j])
            kernel_list.append([kernels[i], kernels[j]])
    # print(labels[40:])

    import warnings
    warnings.filterwarnings('error')

    cls_out = Parallel(
        n_jobs=-2, verbose=10)(delayed(limber_integrals.cl_limber_z_ell)(
            chispline,
            hspline,
            rbs,
            ini_pars['lbins'],
            kernel_1=ker[0],
            kernel_2=ker[1],
            zmin=max(ker[0].zmin, ker[1].zmin),
            zmax=min(ker[0].zmax, ker[1].zmax)) for ker in kernel_list)
    cls = {k: v for k, v in zip(labels, cls_out)}
    # print(cls.keys())
    # profiler.stop()
    # profiler.run_viewer()
    noisy = True
    # cls['cib_fitcib_fit'] = [3500. * (1. * l / 3000.)**(-1.25) for l in lbins]
    # cls['kcib_fit'] = cls['kcib']

    if noisy:
        print('Adding noise to all the spectra')
        # From Planck model, chenged a little bit to match Blake levels
        # print clcib
        if 'cibcib' in cls.keys():
            cls['cibcib'] = np.array(cls['cibcib']) + 525.
        # cls['cib_fitcib_fit'] = np.array(cls['cib_fitcib_fit']) + 525.6
        #    150 deg2 for the SV area and 5000 deg2 for the full (5-year) survey.
        # 0.00363618733637157 = 150 0.1212062445 = 5000 rem in total in a sphere
        # 4pi rad or 4pi*(180/pi)**2 deg**2

        # from Giannantonio Fosalba
        # galaxy number density of 10 arc min^-2

        # arcmin_to_rad = np.pi / 180. / 60
        # gal_arcmnin2 = 3.51
        # gal_rad_sq = gal_arcmnin2 / arcmin_to_rad**2
        # degree_sq = 500
        # rad_sq = degree_sq * (np.pi / 180)**2
        # # fsky = rad_sq / 4. / np.pi
        # n_gal = 3207184.
        # print(1. / (n_gal / rad_sq))
        # # they mention N=2.1 10^-8 in Fosalba Giann

        if 'desdes' in cls.keys():
            print('Adding noise to DES')
            nlgg = (8. / (0.000290888)**2)**(-1)
            cls['desdes'] = np.array(cls['desdes']) + nlgg

        for n, fract in enumerate(galaxies_fraction_des):
            print('Adding noise to DES bins')
            name = 'des_bin{}'.format(int(n))
            # equivalent to 3.5 gals per arcmin if you want to compare.
            nlgg = (8. / (0.000290888)**2)**(-1)
            cls[name + name] = cls[name + name] + nlgg / fract

            # ===============================================
        # desi has 0.63 gals per arcmin2
        if 'spitzerspitzer' in cls.keys():
            print('Adding noise to spitzer')
            cls['spitzerspitzer'] = np.array(
                cls['spitzerspitzer']) + (8. / (0.000290888)**2)**(-1)

        # ===============================================
        # desi has 0.63 gals per arcmin2
        if 'desidesi' in cls.keys():
            print('Adding noise to DESI')
            cls['desidesi'] = np.array(
                cls['desidesi']) + (0.63 / (0.000290888)**2)**(-1)

        if 'euclideuclid' in cls.keys():
            print('Adding noise to EUCLID')
            cls['euclideuclid'] = np.array(
                cls['euclideuclid']) + (30 / (0.000290888)**2)**(-1)

        #  from Simone Our conservative masking leaves f sky = 0.47 and about
        # 50 million galaxies.
        if 'wisewise' in cls.keys():
            print('Adding noise to WISE')
            steradians_on_sphere = 4 * np.pi
            fsky = 0.447
            n_gal = 50e6
            gal_per_ster = n_gal / (steradians_on_sphere * fsky)
            cls['wisewise'] = np.array(cls['wisewise']) + 1 / gal_per_ster

    # SAVE
    obj = '_delens_SPT_DEEP'
    section = "limber_spectra"
    import os.path
    if os.path.isfile('../Data/' + section + obj + '.pkl'):

        with open('../Data/' + section + obj + '.pkl', 'rb') as f:
            saved = pk.load(f)

        for key in cls.keys():
            if key in saved.keys():
                print("key", key, "already exist delete to overwrite")
            else:
                print('adding key=', key)
                saved[key] = cls[key]
        with open('../Data/' + section + obj + '.pkl', 'wb') as f:
            print(saved.keys(), 'savings')
            pk.dump(saved, f)

    else:
        with open('../Data/' + section + obj + '.pkl', 'wb') as f:
            pk.dump(cls, f)

    np.save('../Data/' + 'ells', ini_pars['lbins'])
    # profiler.stop()
    # profiler.run_viewer()
    return ini_pars['lbins'], cls  # , galaxies_fraction_lsst

    # =======================
    # SAVE IN DATABLOCK

if __name__ == "__main__":
    ini_pars = setup(ini_file='./gal_delens_values.ini')
    ells, cls = main(ini_pars)
