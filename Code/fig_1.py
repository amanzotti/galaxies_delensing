
'''

Compute planck CIB (hall method for now) X phi lensing potential



'''


from cosmosis.datablock import names, option_section
import numpy as np
import lensing_cmb_kernel as lens
import hall_CIB_kernel as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d
import sys
import lensing
# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters
distances = names.distances

# TODO


def cl_limber_x(z_chi, hspline, rbs, l, k1, k2=None, xmin=0.0, xmax=13000.):
    """ calculate the cross-spectrum at multipole l between kernels k1 and k2 in the limber approximation. Comoving distance version. See  cl_limber_z for the redshift version.



            Notes: Here everything is assumed in h units. Maybe not the best choice but that is it.

            Args:
              z_chi: z(chi) redshift as a function of comoving distance.
              hspline: H(z). not used here kept to uniform to cl_limber_z
              rbs: Power spectrum spline P(k,z) k and P in h units
              l: angular multipole
              k1: First kernel
              k2: Optional Second kernel otherwise k2=k1
              xmin: Min range of integration, comoving distance
              xmax: Max range of integration, comoving distance


            Returns:

              cl_limber : C_l = \int_chi_min^chi_max d\chi {1/\chi^2} K_A(\chi) K_B(\chi)\times P_\delta(k=l/\chi;z)

               """

    if k2 == None:
        k2 = k1

    def integrand(x):
        z = z_chi(x)
        return 1. / x ** 2 * k1.w_lxz(l, x, z) * k2.w_lxz(l, x, z) * rbs.ev((l + 0.5) / x, z)

    return scipy.integrate.quad(integrand, xmin, xmax, limit=100)[0]


def cl_limber_z(chi_z, hspline, rbs, l, k1, k2=None, zmin=0.0, zmax=1100.):
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

    if k2 == None:
        k2 = k1

    def integrand(z):

        x = chi_z(z)
        return 1. / x ** 2 / hspline(z) * k1.w_lxz(l, x, z) * k2.w_lxz(l, x, z) * rbs.ev((l + 0.5) / x, z)

    return scipy.integrate.quad(integrand, zmin, zmax, limit=100)[0]


def setup(options):

    llmin = options.get_double(option_section, "llmin", default=1.)
    llmax = options.get_double(option_section, "llmax", default=3.5)
    dlnl = options.get_double(option_section, "dlnl", default=.1)
    zmin = options.get_double(option_section, "zmin", default=1e-2)
    zmax = options.get_double(option_section, "zmax",
                              default=8.)

    # Frequency of the CIB
    # you can pass an array but they have to be separated by commas  " , "
    # nu = np.fromstring(nu, dtype=float, sep=',')
    zc = options.get_double(option_section, "zc", default=2.)
    zs = options.get_double(option_section, "zs", default=2.)
    b = options.get_double(option_section, "b", default=1.)

    blockname = options.get_string(option_section, "matter_power", default="matter_power_lin")
    # blockname = options.get_string(option_section, "matter_power", default="matter_power_nl")
    # maybe the suffix for saving the spectra
    # zmax (or take it from CAMB)
    # maybe choose between kappa and others

    print 'llmin = ', llmin
    print 'llmax = ', llmax
    print 'dlnl = ', dlnl
    print 'matter_power = ', blockname
    print ' '

    lbins = np.arange(llmin, llmax, dlnl)
    lbins = 10. ** lbins
    return (lbins, blockname, zmin, zmax, zc, zs, b)


def execute(block, config):
    # Just a simple rename for clarity.
    lbins, blockname, zmin, zmax, zc, zs, b = config

    lbins = np.arange(2, 2000, 15)

    # lbins = np.arange(2, 1000)

    # LOAD POWER in h units
    # =======================

    zpower = block[blockname, "z"]
    kpower = block[blockname, "k_h"]
    powerarray = block[blockname, "p_k"].reshape([np.size(zpower), np.size(kpower)]).T
    rbs = RectBivariateSpline(kpower, zpower, powerarray)

    # Cosmological parameters
    # =======================

    ells = block['cmb_cl', 'ell']
    clee = block['cmb_cl', 'ee']
    clpp = block['cmb_cl', 'PP']
    print ells

    clee *= 2. * np.pi / (ells.astype(float) * (ells.astype(float) + 1.))
    clpp = clpp / (ells.astype(float)) ** 4

    clbb = np.array(lensing.utils.calc_lensed_clbb_first_order(
        lbins, clee, clpp, lmax=ells[-1], nx=1024, dx=2. / 60. / 180. * np.pi).cl, dtype=float)

# TEST DERIVATIVE

    # print clbb.ls, clbb.cl
    # clbb_plus = np.zeros_like(clbb)
    # clbb_minus = np.zeros_like(clbb)
    # clbb_der = np.zeros(np.size(lbins))

    # for i, l in enumerate(lbins):

    #     print i, 'bin from ', lbins[i], 'to', lbins[i + 1]
    #     clpp[lbins[i] - 2:lbins[i + 1] - 2] += clpp[lbins[i] - 2:lbins[i + 1] - 2] * 2.
    #     clbb_plus = np.array(lensing.utils.calc_lensed_clbb_first_order(
    #         lbins, clee, clpp, lmax=ells[-1], nx=1024, dx=2. / 60. / 180. * np.pi).cl, dtype=float)
    # print 'diff= ',clbb_plus-clbb
    # print np.shape(clbb_plus)

    #     clpp[lbins[i] - 2:lbins[i + 1] - 2] -= 2. * clpp[lbins[i] - 2:lbins[i + 1] - 2] * 2.
    #     clbb_minus = np.array(lensing.utils.calc_lensed_clbb_first_order(
    #         lbins, clee, clpp, lmax=ells[-1], nx=1024, dx=2. / 60. / 180. * np.pi).cl, dtype=float)

    #     clbb_stacked = np.vstack((clbb_minus, clbb, clbb_plus))

    #     derivat = np.gradient(clbb_stacked, clpp[i] * 0.1)[0][1] * clpp[i]
    #     print 'deriv', derivat
    #     clbb_der[i] = np.mean(derivat[:100])
    #     print 'der', clbb_der[i]
    #     clpp[lbins[i] - 2:lbins[i + 1] - 2] += clpp[lbins[i] - 2:lbins[i + 1] - 2] * 2.

    # sys.exit()

    # print 'secondo clpp',np.where(clpp<0),clpp
    # print ells, clee, clpp

    clbb = lensing.utils.calc_lensed_clbb_first_order(
        lbins, clee, clpp, lmax=ells[-1], nx=1024, dx=2. / 60. / 180. * np.pi)

    omega_m = block[cosmo, "omega_m"]
    h0 = block[cosmo, "h0"]
    # Distances
    h = block[distances, "h"]
    tmp = h[::-1]
    h = tmp

    zdist = block[distances, "z"]
    tmp = zdist[::-1]  # reverse them so they are going in ascending order
    zdist = tmp
    # =======================

    # Distances
    # =======================
    # reverse them so they are going in ascending order

    d_m = block[distances, "d_m"]
    tmp = d_m[::-1]
    d_m = tmp

    xlss = block[distances, "chistar"]

    # These have dimensions of Mpc; change to h^{-1} Mpc
    d_m *= h0
    h /= h0
    xlss *= h0
    # now in units of h^{-1} Mpc or the inverse
    chispline = interp1d(zdist, d_m)
    z_chi_spline = interp1d(d_m, zdist)
    hspline = interp1d(zdist, h)

    # =======================
    # DEFINE KERNEL
    chs = [(857e9,       1.e-6 / np.sqrt(4.99)),
           (545e9,       1.e-6 / np.sqrt(3391.5)),
           (353e9,       1.e-6 / np.sqrt(83135.)),
           (217e9,       1.e-6 / np.sqrt(231483))]

    chs = [(857e9,       1.e-6 / np.sqrt(4.99))]

    lkern = lens.kern(zdist, omega_m, h0, xlss)

    cl_CIBphi = np.zeros((np.size(lbins), np.shape(chs)[0]))
    cl_CIB = np.zeros((np.size(lbins), np.shape(chs)[0]))
    cl_phi = np.zeros((np.size(lbins), 1))

    # Compute Cl implicit loops on ell
    # =======================

    # for i, (nu, j2k) in enumerate(chs):

    #     print nu
    #     gkern = cib_hall.ssed_kern(h0, zdist, chispline, nu, jbar_kwargs={'zc': 2.0, 'sigmaz': zs})
    # factors does not matter they cancel out, cause we want the correlation factor
    #     cl_CIBphi[:, i] = [
    #         cl_limber_x(z_chi_spline, hspline, rbs, l, gkern, lkern, xmax=chispline(zmax)) for l in lbins]

    #     cl_CIB[:, i] = [cl_limber_x(z_chi_spline, hspline, rbs, l, gkern, gkern, xmax=chispline(
    #         zmax)) + cib_hall.shot_noise_radio(nu) + cib_hall.shot_noise_dusty(nu) for l in lbins]

    # cl_phi = [cl_limber_x(z_chi_spline, hspline, rbs, l, lkern, lkern, xmax=chispline(zmax)) for l in lbins]

    # =======================
    # SAVE INTO DATABLOCK
    obj = 'PHI_CIB'
    section = "limber_spectra"
    block[section, "clCIB_" + obj] = cl_CIB
    block[section, "clCIBphi_" + obj] = cl_CIBphi
    block[section, "clphi_" + obj] = cl_phi
    block[section, "ells_" + obj] = lbins
    block[section, "ells_lbins"] = clbb.ls
    block[section, "ells_clbb"] = np.abs(clbb.specs['cl'])

    return 0
