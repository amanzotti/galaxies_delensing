'''

Module containing the limber integrals definitions used in the PS limber folder

'''

import numpy as np
import scipy.integrate
import sys
from scipy.interpolate import  InterpolatedUnivariateSpline
from joblib import Parallel, delayed

def cl_limber_x(z_chi, p_kz, l, k1, k2=None, xmin=0.0, xmax=13000.):
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

        def integrand(double x):
            cdef double z
            z = z_chi(x)
            return 1. / x /x * k1.w_lxz(l, x, z)**2 * p_kz(l / x, z)

    else:

        def integrand(double x):
            cdef double z
            z = z_chi(x)
            return 1. / x /x * k1.w_lxz(l, x, z) * k2.w_lxz(l, x, z) * p_kz(l / x, z)


    return scipy.integrate.quad(integrand, xmin, xmax, limit=300, epsrel=1.49e-06)[0]


def  cl_limber_z(chi_z, hspline, rbs, l, kernel_1, kernel_2=None,  zmin=0.0,  zmax=1100.):
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
    if kernel_2 == None:
        kernel_2 = kernel_1

        def integrand(double z):
            cdef double x,h,k1,pk
            x = chi_z(z)
            h =hspline(z)
            k1 = kernel_1.w_lxz(l, x, z)**2
            pk=rbs.ev((l + 0.5) / x, z)
            return 1. / x/x * h * k1 * pk

    else:

        def integrand(double z):
            cdef double x,h,k1,k2,pk
            x = chi_z(z)
            h =hspline(z)
            k1 = kernel_1.w_lxz(l, x, z)
            k2 = kernel_2.w_lxz(l, x, z)

            pk=rbs.ev((l + 0.5) / x, z)
            return 1. / x /x * h * k1 * k2 * pk

    # print 'here' ,integrand(0.5)

    # sys.exit()
    # func = InterpolatedUnivariateSpline(np.linspace(zmin,zmax,100), np.vectorize(integrand)(np.linspace(zmin,zmax,100)), ext='zeros')
    # print('')
    # print(func(zmax-(zmax-zmin)/2.))
    # print(scipy.integrate.quad(integrand, zmin, zmax, limit=300, epsrel=1.49e-06)[0])
    return scipy.integrate.quad(integrand, zmin, zmax, limit=300, epsrel=1.49e-06)[0]




def integrand_auto(double z,double l, rbs,chi_z,hspline,kernel_1):
    cdef double x,h,k1,pk
    x = chi_z(z)
    h =hspline(z)
    k1 = kernel_1.w_lxz(l, x, z)**2
    pk=rbs.ev((l + 0.5) / x, z)
    return 1. / x/x * h * k1 * pk

def integrand_cross(double z,double l,rbs,chi_z,hspline,kernel_1,kernel_2):
    cdef double x,h,k1,k2,pk
    x = chi_z(z)
    h =hspline(z)
    k1 = kernel_1.w_lxz(l, x, z)
    k2 = kernel_2.w_lxz(l, x, z)
    pk=rbs.ev((l + 0.5) / x, z)
    return 1. / x /x * h * k1 * k2 * pk


def integratio_auto(ell,rbs,chi_z,hspline,kernel_1,integrand, zmin, zmax):
  return scipy.integrate.quad(integrand_auto, zmin, zmax, args=(ell,rbs,chi_z,hspline,kernel_1), limit=300, epsrel=1.49e-06)[0]

def integratio_cross(ell,rbs,chi_z,hspline,kernel_1,kernel_2,integrand, zmin, zmax):
  return scipy.integrate.quad(integrand_cross, zmin, zmax, args=(ell,rbs,chi_z,hspline,kernel_1,kernel_2), limit=300, epsrel=1.49e-06)[0]




def  cl_limber_z_ell_parallel(chi_z, hspline, rbs, lbins, kernel_1, kernel_2=None,  zmin=0.0,  zmax=1100.):


  '''
  As the one before but now different ells are computed in parallel
  '''


    if kernel_2 == None:
        kernel_2 = kernel_1
        return Parallel(n_jobs=-2, verbose=0)(delayed(integratio_auto)(i,rbs,chi_z,hspline,kernel_1,integrand_auto, zmin, zmax)  for i in lbins)

    else:
        return Parallel(n_jobs=-2, verbose=0)(delayed(integratio_cross)(i,rbs,chi_z,hspline,kernel_1,kernel_2,integrand_cross, zmin, zmax)  for i in lbins)

