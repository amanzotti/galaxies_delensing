PROGRAM compute_chi2_lensingiswpowerspectrum
! A code for computing the full-sky, cosmic-variance only chi-squared of the 
! lensing-ISW power spectrum, including variance from the lensing signal itself.
!
! Chi-squared is computed as
!
! Chi^2 = sum_l [Q(l)]^2/{C(l)*(N(l)+Ctheta(l))+[Q(l)]^2}
!
! where
! - C(l) is the lensed CMB temperature power spectrum
! - Q(l) is the lensing-ISW cross-power spectrum
!   [more precisely, this is the lensing-temperature power spectrum, as it 
!    includes all the linear effects (not just the late-time ISW)]
! - Ctheta(l) is the power spectrum of lensing potential
!
! all of which are computed from CAMB, and
!
! - N(l) is the noisebias of the reconstructed lensing potential 
!   (computed by "compute_noise" in this package)
!
! REF: Lewis, Challinor & Hanson, JCAP 03, 018 (2011), arXiv:1101.2234
! April 25, 2012: E.Komatsu 
  IMPLICIT none
  character(len=128) :: filename
  integer :: lmax=1500 ! maximum multipole
  integer :: lmin=2    ! minimum multipole
  integer :: i,l,info,dum
  double precision :: twopi,dummy
  double precision :: chi2a,chi2b,chi2c,chi2d
  double precision, allocatable, dimension(:) :: cl,nl,ctheta,ql
  print*,'lmin=',lmin
  print*,'lmax=',lmax
  twopi=2d0*3.1415926535d0
! read in noisebias, N(l)
  filename='multipole_noisebias.txt'
  print*,'noisebias file= ',trim(filename)
  open(1,file=filename,status='old')
  allocate(nl(lmax))
  do i=2,lmax
     read(1,*)dum,nl(i)
  enddo
  close(1)
! read in lens-temperature cross power spectrum and lensing potential power
! spectrum
  filename='wmap5baosn_max_likelihood_lenspotentialCls.dat'
  print*,'Q(l) and C^theta_l file= ',trim(filename)
  open(1,file=filename,status='old')
  allocate(ql(lmax),ctheta(lmax))
  do i=2,lmax
     read(1,*)dum,dummy,dummy,dummy,dummy,ctheta(i),ql(i),dummy
     ctheta(i)=ctheta(i)/dble(i*(i+1))**2d0*twopi
     ql(i)=ql(i)/dble(i*(i+1))**1.5d0*twopi/2.726d6
  enddo
  close(1)
! read in lensed Cl
  allocate(cl(lmax))
  filename='wmap5baosn_max_likelihood_lensedCls.dat'
  print*,'cl file (lensed)= ',trim(filename)
  open(2,file=filename,status='old')
  do l=2,lmax
     read(2,*)dum,cl(l)
     cl(l)=cl(l)/dble(l)/(dble(l)+1d0)*twopi/(2.726d6)**2d0
  enddo
  close(2)
! begin computing chi^2
  chi2a=0d0
  chi2b=0d0
  chi2c=0d0
  chi2d=0d0
  do l=lmin,lmax
     chi2a=chi2a+dble(2*l+1)*ql(l)**2d0/(cl(l)*(ctheta(l)+nl(l))+ql(l)**2d0)
     chi2b=chi2b+dble(2*l+1)*ql(l)**2d0/(cl(l)*(ctheta(l)+nl(l)))
     chi2c=chi2c+dble(2*l+1)*ql(l)**2d0/(cl(l)*nl(l)+ql(l)**2d0)
     chi2d=chi2d+dble(2*l+1)*ql(l)**2d0/(cl(l)*nl(l))
  enddo
  print*,'chi2(all included)=',chi2a
  print*,'chi2(Q(l) in covariance ignored)=',chi2b
  print*,'chi2(Ctheta in covariance ignored)=',chi2c
  print*,'chi2(Ctheta and Q(l) in covariance ignored)=',chi2d
  print*,'sqrt(chi2)=',dsqrt(chi2a),dsqrt(chi2b),dsqrt(chi2c),dsqrt(chi2d)
  deallocate(ql,cl,ctheta,nl)
END PROGRAM compute_chi2_lensingiswpowerspectrum
