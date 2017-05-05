PROGRAM compute_noisebias
! A for computing "noisebias," Nl, of the reconstructed lensing potential:
!
! 1/Nl = 1/(2l+1) * sum_{l2=l}^{lmax} * sum_{l3=l2}^{lmax}
!                 * A_{l1l2l3}^2/[C(l2)C(l3)Delta(l1,l2,l3)].
!
! REF: Lewis, Challinor & Hanson, JCAP 03, 018 (2011), arXiv:1101.2234
! April 25, 2012: E.Komatsu

  IMPLICIT none
  character(len=128) :: filename
  integer :: lmax=3000 ! maximum multipole
  integer :: lmin=2    ! minimum multipole
  integer :: i,l,l1,l2,l3,info,dum
  double precision :: l3min,l3max,gaunt,fourpi,twopi,delta
  double precision :: Al1l2l3,noisebias
  double precision, allocatable, dimension(:) :: cl,wigner3j
  print*,'lmin=',lmin
  print*,'lmax=',lmax
  twopi=2d0*3.1415926535d0
  fourpi=4d0*3.1415926535d0
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
! begin computing noisebias
  allocate(wigner3j(2*lmax+1))
  open(3,file='multipole_noisebias.txt',status='unknown')
  do l1=lmin,lmax
     noisebias=0d0
     do l2=l1,lmax
        call DRC3JJ(dble(l1),dble(l2),0d0,0d0, &
             l3min,l3max,wigner3j,2*lmax+1,info)
        if ((int(l3max).ge.l2)) then
           do l3=l2,int(l3max)
              if ((l3.le.lmax).and.(mod(l1+l2+l3,2).eq.0))then
                 gaunt=wigner3j(l3-int(l3min)+1)/dsqrt(fourpi) &
                      *dsqrt(dble(2*l1+1))*dsqrt(dble(2*l2+1)) &
                      *dsqrt(dble(2*l3+1))
                 Al1l2l3=0.5d0*gaunt*( &
                      (l3*(l3+1d0)-l2*(l2+1d0)+l1*(l1+1d0))*cl(l3)  &
                      +(l2*(l2+1d0)-l3*(l3+1d0)+l1*(l1+1d0))*cl(l2) &
                      )
                 noisebias=noisebias+Al1l2l3**2d0/cl(l2)/cl(l3)/delta(l1,l2,l3)
              endif
           enddo
        endif
     enddo
     write(3,'(1I5,1E18.8)')l1,dble(2*l1+1)/noisebias
  enddo
  close(3)
  deallocate(cl,wigner3j)
END PROGRAM compute_noisebias
!---------------------------------------------------------
double precision function delta(l1,l2,l3)
! delta=6 for l1=l2=l3; delta=2 for l1=l2, l2=l3, or l3=l1; delta=1 otherwise
  implicit none
  integer, intent(IN) :: l1,l2,l3
  if((l1.eq.l2).and.(l2.eq.l3))then
     delta=6d0
  elseif((l1.eq.l2).or.(l2.eq.l3).or.(l3.eq.l1))then
     delta=2d0
  else
     delta=1d0
  endif
  return
END FUNCTION delta
